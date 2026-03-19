import asyncio
import json
import os
import random

# Disable litellm's background LoggingWorker entirely.
# litellm (<=1.78) has a GLOBAL_LOGGING_WORKER singleton with an asyncio.Queue
# that gets bound to the first event loop that touches it. When multiple
# AlgorithmRunner threads each spin up their own event loops (for SlowBurnLLM),
# the Queue raises "bound to a different event loop" from the second thread on.
# See: https://github.com/BerriAI/litellm/issues/17813
#      https://github.com/BerriAI/litellm/issues/14521
#
# Replacing the module-level GLOBAL_LOGGING_WORKER variable is NOT sufficient
# because litellm.utils and litellm.caching.caching_handler both do
# `from ... import GLOBAL_LOGGING_WORKER` at import time, holding their own
# stale reference. The only fix that reaches all call sites is to patch the
# CLASS METHOD so every instance (past and future) becomes a no-op.
import warnings
from typing import Any, Dict, List, Optional, Tuple

import litellm
from concurry import CallLimit, LimitSet, RateLimit, ResourceLimit, Worker
from concurry.core.limit.limit_set import BaseLimitSet
from morphic import validate
from morphic.typed import format_exception_msg
from slowburn import SlowBurnLLM

warnings.filterwarnings(
    "ignore", message="coroutine .* was never awaited", category=RuntimeWarning
)

try:
    from litellm.litellm_core_utils.logging_worker import LoggingWorker

    def _noop_enqueue(self, coroutine=None, async_coroutine=None):
        coro = coroutine or async_coroutine
        if coro is not None:
            coro.close()

    LoggingWorker.ensure_initialized_and_enqueue = _noop_enqueue
    LoggingWorker.start = lambda self: None
    LoggingWorker.enqueue = _noop_enqueue
except (ImportError, AttributeError):
    pass

from prompt_moo.algorithm import GPO, OPRO, TextGrad
from prompt_moo.config import promptmoo_config
from prompt_moo.data_input import Dataset
from prompt_moo.data_structures import Task
from prompt_moo.prompt_template_utils import PromptTemplate
from prompt_moo.task_predictor import parse_task_response


def parse_task_response_retry_until(result: str, **context) -> bool:
    """Retry until the task response is valid JSON."""
    try:
        parse_task_response(result)
        return True
    except Exception:
        return False


LLM_CONFIGS = {
    "llama3.1": {
        "task_model": "openrouter/meta-llama/llama-3.1-8b-instruct",
        "other_model": "openrouter/meta-llama/llama-3.1-70b-instruct",
        "reasoning": False,
        "provider_order": {
            "openrouter/meta-llama/llama-3.1-8b-instruct": [
                "novita/fp8",
                "nebius/fp8",
                "deepinfra/bf16",
                "nebius/fast",
            ],
            "openrouter/meta-llama/llama-3.1-70b-instruct": [
                "novita/fp8",
                "nebius/fp8",
                "deepinfra/bf16",
                "nebius/fast",
            ],
        },
    },
    "qwen3.5": {
        "task_model": "openrouter/qwen/qwen3.5-9b",
        "other_model": "openrouter/qwen/qwen3.5-397b-a17b",
        "reasoning": False,
        "provider_order": {
            "openrouter/qwen/qwen3.5-9b": ["venice/fp8", "together"],
            "openrouter/qwen/qwen3.5-397b-a17b": [
                "atlas-cloud/fp8",
                "parasail/fp8",
                "alibaba",
                "novita",
                "together",
            ],
        },
    },
    "qwen3": {
        "task_model": "openrouter/qwen/qwen3-8b",
        "other_model": "openrouter/qwen/qwen3-235b-a22b-2507",
        "reasoning": False,
        "provider_order": {
            "openrouter/qwen/qwen3-8b": ["alibaba", "atlas-cloud/fp8"],
            "openrouter/qwen/qwen3-235b-a22b-2507": [
                "google-vertex",
                "parasail/fp8",
                "alibaba",
                "wandb/bf16",
                "together",
                "atlas-cloud/fp8",
                "novita/fp8",
                "friendli",
            ],
        },
    },
    "gpt5": {
        "task_model": "openrouter/openai/gpt-5-nano",
        "other_model": "openrouter/openai/gpt-5.2",
        "reasoning": False,
        "provider_order": {
            "openrouter/openai/gpt-5-nano": ["openai", "azure"],
            "openrouter/openai/gpt-5.2": ["openai", "azure"],
        },
    },
    "claude4.5": {
        "task_model": "openrouter/anthropic/claude-haiku-4.5",
        "other_model": "openrouter/anthropic/claude-sonnet-4.5",
        "reasoning": False,
        "provider_order": {
            "openrouter/anthropic/claude-haiku-4.5": [
                "google-vertex",
                "anthropic",
                "amazon-bedrock",
            ],
            "openrouter/anthropic/claude-sonnet-4.5": [
                "google-vertex/global",
                "google-vertex",
                "anthropic",
                "amazon-bedrock",
            ],
        },
    },
}

REASONING_EXTRA_TOKENS = 2000

QWEN_NO_THINK_SUFFIX = "\n/no_think"


def _detect_reasoning_family(model_name: str) -> Optional[str]:
    """Detect the reasoning parameter family from the model name."""
    if "/qwen/" in model_name:
        return "qwen"
    if "/openai/" in model_name:
        return "openai"
    if "/anthropic/" in model_name:
        return "anthropic"
    return None


def _build_litellm_params(
    *,
    providers: Optional[List[str]],
    model_name: str,
    reasoning: bool,
) -> Dict[str, Any]:
    """Build the litellm_params dict with provider prefs and reasoning config.

    For Qwen models with reasoning disabled, we send three complementary signals:
    1. ``enable_thinking: false`` as a top-level extra_body param (OpenRouter native).
    2. ``chat_template_kwargs: {"enable_thinking": false}`` for vLLM/TGI backends.
    3. ``litellm.drop_params = True`` is already set in SlowBurnLLM, so providers
       that don't recognize these params will silently ignore them.

    Args:
        providers: Provider order list for OpenRouter routing.
        model_name: Full litellm model identifier (used to detect family).
        reasoning: Whether to enable reasoning/thinking.
    """
    extra_body: Dict[str, Any] = {}

    if providers is not None:
        ## Shuffle the providers list
        random.shuffle(providers)
        extra_body["provider"] = {"order": providers, "allow_fallbacks": True}

    family = _detect_reasoning_family(model_name)

    if family == "qwen":
        extra_body["enable_thinking"] = reasoning
        extra_body["chat_template_kwargs"] = {"enable_thinking": reasoning}
    elif family == "openai":
        extra_body["reasoning"] = {"effort": "medium" if reasoning else "none"}
    elif family == "anthropic":
        extra_body["reasoning"] = {"effort": "medium" if reasoning else "none"}

    if len(extra_body) == 0:
        return {}
    return {"extra_body": extra_body}


def _get_prompt_suffix(*, model_name: str, reasoning: bool) -> str:
    """Return a prompt suffix to append to every user message for this worker.

    For Qwen models with reasoning disabled, returns the ``/no_think`` token
    as a belt-and-suspenders complement to the API-level ``enable_thinking: false``.
    Some OpenRouter providers ignore the API param; the in-message token is
    always respected by the Qwen chat template.
    """
    if not reasoning and _detect_reasoning_family(model_name) == "qwen":
        return QWEN_NO_THINK_SUFFIX
    return ""


def _stamp_prompt_suffix(llm: SlowBurnLLM, *, model_name: str, reasoning: bool) -> None:
    """Set ``_prompt_suffix`` on a SlowBurnLLM worker instance.

    Pipeline components read this via ``get_prompt_suffix(llm_pool)`` and
    append it to every user message before calling ``call_llm_batch``.
    """
    suffix = _get_prompt_suffix(model_name=model_name, reasoning=reasoning)
    object.__setattr__(llm, "_prompt_suffix", suffix)


def get_prompt_suffix(llm_pool: Any) -> str:
    """Read the prompt suffix stored on an LLM worker (empty string if none).

    Re-exported from ``prompt_moo.llm_utils`` for convenience.
    """
    from prompt_moo.llm_utils import get_prompt_suffix as _get

    return _get(llm_pool)


def _build_retry_config(*, cfg: Any) -> Dict[str, Any]:
    """Build retry config dict shared by all LLM factory functions.

    Args:
        cfg: PromptMOODefaults instance.

    Returns:
        Dict of retry-related kwargs for SlowBurnLLM.options().
    """
    return dict(
        num_retries={"call_llm": cfg.num_retries, "*": 0},
        retry_wait={"call_llm": cfg.retry_wait, "*": 1},
        retry_algorithm={"call_llm": cfg.retry_algorithm, "*": "Exponential"},
        retry_jitter={"call_llm": cfg.retry_jitter, "*": 0},
        retry_on={
            "call_llm": [
                ValueError,
                asyncio.TimeoutError,
                litellm.Timeout,
                litellm.APIError,
                litellm.APIConnectionError,
                litellm.BadRequestError,
                litellm.InternalServerError,
                litellm.RateLimitError,
                litellm.ServiceUnavailableError,
            ],
            "*": [],
        },
    )


def create_shared_limits() -> BaseLimitSet:
    """Create shared LimitSet for all LLM workers.

    Reads capacities from promptmoo_config.defaults at call time.

    Returns:
        LimitSet configured for rate limiting across all LLM workers.
    """
    cfg = promptmoo_config.defaults
    return LimitSet(
        limits=[
            ResourceLimit(key="parallel_calls", capacity=cfg.max_parallel_calls),
            CallLimit(window_seconds=60, capacity=cfg.max_rpm),
            RateLimit(
                key="input_tokens", window_seconds=60, capacity=cfg.max_input_tpm
            ),
            RateLimit(
                key="output_tokens", window_seconds=60, capacity=cfg.max_output_tpm
            ),
        ],
        mode="asyncio",
        shared=True,
    )


@validate
def create_task_llm(
    *, llm: str, api_key: str, limits: BaseLimitSet, reasoning: bool = False
) -> SlowBurnLLM:
    """Create task LLM using SlowBurnLLM.

    Args:
        llm: LLM family key in LLM_CONFIGS.
        api_key: API key for LLM service.
        limits: Shared LimitSet for rate limiting.
        reasoning: Enable reasoning/thinking mode. Adds REASONING_EXTRA_TOKENS to max_tokens.
    """
    if llm not in LLM_CONFIGS:
        raise ValueError(f"Unknown LLM: {llm}. Options: {list(LLM_CONFIGS.keys())}")

    cfg = promptmoo_config.defaults
    config = LLM_CONFIGS[llm]
    model_name = config["task_model"]
    providers = config["provider_order"].get(model_name)
    reasoning = config["reasoning"]
    max_tokens = cfg.task_llm_max_tokens
    if reasoning and _detect_reasoning_family(model_name) is not None:
        max_tokens += REASONING_EXTRA_TOKENS

    llm = SlowBurnLLM.options(
        mode="asyncio",
        limits=limits,
        **_build_retry_config(cfg=cfg),
    ).init(
        name="task_llm",
        model_name=model_name,
        api_key=api_key,
        temperature=cfg.task_llm_temperature,
        max_tokens=max_tokens,
        timeout=cfg.task_llm_timeout,
        litellm_params=_build_litellm_params(
            providers=providers,
            model_name=model_name,
            reasoning=reasoning,
        ),
    )
    _stamp_prompt_suffix(llm, model_name=model_name, reasoning=reasoning)
    return llm


@validate
def create_optimizer_llm(
    *, llm: str, api_key: str, limits: BaseLimitSet
) -> SlowBurnLLM:
    """Create optimizer LLM using SlowBurnLLM.

    Args:
        llm: LLM family key in LLM_CONFIGS.
        api_key: API key for LLM service.
        limits: Shared LimitSet for rate limiting.
    """
    if llm not in LLM_CONFIGS:
        raise ValueError(f"Unknown LLM: {llm}. Options: {list(LLM_CONFIGS.keys())}")

    cfg = promptmoo_config.defaults
    config = LLM_CONFIGS[llm]
    model_name = config["other_model"]
    providers = config["provider_order"].get(model_name)
    reasoning = config["reasoning"]
    max_tokens = cfg.optimizer_llm_max_tokens
    if reasoning and _detect_reasoning_family(model_name) is not None:
        max_tokens += REASONING_EXTRA_TOKENS

    llm = SlowBurnLLM.options(
        mode="asyncio",
        limits=limits,
        **_build_retry_config(cfg=cfg),
    ).init(
        name="optimizer_llm",
        model_name=model_name,
        api_key=api_key,
        temperature=cfg.optimizer_llm_temperature,
        max_tokens=max_tokens,
        timeout=cfg.optimizer_llm_timeout,
        litellm_params=_build_litellm_params(
            providers=providers,
            model_name=model_name,
            reasoning=reasoning,
        ),
    )
    _stamp_prompt_suffix(llm, model_name=model_name, reasoning=reasoning)
    return llm


@validate
def create_gradient_llm(*, llm: str, api_key: str, limits: BaseLimitSet) -> SlowBurnLLM:
    """Create gradient LLM using SlowBurnLLM.

    Args:
        llm: LLM family key in LLM_CONFIGS.
        api_key: API key for LLM service.
        limits: Shared LimitSet for rate limiting.
    """
    if llm not in LLM_CONFIGS:
        raise ValueError(f"Unknown LLM: {llm}. Options: {list(LLM_CONFIGS.keys())}")

    cfg = promptmoo_config.defaults
    config = LLM_CONFIGS[llm]
    model_name = config["other_model"]
    providers = config["provider_order"].get(model_name)
    reasoning = config["reasoning"]
    max_tokens = cfg.gradient_llm_max_tokens
    if reasoning and _detect_reasoning_family(model_name) is not None:
        max_tokens += REASONING_EXTRA_TOKENS

    llm = SlowBurnLLM.options(
        mode="asyncio",
        limits=limits,
        **_build_retry_config(cfg=cfg),
    ).init(
        name="gradient_llm",
        model_name=model_name,
        api_key=api_key,
        temperature=cfg.gradient_llm_temperature,
        max_tokens=max_tokens,
        timeout=cfg.gradient_llm_timeout,
        litellm_params=_build_litellm_params(
            providers=providers,
            model_name=model_name,
            reasoning=reasoning,
        ),
    )
    _stamp_prompt_suffix(llm, model_name=model_name, reasoning=reasoning)
    return llm


@validate
def create_loss_llm(*, llm: str, api_key: str, limits: BaseLimitSet) -> SlowBurnLLM:
    """Create loss LLM using SlowBurnLLM.

    Args:
        llm: LLM family key in LLM_CONFIGS.
        api_key: API key for LLM service.
        limits: Shared LimitSet for rate limiting.
    """
    if llm not in LLM_CONFIGS:
        raise ValueError(f"Unknown LLM: {llm}. Options: {list(LLM_CONFIGS.keys())}")

    cfg = promptmoo_config.defaults
    config = LLM_CONFIGS[llm]
    model_name = config["other_model"]
    providers = config["provider_order"].get(model_name)
    reasoning = config["reasoning"]
    max_tokens = cfg.loss_llm_max_tokens
    if reasoning and _detect_reasoning_family(model_name) is not None:
        max_tokens += REASONING_EXTRA_TOKENS

    llm = SlowBurnLLM.options(
        mode="asyncio",
        limits=limits,
        **_build_retry_config(cfg=cfg),
    ).init(
        name="loss_llm",
        model_name=model_name,
        api_key=api_key,
        temperature=cfg.loss_llm_temperature,
        max_tokens=max_tokens,
        timeout=cfg.loss_llm_timeout,
        litellm_params=_build_litellm_params(
            providers=providers,
            model_name=model_name,
            reasoning=reasoning,
        ),
    )
    _stamp_prompt_suffix(llm, model_name=model_name, reasoning=reasoning)
    return llm


# Dataset configurations
DATASET_CONFIGS = {
    "SummEval": {
        "prompt_prefix": "Evaluate the summary. Output JSON with the requested metric scores. Do NOT include reasoning or explanations. Each metric should contain a single integer. Formats like '4/5' or '4|5' are invalid.",
        "task_output_formats": {
            "fluency": "An integer between 1 to 5",
            "coherence": "An integer between 1 to 5",
            "relevance": "An integer between 1 to 5",
            "consistency": "An integer between 1 to 5",
        },
        "task_losses": {
            "fluency": "accuracy",
            "coherence": "accuracy",
            "relevance": "accuracy",
            "consistency": "accuracy",
        },
    },
    "WildGuard": {
        "prompt_prefix": "Evaluate the prompt and model response. Output JSON only. Do NOT include reasoning or explanations.",
        "task_output_formats": {
            "prompt_harm_label": '"unharmful" | "harmful"',
            "response_harm_label": '"unharmful" | "harmful"',
            "response_refusal_label": '"compliance" | "refusal"',
        },
        "task_losses": {
            "prompt_harm_label": "accuracy",
            "response_harm_label": "accuracy",
            "response_refusal_label": "accuracy",
        },
    },
    "BRIGHTER": {
        "prompt_prefix": "Evaluate the emotion intensities in the text. Output JSON with intensity scores 0-3. Do NOT include reasoning or explanations. Each entry of anger, fear, joy, sadness, surprise should contain a single integer between 0 and 3. So entries like '0/3' or '0|3' or '0.5' are invalid.",
        "task_output_formats": {
            "anger": "An integer between 0 to 3",
            "fear": "An integer between 0 to 3",
            "joy": "An integer between 0 to 3",
            "sadness": "An integer between 0 to 3",
            "surprise": "An integer between 0 to 3",
        },
        "task_losses": {
            "anger": "accuracy",
            "fear": "accuracy",
            "joy": "accuracy",
            "sadness": "accuracy",
            "surprise": "accuracy",
        },
    },
}


@validate
def build_prompt_skeleton(
    *,
    dataset_name: str,
    tasks: List[Task],
    task_output_formats: Optional[Dict[str, str]] = None,
) -> str:
    """Build prompt skeleton dynamically based on selected tasks.

    Args:
        dataset_name: Name of the dataset
        tasks: List of tasks to include in the prompt
        task_output_formats: Optional dict mapping task names to output format specs.
            If not provided, will use DATASET_CONFIGS.

    Returns:
        Complete prompt skeleton with dynamic JSON format section
    """
    config = DATASET_CONFIGS[dataset_name]
    prompt_prefix = config["prompt_prefix"]

    if task_output_formats is None:
        task_output_formats = config["task_output_formats"]

    task_names = [task.task_name for task in tasks]
    if len(task_names) == 1:
        task_list_str = f"Output ONLY the '{task_names[0]}' metric."
    else:
        task_list_str = f"Output the following metrics: {', '.join(task_names)}."

    json_lines = []
    for task in tasks:
        task_name = task.task_name
        if task_name not in task_output_formats:
            raise ValueError(
                f"Task '{task_name}' not found in output formats for dataset '{dataset_name}'"
            )
        output_format = task_output_formats[task_name]
        json_lines.append(f'  "{task_name}": {output_format}')

    json_format = "{\n" + ",\n".join(json_lines) + "\n}"

    skeleton = f"""{prompt_prefix}
{task_list_str}
Output format (follow this EXACTLY):
{json_format}
"""

    return skeleton


@validate
def get_initial_prompt(
    *,
    dataset_name: str,
    tasks: List[Task],
    task_output_formats: Optional[Dict[str, str]] = None,
) -> PromptTemplate:
    """Get initial prompt for a dataset with specified tasks.

    Args:
        dataset_name: Name of the dataset
        tasks: List of tasks to include in the prompt
        task_output_formats: Optional dict mapping task names to output format specs.

    Returns:
        PromptTemplate configured for the specified tasks
    """
    skeleton = build_prompt_skeleton(
        dataset_name=dataset_name,
        tasks=tasks,
        task_output_formats=task_output_formats,
    )
    return PromptTemplate.of(
        "multi",
        skeleton=skeleton,
        instruction={t.task_name: t.task_instruction for t in tasks},
        tasks=tasks,
    )


@validate
def get_task_losses(
    *, dataset_name: str, tasks: Optional[List[Task]] = None
) -> Dict[str, str]:
    """Get task losses for a dataset.

    Args:
        dataset_name: Name of the dataset
        tasks: Optional list of tasks to filter losses for.

    Returns:
        Dict mapping task names to loss function names
    """
    all_losses = DATASET_CONFIGS[dataset_name]["task_losses"]
    if tasks is not None:
        task_names = {t.task_name for t in tasks}
        return {k: v for k, v in all_losses.items() if k in task_names}
    return all_losses


@validate
def get_single_task_prompt(
    *,
    task: Task,
    dataset_name: str,
    task_output_formats: Optional[Dict[str, str]] = None,
) -> PromptTemplate:
    """Get initial prompt for a single task.

    Args:
        task: The task to create a prompt for
        dataset_name: Name of the dataset
        task_output_formats: Optional dict mapping task names to output format specs.

    Returns:
        PromptTemplate configured for this single task
    """
    skeleton = build_prompt_skeleton(
        dataset_name=dataset_name,
        tasks=[task],
        task_output_formats=task_output_formats,
    )
    return PromptTemplate.of(
        "multi",
        skeleton=skeleton,
        instruction={task.task_name: task.task_instruction},
        tasks=[task],
    )


def find_last_prompt(output_dir: str) -> Tuple[Optional[int], Optional[str]]:
    """Find the latest saved prompt in an output directory."""
    prompts_dir = os.path.join(output_dir, "prompts")
    if not os.path.exists(prompts_dir):
        return None, None

    for i in range(100, -1, -1):
        prompt_path = os.path.join(prompts_dir, f"step_{i}_new.txt")
        if os.path.exists(prompt_path):
            with open(prompt_path, "r") as f:
                return i, f.read()
    return None, None


def check_run_status(output_dir: str) -> Dict[str, Any]:
    """Check the status of a run based on its output files."""
    summary_path = os.path.join(output_dir, "run_summary.json")

    if not os.path.exists(output_dir):
        return {"status": "not_found", "error_step": None, "last_prompt_step": None}

    result = {"status": "incomplete", "error_step": None, "last_prompt_step": None}

    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary = json.load(f)

        if "completed_at" in summary:
            result["status"] = "completed"
            return result

        if "error_step" in summary:
            result["status"] = "error"
            result["error_step"] = summary["error_step"]

    last_step, _ = find_last_prompt(output_dir)
    result["last_prompt_step"] = last_step

    return result


def resume_failed_runs(
    futures: Dict[str, Any],
    experiments: List[Dict[str, Any]],
    runner_pool: Any,
    run_name: str,
) -> Dict[str, Any]:
    """Check inactive runs and re-submit failed ones from their error step."""
    resumed_futures = {}

    for exp in experiments:
        exp_key = f"{exp['dataset_name']}_{exp['algorithm']}_{exp['llm']}"
        output_dir = exp.get("output_dir")

        if not output_dir:
            continue

        status = check_run_status(output_dir)

        if status["status"] == "error":
            error_step = status["error_step"]
            last_prompt_step, prompt_content = find_last_prompt(output_dir)

            resume_step = (
                error_step if error_step is not None else (last_prompt_step or 0)
            )

            print(f"[RESUME] Re-Submitting {exp_key} from step {resume_step}")

            future = runner_pool.run(
                run_name=run_name,
                dataset=exp["dataset"],
                output_dir=output_dir,
                algo_name=exp["algorithm"],
                llm=exp["llm"],
                steps=exp["steps"],
                api_key=exp["api_key"],
                batch_size=exp["batch_size"],
                loss_batch_size=exp["loss_batch_size"],
                gradient_batch_size=exp["gradient_batch_size"],
                eval_every=exp["eval_every"],
                verbosity=1,
                start_step=resume_step,
                resume_prompt=prompt_content,
            )
            resumed_futures[exp_key] = future

    return resumed_futures


class AlgorithmRunner(Worker):
    """Worker for running algorithms in parallel.

    Each AlgorithmRunner instance creates a shared LimitSet that is used by all
    LLM workers it instantiates, ensuring proper rate limiting across all LLM calls.
    """

    def run(
        self,
        *,
        dataset: Dataset,
        algo_name: str,
        api_key: str,
        steps: int,
        batch_size: int,
        loss_batch_size: int,
        gradient_batch_size: int,
        eval_every: int,
        run_name: str = "run1",
        llm: str = "llama3.1",
        verbosity: int = 1,
        start_step: int = 0,
        resume_prompt: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run algorithm and return results.

        Args:
            dataset: Dataset to run on
            algo_name: Algorithm name ("gpo", "opro", "textgrad")
            api_key: API key for LLM service
            steps: Number of training steps
            batch_size: Batch size for training
            loss_batch_size: Batch size for loss computation
            gradient_batch_size: Batch size for gradient computation
            eval_every: Evaluate every N steps
            run_name: Name for this run
            llm: LLM family to use
            verbosity: Logging verbosity (0=silent, 1=default, 2=detailed, 3=debug)
            start_step: Resume from this step
            resume_prompt: Resume from this prompt
        """
        print(
            f"[AlgorithmRunner] Starting {algo_name} on {dataset.dataset_name} "
            f"(run: {run_name}, llm: {llm})"
        )

        try:
            tasks = dataset.tasks

            task_output_formats = (
                dataset.task_output_formats
                if len(dataset.task_output_formats) > 0
                else None
            )
            initial_prompt = get_initial_prompt(
                dataset_name=dataset.dataset_name,
                tasks=tasks,
                task_output_formats=task_output_formats,
            )
            task_losses = get_task_losses(
                dataset_name=dataset.dataset_name, tasks=tasks
            )

            shared_limits = create_shared_limits()

            task_llm = create_task_llm(llm=llm, api_key=api_key, limits=shared_limits)
            optimizer_llm = create_optimizer_llm(
                llm=llm, api_key=api_key, limits=shared_limits
            )
            gradient_llm = create_gradient_llm(
                llm=llm, api_key=api_key, limits=shared_limits
            )
            loss_llm = create_loss_llm(llm=llm, api_key=api_key, limits=shared_limits)

            common_params = {
                "tasks": tasks,
                "steps": steps,
                "batch_size": batch_size,
                "loss_batch_size": loss_batch_size,
                "gradient_batch_size": gradient_batch_size,
                "eval_every": eval_every,
                "name": f"{dataset.dataset_name}_{algo_name}_{run_name}",
                "verbosity": verbosity,
            }

            if algo_name == "gpo":
                algo = GPO(
                    task_llm=task_llm,
                    gradient_llm=gradient_llm,
                    optimizer_llm=optimizer_llm,
                    loss_llm=loss_llm,
                    task_losses=task_losses,
                    k=5,
                    warmup_steps=5,
                    **common_params,
                )
            elif algo_name == "textgrad":
                algo = TextGrad(
                    task_llm=task_llm,
                    gradient_llm=gradient_llm,
                    optimizer_llm=optimizer_llm,
                    loss_llm=loss_llm,
                    **common_params,
                )
            elif algo_name == "opro":
                algo = OPRO(
                    task_llm=task_llm,
                    optimizer_llm=optimizer_llm,
                    task_losses=task_losses,
                    k=5,
                    **common_params,
                )
            else:
                raise ValueError(f"Unknown algorithm: {algo_name}")

            results = algo.train(
                dataset=dataset,
                initial_prompt=initial_prompt,
                output_dir=kwargs.get("output_dir"),
                start_step=start_step,
            )

            print(f"[AlgorithmRunner] Completed {algo_name} on {dataset.dataset_name}")
            print("#" * 80)
            return {
                "status": "success",
                "dataset": dataset.dataset_name,
                "algorithm": algo_name,
                "run_name": run_name,
                "llm": llm,
                "steps": steps,
                "batch_size": batch_size,
                "loss_batch_size": loss_batch_size,
                "gradient_batch_size": gradient_batch_size,
                "eval_every": eval_every,
                "results": results,
                **kwargs,
            }
        except Exception as e:
            print(
                f"[AlgorithmRunner] Failed {algo_name} on {dataset.dataset_name}:\n"
                f"{format_exception_msg(e)}"
            )
            print("#" * 80)
            return {
                "status": "error",
                "dataset": dataset.dataset_name,
                "algorithm": algo_name,
                "run_name": run_name,
                "llm": llm,
                "steps": steps,
                "batch_size": batch_size,
                "loss_batch_size": loss_batch_size,
                "gradient_batch_size": gradient_batch_size,
                "eval_every": eval_every,
                "error": str(e),
                **kwargs,
            }
