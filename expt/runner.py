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
from prompt_moo.prompt_template import PromptTemplate
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


@validate
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
    *,
    llm: str,
    api_key: str,
    limits: BaseLimitSet,
    reasoning: bool = False,
    temperature: Optional[float] = None,
) -> SlowBurnLLM:
    """Create task LLM using SlowBurnLLM.

    Args:
        llm: LLM family key in LLM_CONFIGS.
        api_key: API key for LLM service.
        limits: Shared LimitSet for rate limiting.
        reasoning: Enable reasoning/thinking mode. Adds REASONING_EXTRA_TOKENS to max_tokens.
        temperature: Explicit temperature override from the algorithm class.
            When ``None``, falls back to ``promptmoo_config.defaults.task_llm_temperature``.
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

    resolved_temperature = (
        temperature if temperature is not None else cfg.task_llm_temperature
    )
    llm = SlowBurnLLM.options(
        mode="asyncio",
        limits=limits,
        **_build_retry_config(cfg=cfg),
    ).init(
        name="task_llm",
        model_name=model_name,
        api_key=api_key,
        temperature=resolved_temperature,
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
    *,
    llm: str,
    api_key: str,
    limits: BaseLimitSet,
    temperature: Optional[float] = None,
) -> SlowBurnLLM:
    """Create optimizer LLM using SlowBurnLLM.

    Args:
        llm: LLM family key in LLM_CONFIGS.
        api_key: API key for LLM service.
        limits: Shared LimitSet for rate limiting.
        temperature: Explicit temperature override from the algorithm class.
            When ``None``, falls back to ``promptmoo_config.defaults.optimizer_llm_temperature``.
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

    resolved_temperature = (
        temperature if temperature is not None else cfg.optimizer_llm_temperature
    )
    llm = SlowBurnLLM.options(
        mode="asyncio",
        limits=limits,
        **_build_retry_config(cfg=cfg),
    ).init(
        name="optimizer_llm",
        model_name=model_name,
        api_key=api_key,
        temperature=resolved_temperature,
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
def create_gradient_llm(
    *,
    llm: str,
    api_key: str,
    limits: BaseLimitSet,
    temperature: Optional[float] = None,
) -> SlowBurnLLM:
    """Create gradient LLM using SlowBurnLLM.

    Args:
        llm: LLM family key in LLM_CONFIGS.
        api_key: API key for LLM service.
        limits: Shared LimitSet for rate limiting.
        temperature: Explicit temperature override from the algorithm class.
            When ``None``, falls back to ``promptmoo_config.defaults.gradient_llm_temperature``.
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

    resolved_temperature = (
        temperature if temperature is not None else cfg.gradient_llm_temperature
    )
    llm = SlowBurnLLM.options(
        mode="asyncio",
        limits=limits,
        **_build_retry_config(cfg=cfg),
    ).init(
        name="gradient_llm",
        model_name=model_name,
        api_key=api_key,
        temperature=resolved_temperature,
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
def create_loss_llm(
    *,
    llm: str,
    api_key: str,
    limits: BaseLimitSet,
    temperature: Optional[float] = None,
) -> SlowBurnLLM:
    """Create loss LLM using SlowBurnLLM.

    Args:
        llm: LLM family key in LLM_CONFIGS.
        api_key: API key for LLM service.
        limits: Shared LimitSet for rate limiting.
        temperature: Explicit temperature override from the algorithm class.
            When ``None``, falls back to ``promptmoo_config.defaults.loss_llm_temperature``.
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

    resolved_temperature = (
        temperature if temperature is not None else cfg.loss_llm_temperature
    )
    llm = SlowBurnLLM.options(
        mode="asyncio",
        limits=limits,
        **_build_retry_config(cfg=cfg),
    ).init(
        name="loss_llm",
        model_name=model_name,
        api_key=api_key,
        temperature=resolved_temperature,
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


@validate
def build_prompt_skeleton(
    *,
    dataset: Dataset,
    tasks: List[Task],
) -> str:
    """Build prompt skeleton dynamically from the Dataset object.

    The skeleton is the frozen portion of the task prompt that does NOT
    change during optimization.  It contains:
    1. The evaluation directive (what to evaluate)
    2. General output constraints (no reasoning)

    The per-task ``## Output format`` section is NOT part of the skeleton.
    It lives in ``task_output_formats`` on ``PromptTemplate`` and is rendered
    dynamically by ``render_instructions()``, which supports per-task
    filtering via ``task_filter`` to prevent cross-task leakage in
    TextGrad's ``separate_tasks`` mode.

    The mutable per-task instructions are also NOT part of the skeleton;
    they are appended by ``PromptTemplate.render_instructions()``.

    Args:
        dataset: Dataset instance.
        tasks: List of tasks to include (order is preserved).

    Returns:
        Prompt skeleton string (without ``## Output format`` section).
    """
    from prompt_moo.task_output_spec import collective_output_noun

    task_output_specs: Dict[str, Any] = dataset.task_output_specs
    evaluation_directive: str = dataset.evaluation_directive

    if len(evaluation_directive) == 0:
        if len(dataset.prompt_prefix) > 0:
            evaluation_directive = dataset.prompt_prefix
        else:
            raise ValueError(
                f"Dataset {dataset.dataset_name!r} has empty evaluation_directive. "
                f"Define it as a ClassVar on the Dataset subclass."
            )

    if len(task_output_specs) == 0:
        task_output_formats: Dict[str, str] = dataset.task_output_formats
        if len(task_output_formats) == 0:
            raise ValueError(
                f"Dataset {dataset.dataset_name!r} has empty task_output_specs "
                f"and task_output_formats. Define task_output_specs."
            )
        for task in tasks:
            if task.task_name not in task_output_formats:
                raise ValueError(
                    f"Task '{task.task_name}' not found in task_output_formats "
                    f"for dataset '{dataset.dataset_name}'"
                )
        output_noun_plural: str = "values"
    else:
        for task in tasks:
            if task.task_name not in task_output_specs:
                raise ValueError(
                    f"Task '{task.task_name}' not found in task_output_specs "
                    f"for dataset '{dataset.dataset_name}'"
                )
        output_noun_plural = collective_output_noun(
            {t.task_name: task_output_specs[t.task_name] for t in tasks}
        )

    skeleton: str = (
        f"{evaluation_directive.strip()}"
        f"\nUse the Instructions below to perform your evaluation. "
        f"Output a JSON with the requested {output_noun_plural}. "
        f"Do NOT include reasoning or explanations.\n"
    )

    return skeleton


def _build_task_output_formats(
    *,
    dataset: Dataset,
    tasks: List[Task],
) -> Dict[str, str]:
    """Build per-task output format specs for ``PromptTemplate.task_output_formats``.

    Reads from ``dataset.task_output_specs`` (preferred) or falls back to
    ``dataset.task_output_formats`` (a plain string dict).

    Args:
        dataset: Dataset instance.
        tasks: List of tasks to include (order is preserved).

    Returns:
        Dict mapping task_name to its format string (e.g. ``"1|2|3|4|5"``).
    """
    task_output_specs: Dict[str, Any] = dataset.task_output_specs
    if len(task_output_specs) > 0:
        result: Dict[str, str] = {}
        for task in tasks:
            if task.task_name not in task_output_specs:
                raise ValueError(
                    f"Task '{task.task_name}' not found in task_output_specs "
                    f"for dataset '{dataset.dataset_name}'"
                )
            result[task.task_name] = task_output_specs[task.task_name].format_str
        return result

    task_output_formats: Dict[str, str] = dataset.task_output_formats
    if len(task_output_formats) == 0:
        raise ValueError(
            f"Dataset {dataset.dataset_name!r} has empty task_output_specs "
            f"and task_output_formats. Define task_output_specs."
        )
    result = {}
    for task in tasks:
        if task.task_name not in task_output_formats:
            raise ValueError(
                f"Task '{task.task_name}' not found in task_output_formats "
                f"for dataset '{dataset.dataset_name}'"
            )
        result[task.task_name] = task_output_formats[task.task_name]
    return result


@validate
def get_initial_prompt(
    *,
    dataset: Dataset,
    tasks: List[Task],
) -> PromptTemplate:
    """Get initial prompt for a dataset with specified tasks.

    Args:
        dataset: Dataset instance.
        tasks: List of tasks to include in the prompt.

    Returns:
        PromptTemplate configured for the specified tasks, with
        ``task_output_formats`` populated for per-task filtering support.
    """
    skeleton: str = build_prompt_skeleton(
        dataset=dataset,
        tasks=tasks,
    )
    task_output_formats: Dict[str, str] = _build_task_output_formats(
        dataset=dataset,
        tasks=tasks,
    )
    return PromptTemplate(
        skeleton=skeleton,
        instruction={t.task_name: t.task_instruction for t in tasks},
        tasks=tasks,
        input_col_labels=dataset.input_col_labels,
        task_output_formats=task_output_formats,
    )


@validate
def get_task_losses(
    *, dataset: Dataset, tasks: Optional[List[Task]] = None
) -> Dict[str, str]:
    """Get task losses from the Dataset object.

    Args:
        dataset: Dataset instance (carries task_losses).
        tasks: Optional list of tasks to filter losses for.

    Returns:
        Dict mapping task names to loss function names.
    """
    all_losses = dataset.task_losses
    if len(all_losses) == 0:
        raise ValueError(
            f"Dataset {dataset.dataset_name!r} has empty task_losses. "
            f"Define it as a ClassVar on the Dataset subclass."
        )
    if tasks is not None:
        task_names = {t.task_name for t in tasks}
        return {k: v for k, v in all_losses.items() if k in task_names}
    return all_losses


@validate
def get_single_task_prompt(
    *,
    task: Task,
    dataset: Dataset,
) -> PromptTemplate:
    """Get initial prompt for a single task.

    Args:
        task: The task to create a prompt for.
        dataset: Dataset instance.

    Returns:
        PromptTemplate configured for this single task.
    """
    skeleton = build_prompt_skeleton(
        dataset=dataset,
        tasks=[task],
    )
    return PromptTemplate(
        skeleton=skeleton,
        instruction={task.task_name: task.task_instruction},
        tasks=[task],
        input_col_labels=dataset.input_col_labels,
    )


@validate
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


@validate
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


@validate
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

    Algorithm-specific parameters (e.g. ``k``, ``validation_metric``,
    ``trajectory_strategy``) are passed through via
    ``**algo_params`` and forwarded directly to the algorithm constructor.
    The runner does not hard-code any algorithm hyperparameters.
    """

    # @validate omitted: Worker methods are serialized for Ray remote execution.
    # validate_call closure is not picklable across process boundaries.
    def run(
        self,
        *,
        dataset: Dataset,
        algo_name: str,
        api_key: str,
        steps: Optional[int] = None,
        batch_size: Optional[int] = None,
        eval_every: Optional[int] = None,
        run_name: str = "run1",
        llm: str = "qwen3",
        verbosity: int = 1,
        start_step: int = 0,
        resume_prompt: Optional[str] = None,
        output_dir: Optional[str] = None,
        **algo_params,
    ) -> Dict[str, Any]:
        """Run algorithm and return results.

        Args:
            dataset: Dataset to run on.
            algo_name: Algorithm name ("gpo", "opro", "textgrad").
            api_key: API key for LLM service.
            steps: Number of training steps.
            batch_size: Batch size for training.
            eval_every: Evaluate every N steps.
            run_name: Name for this run.
            llm: LLM family to use.
            verbosity: Logging verbosity (0=silent, 1=default, 2=detailed, 3=debug).
            start_step: Resume from this step.
            resume_prompt: Resume from this prompt content.
            output_dir: Output directory (auto-generated if None).
            **algo_params: Algorithm-specific hyperparameters passed directly
                to the algorithm constructor.  Examples:
                - GPO: ``k=5``, ``trajectory_strategy="relevance"``
                - OPRO: ``k=20``, ``num_candidates=8``
                - TextGrad: ``validation_metric="accuracy"``,
                  ``validation_gate_samples=50``
                Any parameter accepted by the algorithm's constructor can
                be passed here.  Unknown keys will be rejected by Pydantic
                validation at construction time.
        """
        print(
            f"[AlgorithmRunner] Starting {algo_name} on {dataset.dataset_name} "
            f"(run: {run_name}, llm: {llm})"
        )

        try:
            tasks = dataset.tasks

            initial_prompt = get_initial_prompt(
                dataset=dataset,
                tasks=tasks,
            )
            task_losses = get_task_losses(dataset=dataset, tasks=tasks)

            shared_limits = create_shared_limits()

            # Resolve the algorithm class so we can read its temperature defaults
            # BEFORE creating LLM workers.  If the caller passed explicit
            # temperature overrides in algo_params, those take priority over
            # the class-level defaults.
            algo_cls_map = {
                "gpo": GPO,
                "opro": OPRO,
                "textgrad": TextGrad,
            }
            if algo_name not in algo_cls_map:
                raise ValueError(
                    f"Unknown algorithm: {algo_name!r}. "
                    f"Must be 'gpo', 'textgrad', or 'opro'."
                )
            algo_cls = algo_cls_map[algo_name]

            def _resolve_temperature(
                field_name: str,
            ) -> Optional[float]:
                """Read temperature from algo_params (explicit override) or class default."""
                if field_name in algo_params:
                    return algo_params[field_name]
                field_info = algo_cls.model_fields.get(field_name)
                if field_info is not None:
                    return field_info.default
                return None

            task_llm = create_task_llm(
                llm=llm,
                api_key=api_key,
                limits=shared_limits,
                temperature=_resolve_temperature("task_llm_temperature"),
            )
            optimizer_llm = create_optimizer_llm(
                llm=llm,
                api_key=api_key,
                limits=shared_limits,
                temperature=_resolve_temperature("optimizer_llm_temperature"),
            )
            gradient_llm = create_gradient_llm(
                llm=llm,
                api_key=api_key,
                limits=shared_limits,
                temperature=_resolve_temperature("gradient_llm_temperature"),
            )
            loss_llm = create_loss_llm(
                llm=llm,
                api_key=api_key,
                limits=shared_limits,
                temperature=_resolve_temperature("loss_llm_temperature"),
            )

            common_params: Dict[str, Any] = {
                "tasks": tasks,
                "name": f"{dataset.dataset_name}_{algo_name}_{run_name}",
                "verbosity": verbosity,
            }
            if steps is not None:
                common_params["steps"] = steps
            if batch_size is not None:
                common_params["batch_size"] = batch_size
            if eval_every is not None:
                common_params["eval_every"] = eval_every

            if algo_name == "gpo":
                algo = algo_cls(
                    task_llm=task_llm,
                    optimizer_llm=optimizer_llm,
                    task_losses=task_losses,
                    **{
                        **common_params,
                        **algo_params,
                    },
                )
            elif algo_name == "textgrad":
                textgrad_defaults: Dict[str, Any] = {
                    "validation_metric": "none",
                }
                textgrad_defaults.update(algo_params)
                algo = algo_cls(
                    task_llm=task_llm,
                    gradient_llm=gradient_llm,
                    optimizer_llm=optimizer_llm,
                    loss_llm=loss_llm,
                    task_losses=task_losses,
                    **{
                        **common_params,
                        **textgrad_defaults,
                    },
                )
            elif algo_name == "opro":
                algo = algo_cls(
                    task_llm=task_llm,
                    optimizer_llm=optimizer_llm,
                    task_losses=task_losses,
                    **{
                        **common_params,
                        **algo_params,
                    },
                )
            else:
                raise ValueError(
                    f"Unknown algorithm: {algo_name!r}. "
                    f"Must be 'gpo', 'textgrad', or 'opro'."
                )

            results = algo.train(
                dataset=dataset,
                initial_prompt=initial_prompt,
                output_dir=output_dir,
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
                "eval_every": eval_every,
                "results": results,
                **{
                    **common_params,
                    **algo_params,
                },
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
                "eval_every": eval_every,
                "error": format_exception_msg(e),
                **{
                    **common_params,
                    **algo_params,
                },
            }
