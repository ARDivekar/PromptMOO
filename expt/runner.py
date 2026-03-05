import asyncio
import json, os
from typing import Any, Dict, List, Optional, Tuple

import litellm
from concurry import CallLimit, LimitSet, RateLimit, RetryAlgorithm, Worker
from concurry.core.limit.limit_set import BaseLimitSet
from morphic import validate
from morphic.typed import format_exception_msg

from prompt_moo.algorithm import GPO, OPRO, TextGrad
from prompt_moo.data_input import Dataset
from prompt_moo.data_structures import Task
from prompt_moo.llm_workers import LLM
from prompt_moo.prompt_template_utils import PromptTemplate
from prompt_moo.task_predictor import parse_task_response


def parse_task_response_retry_until(result: str, **context) -> bool:
    """Retry until the task response is valid JSON."""
    try:
        parse_task_response(result)
        return True
    except Exception:
        return False


# LLM configuration mapping
LLM_CONFIGS = {
    "llama3.1": {
        "task_model": "meta-llama/llama-3.1-8b-instruct",
        "other_model": "meta-llama/llama-3.1-70b-instruct",
        "provider_order": {
            "meta-llama/llama-3.1-8b-instruct": ["novita/fp8", "nebius/fp8", "deepinfra/bf16", "nebius/fast",],
            "meta-llama/llama-3.1-70b-instruct": ["novita/fp8", "nebius/fp8", "deepinfra/bf16", "nebius/fast",],
        }
    },
    "qwen3": {
        "task_model": "qwen/qwen3-8b",
        "other_model": "qwen/qwen3-vl-235b-a22b-thinking",
        "provider_order" : {
            "qwen/qwen3-8b": ["novita/fp8", "fireworks"],
            "qwen/qwen3-vl-235b-a22b-thinking": ["novita/fp8", "fireworks"],
        }
    },
    "gpt4.1": {
        "task_model": "openai/gpt-4.1-nano",
        "other_model": "openai/gpt-4.1",
        "provider_order": {
            "openai/gpt-4.1-nano": ["openai", "azure"],
            "openai/gpt-4.1": ["openai", "azure"],
        }
    },
}


def create_shared_limits() -> BaseLimitSet:
    """Create shared LimitSet for all LLM workers.

    Returns:
        LimitSet configured for rate limiting across all LLM workers.
    """
    return LimitSet(
        limits=[
            # Calls/min (Provider Limit)
            CallLimit(window_seconds=60, capacity=500),
            # Input tokens/min (Cost Control)
            RateLimit(key="input_tokens", window_seconds=60, capacity=10_000_000),
            # Output tokens/min (Cost Control)
            RateLimit(key="output_tokens", window_seconds=60, capacity=1_000_000),
        ],
        mode="asyncio",
        shared=True,
    )


@validate
def create_task_llm(*, llm: str, api_key: str, limits: Optional[Any] = None):
    """Create task LLM (AsyncIO worker for concurrent calls).

    Args:
        llm: LLM family to use. Options: "llama3.1", "qwen3", "gpt4.1"
        api_key: API key for LLM service (required)
        limits: Shared LimitSet for rate limiting (should be shared across all LLMs)
    """
    if llm not in LLM_CONFIGS:
        raise ValueError(f"Unknown LLM: {llm}. Options: {list(LLM_CONFIGS.keys())}")

    model_name = LLM_CONFIGS[llm]["task_model"]
    return LLM.options(
        mode="asyncio",
        num_retries={"call_llm": 10, "*": 0},
        retry_wait={"call_llm": 2, "*": 1},
        retry_algorithm={"call_llm": RetryAlgorithm.Fibonacci, "*": RetryAlgorithm.Exponential},
        retry_jitter={"call_llm": 0.3, "*": 0},
        retry_on={
            "call_llm": [
                ValueError,
                asyncio.TimeoutError,
                litellm.Timeout,
                litellm.APIError,
                litellm.RateLimitError,
                litellm.ServiceUnavailableError,
            ],
            "*": [],
        },
        limits=limits,
    ).init(
        name="task_llm",
        model_name=model_name,
        api_key=api_key,
        temperature=0.1,
        max_tokens=256,
        timeout=60.0,
    )


@validate
def create_optimizer_llm(
    *, llm: str, api_key: str, limits: Optional[BaseLimitSet] = None
):
    """Create optimizer LLM (AsyncIO worker for concurrent calls).

    Args:
        llm: LLM family to use. Options: "llama3.1", "qwen3", "gpt4.1"
        api_key: API key for LLM service (required)
        limits: Shared LimitSet for rate limiting (should be shared across all LLMs)
    """
    if llm not in LLM_CONFIGS:
        raise ValueError(f"Unknown LLM: {llm}. Options: {list(LLM_CONFIGS.keys())}")

    model_name = LLM_CONFIGS[llm]["other_model"]
    return LLM.options(
        mode="asyncio",
        num_retries={"call_llm": 10, "*": 0},
        retry_wait={"call_llm": 2, "*": 1},
        retry_algorithm={"call_llm": RetryAlgorithm.Fibonacci, "*": RetryAlgorithm.Exponential},
        retry_jitter={"call_llm": 0.3, "*": 0},
        retry_on={
            "call_llm": [
                ValueError,
                asyncio.TimeoutError,
                litellm.Timeout,
                litellm.APIError,
                litellm.RateLimitError,
                litellm.ServiceUnavailableError,
            ],
            "*": [],
        },
        limits=limits,
    ).init(
        name="optimizer_llm",
        model_name=model_name,
        api_key=api_key,
        temperature=1.0,
        max_tokens=4096,
        timeout=600.0,
    )


@validate
def create_gradient_llm(
    *, llm: str, api_key: str, limits: Optional[BaseLimitSet] = None
):
    """Create gradient LLM (AsyncIO worker for concurrent calls).

    Args:
        llm: LLM family to use. Options: "llama3.1", "qwen3", "gpt4.1"
        api_key: API key for LLM service (required)
        limits: Shared LimitSet for rate limiting (should be shared across all LLMs)
    """
    if llm not in LLM_CONFIGS:
        raise ValueError(f"Unknown LLM: {llm}. Options: {list(LLM_CONFIGS.keys())}")

    model_name = LLM_CONFIGS[llm]["other_model"]
    return LLM.options(
        mode="asyncio",
        num_retries={"call_llm": 10, "*": 0},
        retry_wait={"call_llm": 2, "*": 1},
        retry_algorithm={"call_llm": RetryAlgorithm.Fibonacci, "*": RetryAlgorithm.Exponential},
        retry_jitter={"call_llm": 0.3, "*": 0},
        retry_on={
            "call_llm": [
                ValueError,
                asyncio.TimeoutError,
                litellm.Timeout,
                litellm.APIError,
                litellm.RateLimitError,
                litellm.ServiceUnavailableError,
            ],
            "*": [],
        },
        limits=limits,
    ).init(
        name="gradient_llm",
        model_name=model_name,
        api_key=api_key,
        temperature=0.1,
        max_tokens=2048,
        timeout=300.0,
    )


@validate
def create_loss_llm(*, llm: str, api_key: str, limits: Optional[BaseLimitSet] = None):
    """Create loss LLM (AsyncIO worker for concurrent calls).

    Args:
        llm: LLM family to use. Options: "llama3.1", "qwen3", "gpt4.1"
        api_key: API key for LLM service (required)
        limits: Shared LimitSet for rate limiting (should be shared across all LLMs)
    """
    if llm not in LLM_CONFIGS:
        raise ValueError(f"Unknown LLM: {llm}. Options: {list(LLM_CONFIGS.keys())}")

    model_name = LLM_CONFIGS[llm]["other_model"]
    return LLM.options(
        mode="asyncio",
        num_retries={"call_llm": 10, "*": 0},
        retry_wait={"call_llm": 2, "*": 1},
        retry_algorithm={"call_llm": RetryAlgorithm.Fibonacci, "*": RetryAlgorithm.Exponential},
        retry_jitter={"call_llm": 0.3, "*": 0},
        retry_on={
            "call_llm": [
                ValueError,
                asyncio.TimeoutError,
                litellm.Timeout,
                litellm.APIError,
                litellm.RateLimitError,
                litellm.ServiceUnavailableError,
            ],
            "*": [],
        },
        limits=limits,
    ).init(
        name="loss_llm",
        model_name=model_name,
        api_key=api_key,
        temperature=0.1,
        max_tokens=256,
        timeout=60.0,
    )


# Dataset configurations
DATASET_CONFIGS = {
    "SummEval": {
        "prompt_prefix": "Evaluate the summary. Output JSON with the requested metric scores. Do NOT include reasoning or explanations. Each metric should contain a single integer. Formats like '4/5' or '4|5' are invalid.",
        "task_output_formats": {
            "fluency": "An integer between 1 to 5",                         # "1|2|3|4|5",
            "coherence": "An integer between 1 to 5",                       # "1|2|3|4|5",
            "relevance": "An integer between 1 to 5",                       # "1|2|3|4|5",
            "consistency": "An integer between 1 to 5",                     # "1|2|3|4|5",
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
            "anger": "An integer between 0 to 3",                       # "0|1|2|3",
            "fear": "An integer between 0 to 3",                        # "0|1|2|3",
            "joy": "An integer between 0 to 3",                         # "0|1|2|3",
            "sadness": "An integer between 0 to 3",                     # "0|1|2|3",
            "surprise": "An integer between 0 to 3",                    # "0|1|2|3",
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

    # Use provided task_output_formats, or fall back to DATASET_CONFIGS
    if task_output_formats is None:
        task_output_formats = config["task_output_formats"]

    # Build dynamic task list for the prompt
    task_names = [task.task_name for task in tasks]
    if len(task_names) == 1:
        task_list_str = f"Output ONLY the '{task_names[0]}' metric."
    else:
        task_list_str = f"Output the following metrics: {', '.join(task_names)}."

    # Build JSON format section with only the selected tasks
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

    # Combine into full skeleton
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

    The prompt skeleton is dynamically generated to include only the selected tasks.

    Args:
        dataset_name: Name of the dataset
        tasks: List of tasks to include in the prompt
        task_output_formats: Optional dict mapping task names to output format specs.
            If not provided, will use DATASET_CONFIGS.

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
def get_task_losses(*, dataset_name: str, tasks: Optional[List[Task]] = None) -> Dict[str, str]:
    """Get task losses for a dataset.
    
    Args:
        dataset_name: Name of the dataset
        tasks: Optional list of tasks to filter losses for. If None, returns all losses.
    
    Returns:
        Dict mapping task names to loss function names
    """
    all_losses = DATASET_CONFIGS[dataset_name]["task_losses"]
    
    # If tasks specified, filter to only those tasks
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

    This is useful for running algorithms on individual tasks rather than
    all tasks in a dataset simultaneously. The prompt skeleton will only
    show the selected task in the JSON format section.

    Args:
        task: The task to create a prompt for
        dataset_name: Name of the dataset (to get the skeleton format)
        task_output_formats: Optional dict mapping task names to output format specs.
            If not provided, will use DATASET_CONFIGS.

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
    """Find the latest saved prompt in an output directory.
    
    Iterates through step_{i}_new.txt files from i=100 down to 0 
    and returns the step number and prompt content of the last available one.
    
    Args:
        output_dir: The run output directory containing a 'prompts' subfolder
        
    Returns:
        Tuple of (last_step, prompt_content) or (None, None) if no prompts found
    """
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
    """Check the status of a run based on its output files.
    
    Args:
        output_dir: The run output directory
        
    Returns:
        Dict with:
        - 'status': 'completed' | 'error' | 'incomplete' | 'not_found'
        - 'error_step': step where error occurred (if applicable)
        - 'last_prompt_step': last step with saved prompt
    """
    summary_path = os.path.join(output_dir, "run_summary.json")

    if not os.path.exists(output_dir):
        return {"status" : "not_found", "error_step": None, "last_prompt_step" : None}

    result = {"status" : "incomplete", "error_step" : None, "last_prompt_step" : None}

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
    run_name: str
    ) -> Dict[str, Any]:
    """Check inactive runs and re-submit failed ones from their error step.
    
    Args:
        futures: Dict of experiment_key -> future from initial submission
        experiments: List of experiment configurations
        runner_pool: The AlgorithmRunner pool to submit resumed runs
        run_name: Name for the run
        
    Returns:
        Dict of experiment_key -> new future for resumed runs
    """
    from concurry import wait 

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
            
            resume_step = error_step if error_step is not None else (last_prompt_step or 0)

            print(f"[RESUME] Re-Submitting {exp_key} from step {resume_step}")

            # Re-submit with start_step
            future = runner_pool.run(
                run_name=run_name,
                dataset=exp['dataset'],
                output_dir=output_dir,
                algo_name=exp['algorithm'],
                llm=exp['llm'],
                steps=exp['steps'],
                api_key=exp['api_key'],
                batch_size=exp['batch_size'],
                loss_batch_size=exp['loss_batch_size'],
                gradient_batch_size=exp['gradient_batch_size'],
                eval_every=exp['eval_every'],
                verbosity=1,
                start_step=resume_step,  # Resume from error step
                resume_prompt=prompt_content,  # Use last saved prompt
            )
            resumed_futures[exp_key] = future

    return resumed_futures
            

class AlgorithmRunner(Worker):
    """Ray-based worker for running algorithms in parallel.

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
        start_step: int = 0, ##  Resume from this step
        resume_prompt: Optional[str] = None, ## Resume from this prompt
        **kwargs,
    ) -> Dict[str, Any]:
        """Run algorithm and return results.

        Args:
            dataset: Dataset to run on
            algo_name: Algorithm name ("gpo", "opro", "textgrad")
            api_key: API key for LLM service (required)
            steps: Number of training steps (required)
            batch_size: Batch size for training (required)
            loss_batch_size: Batch size for loss computation (required)
            gradient_batch_size: Batch size for gradient computation (required)
            eval_every: Evaluate every N steps (required)
            run_name: Name for this run (default: "run1")
            llm: LLM family to use ("llama3.1", "qwen3", "gpt4.1") (default: "llama3.1")
            verbosity: Logging verbosity (0=silent, 1=default, 2=detailed, 3=debug) (default: 1)
            start_step: Resume from this step
            resume_prompt: Resume from this prompt
        """
        print(
            f"[AlgorithmRunner] Starting {algo_name} on {dataset.dataset_name} (run: {run_name}, llm: {llm})"
        )

        try:
            tasks = dataset.tasks

            # Get configuration
            # Use task_output_formats from dataset if available, otherwise fall back to DATASET_CONFIGS
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
            task_losses = get_task_losses(dataset_name=dataset.dataset_name, tasks=tasks)

            # Create shared LimitSet for all LLMs in this runner instance
            # All LLM workers will share these limits for efficient rate limiting
            shared_limits = create_shared_limits()

            # Create LLMs with shared limits
            task_llm = create_task_llm(llm=llm, api_key=api_key, limits=shared_limits)
            optimizer_llm = create_optimizer_llm(
                llm=llm, api_key=api_key, limits=shared_limits
            )
            gradient_llm = create_gradient_llm(
                llm=llm, api_key=api_key, limits=shared_limits
            )
            loss_llm = create_loss_llm(llm=llm, api_key=api_key, limits=shared_limits)

            # Common parameters
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

            # Create algorithm instance
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

            if resume_prompt is not None:
                pass


            # Run training
            results = algo.train(
                dataset=dataset,
                initial_prompt=initial_prompt,
                output_dir=kwargs.get("output_dir"),
                start_step=start_step,
            )

            print(
                f"[AlgorithmRunner] Completed {algo_name} on {dataset.dataset_name}"
            )
            print("#" * 80)
            print("#" * 80)
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
                f"[AlgorithmRunner] Failed {algo_name} on {dataset.dataset_name}:\n{format_exception_msg(e)}"
            )
            print("#" * 80)
            print("#" * 80)
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
