"""
Unified end-to-end test with full artifact capture to markdown.

Runs each algorithm (OPRO, GPO, TextGrad) through the real PromptMOO
pipeline with actual LLM calls (qwen3 via OpenRouter), in both single-task
and all-task modes.  Every intermediate artifact — prompts sent to LLMs,
raw LLM responses, parsed outputs, feedbacks, gradients, meta-prompts,
optimizer responses, trajectory state — is read from the ObservabilityManager's
parquet/JSON output and dumped into a structured markdown file for inspection.

Configuration via environment variables:
    E2E_STEPS=2             Number of optimization steps (default: 6)
    E2E_BATCH_SIZE=3        Batch size for training (default: 12)
    E2E_LOSS_BATCH_SIZE=1   Batch size for loss computation (default: 2)
    E2E_GRADIENT_BATCH_SIZE=1  Batch size for gradient computation (default: 3)
    E2E_EVAL_EVERY=99999    Evaluate every N steps (default: 99999)
    E2E_LLM_FAMILY=qwen3    LLM family key (default: qwen3)
    E2E_ALGO_PARAMS='{"k":7}'  JSON string of algorithm-specific overrides (default: "{}")

Usage:
    # Run ALL algorithms (single + all-task for each):
    pytest tests/test_e2e_unified.py -s -v --timeout=600

    # Quick 2-step run for GPO single-task:
    E2E_STEPS=2 E2E_BATCH_SIZE=3 pytest tests/test_e2e_unified.py -s -v --algorithms=gpo -k "single_task" --timeout=600

    # Run specific algorithm(s) via --algorithms flag:
    pytest tests/test_e2e_unified.py -s -v --algorithms=opro --timeout=600
    pytest tests/test_e2e_unified.py -s -v --algorithms=opro,textgrad --timeout=600
    pytest tests/test_e2e_unified.py -s -v --algorithms=opro,gpo,textgrad --timeout=600

    # Run a single algorithm via pytest -k (matches test function name):
    pytest tests/test_e2e_unified.py -s -v -k "opro" --timeout=600
    pytest tests/test_e2e_unified.py -s -v -k "gpo" --timeout=600
    pytest tests/test_e2e_unified.py -s -v -k "textgrad" --timeout=600

    # Run only all-task mode (all algorithms):
    pytest tests/test_e2e_unified.py -s -v -k "all_task" --timeout=600

    # Run only single-task mode (all algorithms):
    pytest tests/test_e2e_unified.py -s -v -k "single_task" --timeout=600

    # Combine --algorithms with -k for precise control:
    pytest tests/test_e2e_unified.py -s -v --algorithms=gpo -k "all_task" --timeout=600

    # Pass algorithm-specific overrides via E2E_ALGO_PARAMS:
    E2E_STEPS=3 E2E_BATCH_SIZE=3 E2E_ALGO_PARAMS='{"use_textual_feedback":true}' \\
      pytest tests/test_e2e_unified.py -s -v --algorithms=gpo -k "all_task" --timeout=600

    E2E_ALGO_PARAMS='{"k":7,"num_candidates":8,"trajectory_strategy":"importance"}' \\
      pytest tests/test_e2e_unified.py -s -v --algorithms=gpo --timeout=600

Requires OPENROUTER_API_KEY in .env.
"""

import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "expt"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tests.conftest import skip_no_api_key

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# All test parameters can be overridden via environment variables.
# This allows quick iteration without editing the file:
#
#     E2E_STEPS=2 E2E_BATCH_SIZE=3 pytest tests/test_e2e_unified.py -s -v --algorithms=gpo
#
# Env vars and their defaults:
#     E2E_STEPS          Number of optimization steps (default: 6)
#     E2E_BATCH_SIZE     Batch size for training (default: 12)
#     E2E_LOSS_BATCH_SIZE    Batch size for loss computation (default: 2)
#     E2E_GRADIENT_BATCH_SIZE  Batch size for gradient computation (default: 3)
#     E2E_EVAL_EVERY     Evaluate every N steps; 99999 = only at start and end (default: 99999)
#     E2E_LLM_FAMILY     LLM family key from runner.LLM_CONFIGS (default: qwen3)
#     E2E_ALGO_PARAMS    JSON string of algorithm-specific overrides merged into
#                        the algorithm constructor kwargs. Applied to ALL algorithms
#                        in the run; unknown keys are rejected by Pydantic at
#                        construction time. (default: "{}")
#
# Algorithm-specific parameters accepted by E2E_ALGO_PARAMS (examples):
#
#   GPO:  {"k": 7, "num_candidates": 8, "trajectory_strategy": "relevance",
#          "use_textual_feedback": true, "warmup_steps": 0,
#          "initial_step_size": 25, "final_step_size": 5}
#   OPRO: {"k": 20, "num_candidates": 8}
#   TextGrad: {"validation_metric": "accuracy"}
#
# Example usage:
#     E2E_STEPS=3 E2E_BATCH_SIZE=3 E2E_ALGO_PARAMS='{"use_textual_feedback":true}' \
#       pytest tests/test_e2e_unified.py -s -v --algorithms=gpo -k "all_task"
#
STEPS = int(os.environ.get("E2E_STEPS", "6"))
BATCH_SIZE = int(os.environ.get("E2E_BATCH_SIZE", "12"))
LOSS_BATCH_SIZE = int(os.environ.get("E2E_LOSS_BATCH_SIZE", "2"))
GRADIENT_BATCH_SIZE = int(os.environ.get("E2E_GRADIENT_BATCH_SIZE", "3"))
EVAL_EVERY = int(os.environ.get("E2E_EVAL_EVERY", "99999"))
LLM_FAMILY = os.environ.get("E2E_LLM_FAMILY", "qwen3")

_raw_algo_params: str = os.environ.get("E2E_ALGO_PARAMS", "{}")
try:
    ALGO_PARAMS: Dict[str, Any] = json.loads(_raw_algo_params)
except json.JSONDecodeError as _e:
    raise ValueError(
        f"E2E_ALGO_PARAMS must be valid JSON, got: {_raw_algo_params!r}"
    ) from _e
if not isinstance(ALGO_PARAMS, dict):
    raise ValueError(
        f"E2E_ALGO_PARAMS must be a JSON object (dict), got: {type(ALGO_PARAMS).__name__}"
    )

ALL_ALGORITHMS = ["opro", "gpo", "textgrad"]

E2E_OUTPUT_DIR = Path(__file__).parent / "e2e_outputs" / "unified"


def _get_requested_algorithms(request: Any) -> List[str]:
    """Read --algorithms from the pytest CLI and return the list to run.

    Args:
        request: The pytest ``request`` fixture.

    Returns:
        List of algorithm name strings.  Defaults to ALL_ALGORITHMS when
        --algorithms is not provided.

    Raises:
        ValueError: If any algorithm name in the comma-separated list is invalid.
    """
    raw_value: Optional[str] = request.config.getoption("--algorithms", default=None)
    if raw_value is None or len(raw_value.strip()) == 0:
        return list(ALL_ALGORITHMS)

    requested = [a.strip().lower() for a in raw_value.split(",") if len(a.strip()) > 0]
    invalid = [a for a in requested if a not in ALL_ALGORITHMS]
    if len(invalid) > 0:
        raise ValueError(
            f"Unknown algorithm(s): {invalid}. Valid values: {ALL_ALGORITHMS}"
        )
    return requested


# ---------------------------------------------------------------------------
# Markdown generation from ObservabilityManager artifacts
# ---------------------------------------------------------------------------
def _build_markdown_report(
    *,
    algo_name: str,
    task_mode: str,
    task_names: List[str],
    output_dir: str,
    num_steps: int,
) -> str:
    """Read all logged artifacts from disk and build a structured markdown report.

    Args:
        algo_name: Algorithm identifier (opro, gpo, textgrad).
        task_mode: "single" or "all".
        task_names: List of task names used in this run.
        output_dir: Path to the run's output directory.
        num_steps: Number of optimization steps completed.

    Returns:
        Markdown string with every intermediate artifact.
    """
    from prompt_moo.observability import ObservabilityManager

    lines: List[str] = []
    lines.append(f"# {algo_name.upper()} E2E Artifact Dump ({task_mode}-task)")
    lines.append("")
    lines.append(f"- **Generated:** {datetime.now().isoformat()}")
    lines.append(f"- **Algorithm:** {algo_name}")
    lines.append(f"- **Task mode:** {task_mode}")
    lines.append(f"- **Tasks:** {', '.join(task_names)}")
    lines.append(f"- **Steps:** {num_steps}")
    lines.append(f"- **Batch size:** {BATCH_SIZE}")
    lines.append(f"- **LLM family:** {LLM_FAMILY}")
    if len(ALGO_PARAMS) > 0:
        lines.append(
            f"- **Algo overrides (E2E_ALGO_PARAMS):** `{json.dumps(ALGO_PARAMS)}`"
        )
    lines.append(f"- **Output dir:** `{output_dir}`")
    lines.append("")

    run_logs = ObservabilityManager.read_run_logs(output_dir)
    lines.append(f"Total rows in run_logs: {len(run_logs)}")
    lines.append("")

    summary_path = os.path.join(output_dir, "run_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary = json.load(f)
        lines.append("## Run Summary")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(summary, indent=2, default=str))
        lines.append("```")
        lines.append("")

    for step_num in range(1, num_steps + 1):
        matching_rows = run_logs[run_logs["step"] == step_num]
        if len(matching_rows) == 0:
            lines.append(f"## Step {step_num} — MISSING FROM LOGS")
            lines.append("")
            continue

        row = matching_rows.iloc[0]
        lines.append("---")
        lines.append(f"## Step {step_num}")
        lines.append("")

        _append_step_section(
            lines=lines,
            row=row,
            step_num=step_num,
            algo_name=algo_name,
            task_names=task_names,
            output_dir=output_dir,
        )

    return "\n".join(lines)


def _parse_json_field(value: Any) -> Any:
    """Parse a JSON string field from the parquet row, or return as-is if already parsed."""
    if isinstance(value, str):
        return json.loads(value)
    return value


def _append_step_section(
    *,
    lines: List[str],
    row: Any,
    step_num: int,
    algo_name: str,
    task_names: List[str],
    output_dir: str,
) -> None:
    """Append all substep sections for one step to the markdown lines list.

    Args:
        lines: Accumulator list of markdown lines.
        row: Pandas Series for this step from run_logs.
        step_num: Step number.
        algo_name: Algorithm name.
        task_names: Task names in this run.
        output_dir: Run output directory.
    """
    # --- Batch ---
    if "batch" in row and row["batch"] is not None:
        batch_data = _parse_json_field(row["batch"])
        lines.append("### Batch")
        lines.append("")
        lines.append(f"- Num samples: {batch_data.get('num_samples', 'N/A')}")
        sample_ids = [s.get("sample_id", "?") for s in batch_data.get("samples", [])]
        lines.append(f"- Sample IDs: {', '.join(sample_ids)}")
        lines.append("")

    # --- Step 1: Predictions ---
    if "predictions" in row and row["predictions"] is not None:
        pred_data = _parse_json_field(row["predictions"])
        num_preds = pred_data.get("num_predictions", 0)
        lines.append("### Step 1: Predictions")
        lines.append("")
        lines.append(f"- Num predictions: {num_preds}")
        lines.append("")

        for idx, pred in enumerate(pred_data.get("predictions", [])):
            lines.append(f"#### Prediction {idx}")
            lines.append("")

            prompt_text = pred.get("prompt")
            if prompt_text is not None:
                lines.append("**Prompt sent to Task LLM:**")
                lines.append("")
                lines.append("```")
                lines.append(prompt_text)
                lines.append("```")
                lines.append("")

            lines.append(f"**Raw response:** `{pred.get('raw_response', 'N/A')}`")
            lines.append("")
            lines.append("**Parsed task_outputs:**")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(pred.get("task_outputs", {}), indent=2))
            lines.append("```")
            lines.append("")

    # --- Algorithm Context ---
    if "algorithm_context" in row and row["algorithm_context"] is not None:
        ctx_data = _parse_json_field(row["algorithm_context"])
        lines.append("### Algorithm Context")
        lines.append("")
        lines.append(f"- Keys: {list(ctx_data.keys())}")

        if "trajectory" in ctx_data:
            trajectory_items = ctx_data["trajectory"]
            lines.append(f"- Trajectory elements: {len(trajectory_items)}")
            for tidx, telem in enumerate(trajectory_items):
                scores_str = ", ".join(
                    f"{tn}: {[f'{fb.get("value", 0):.4f}' for fb in fbs]}"
                    for tn, fbs in telem.get("numeric_scores", {}).items()
                )
                lines.append(
                    f"  - Element {tidx}: scores=[{scores_str}], ranking_key={telem.get('ranking_key')}"
                )

        if "loss_functions" in ctx_data:
            lines.append(f"- Loss functions: {json.dumps(ctx_data['loss_functions'])}")

        if "use_textual_feedback" in ctx_data:
            lines.append(f"- use_textual_feedback: {ctx_data['use_textual_feedback']}")
        lines.append("")

    # --- Step 2: Feedbacks ---
    if "feedbacks" in row and row["feedbacks"] is not None:
        fb_data = _parse_json_field(row["feedbacks"])
        feedbacks_dict = fb_data.get("feedbacks", {})
        lines.append("### Step 2: Loss / Feedback")
        lines.append("")
        lines.append(f"- Tasks with feedback: {list(feedbacks_dict.keys())}")
        lines.append("")

        for task_name, fbs in feedbacks_dict.items():
            lines.append(f"#### Task: {task_name} ({len(fbs)} feedback(s))")
            lines.append("")
            for fidx, fb in enumerate(fbs):
                metric_dict = fb.get("metric")
                if isinstance(metric_dict, dict) and "name" in metric_dict:
                    lines.append(
                        f"- **NumericFeedback** [{fidx}]: "
                        f"{metric_dict['name']} = {metric_dict.get('value', 'N/A')}"
                    )
                if "feedback_text" in fb:
                    feedback_text = fb["feedback_text"]
                    lines.append(f"- **TextualFeedback** [{fidx}]:")
                    lines.append("")
                    if fb.get("feedback_prompt") is not None:
                        lines.append("  **Loss prompt sent to Loss LLM:**")
                        lines.append("")
                        lines.append("  ```")
                        for line in fb["feedback_prompt"].split("\n"):
                            lines.append(f"  {line}")
                        lines.append("  ```")
                        lines.append("")
                    lines.append(f"  **Loss LLM response:** {feedback_text}")
                    lines.append("")
            lines.append("")

    # --- Step 3: Gradients ---
    if "gradients" in row and row["gradients"] is not None:
        grad_data = _parse_json_field(row["gradients"])
        gradients_dict = grad_data.get("gradients", {})
        lines.append("### Step 3: Gradient")
        lines.append("")
        lines.append(f"- Tasks with gradients: {list(gradients_dict.keys())}")
        lines.append("")

        for task_name, grads in gradients_dict.items():
            lines.append(f"#### Task: {task_name} ({len(grads)} gradient(s))")
            lines.append("")
            for gidx, g in enumerate(grads):
                gradient_text = g.get("gradient_text", "N/A")
                is_llm_generated = g.get("gradient_prompt") is not None
                lines.append(
                    f"- **TextGradient** [{gidx}] (LLM-generated: {is_llm_generated})"
                )
                lines.append("")

                if is_llm_generated:
                    lines.append("  **Gradient prompt sent to Gradient LLM:**")
                    lines.append("")
                    lines.append("  ```")
                    for line in g["gradient_prompt"].split("\n"):
                        lines.append(f"  {line}")
                    lines.append("  ```")
                    lines.append("")

                lines.append(f"  **Gradient text:** {gradient_text}")
                lines.append("")

    # --- Step 4: Prompt Update ---
    if "prompt_update" in row and row["prompt_update"] is not None:
        lines.append("### Step 4: Optimizer / Prompt Update")
        lines.append("")

        meta_prompt = row.get("meta_prompt")
        if meta_prompt is not None:
            lines.append("**Meta-prompt sent to Optimizer LLM:**")
            lines.append("")
            lines.append("```")
            lines.append(meta_prompt)
            lines.append("```")
            lines.append("")

        optimizer_response = row.get("meta_prompt_response")
        if optimizer_response is not None:
            lines.append(f"**Optimizer LLM raw response:** `{optimizer_response}`")
            lines.append("")

        pu_data = _parse_json_field(row["prompt_update"])
        old_instr = pu_data.get("old_instruction", {})
        new_instr = pu_data.get("new_instruction", {})
        lines.append("**Old instructions:**")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(old_instr, indent=2, default=str))
        lines.append("```")
        lines.append("")
        lines.append("**New instructions:**")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(new_instr, indent=2, default=str))
        lines.append("```")
        lines.append("")

    # --- Algorithm State ---
    if "algorithm_state" in row and row["algorithm_state"] is not None:
        state_data = _parse_json_field(row["algorithm_state"])
        lines.append("### Algorithm State (after step)")
        lines.append("")
        lines.append(f"- Keys: {list(state_data.keys())}")
        if "trajectory" in state_data:
            lines.append(
                f"- Trajectory size: {state_data.get('trajectory_size', len(state_data['trajectory']))}"
            )
        if "previous_instructions" in state_data:
            lines.append(
                f"- Previous instructions: {json.dumps(state_data['previous_instructions'], default=str)}"
            )
        lines.append("")

    lines.append("")


# ---------------------------------------------------------------------------
# Common backbone: run one algorithm and produce markdown
# ---------------------------------------------------------------------------
def _run_algorithm_e2e(
    *,
    algo_name: str,
    task_mode: str,
    api_key: str,
    dataset: Any,
    tasks: List[Any],
    output_dir: str,
) -> Dict[str, Any]:
    """Run one algorithm for STEPS steps and return the result dict.

    Args:
        algo_name: Algorithm identifier (opro, gpo, textgrad).
        task_mode: "single" or "all" (for labeling only).
        api_key: OpenRouter API key.
        dataset: Dataset instance.
        tasks: List of Task objects.
        output_dir: Where to write run artifacts.

    Returns:
        Dict with status, results, and markdown path.

    Raises:
        ValueError: If algo_name is unrecognized.
        RuntimeError: If the algorithm training loop fails.
    """
    from runner import (
        create_gradient_llm,
        create_loss_llm,
        create_optimizer_llm,
        create_shared_limits,
        create_task_llm,
        get_initial_prompt,
        get_task_losses,
    )

    from prompt_moo.algorithm import GPO, OPRO, TextGrad
    from prompt_moo.config import temp_config

    if algo_name not in ("opro", "gpo", "textgrad"):
        raise ValueError(
            f"Unknown algorithm: {algo_name!r}. Must be 'opro', 'gpo', or 'textgrad'."
        )

    task_names = [t.task_name for t in tasks]

    with temp_config(
        substep_delay=1.0,
        verbosity=2,
    ):
        initial_prompt = get_initial_prompt(dataset=dataset, tasks=tasks)
        task_losses = get_task_losses(dataset=dataset, tasks=tasks)
        shared_limits = create_shared_limits()

        task_llm = create_task_llm(
            llm=LLM_FAMILY, api_key=api_key, limits=shared_limits
        )
        optimizer_llm = create_optimizer_llm(
            llm=LLM_FAMILY, api_key=api_key, limits=shared_limits
        )
        gradient_llm = create_gradient_llm(
            llm=LLM_FAMILY, api_key=api_key, limits=shared_limits
        )
        loss_llm = create_loss_llm(
            llm=LLM_FAMILY, api_key=api_key, limits=shared_limits
        )

        common_params: Dict[str, Any] = {
            "tasks": tasks,
            "steps": STEPS,
            "batch_size": BATCH_SIZE,
            "eval_every": EVAL_EVERY,
            "name": f"unified_e2e_{algo_name}_{task_mode}",
            "verbosity": 2,
        }
        # Algo-specific batch size defaults:
        #   OPRO/GPO: loss_batch_size and gradient_batch_size MUST equal
        #     batch_size (enforced by post_initialize). Let pre_initialize
        #     auto-fill them from batch_size by omitting them here.
        #   TextGrad: loss_batch_size=1 and gradient_batch_size=1 (enforced
        #     by post_initialize). Let pre_initialize auto-fill.

        # Merge env-var overrides (E2E_ALGO_PARAMS) into algorithm-specific
        # defaults. ALGO_PARAMS keys override the hard-coded defaults below.
        algo_overrides: Dict[str, Any] = dict(ALGO_PARAMS)

        # GPO with use_textual_feedback=True needs gradient_llm and loss_llm.
        gpo_needs_textual: bool = (
            algo_name == "gpo" and algo_overrides.get("use_textual_feedback") is True
        )

        try:
            if algo_name == "opro":
                opro_defaults: Dict[str, Any] = {"k": 3, "num_candidates": 2}
                opro_defaults.update(algo_overrides)
                algo = OPRO(
                    task_llm=task_llm,
                    optimizer_llm=optimizer_llm,
                    task_losses=task_losses,
                    **common_params,
                    **opro_defaults,
                )
            elif algo_name == "gpo":
                gpo_defaults: Dict[str, Any] = {
                    "k": 3,
                    "num_candidates": 2,
                    "trajectory_strategy": "relevance",
                }
                gpo_defaults.update(algo_overrides)
                gpo_llm_params: Dict[str, Any] = {
                    "task_llm": task_llm,
                    "optimizer_llm": optimizer_llm,
                }
                if gpo_needs_textual:
                    gpo_llm_params["gradient_llm"] = gradient_llm
                    gpo_llm_params["loss_llm"] = loss_llm
                algo = GPO(
                    task_losses=task_losses,
                    **gpo_llm_params,
                    **common_params,
                    **gpo_defaults,
                )
            elif algo_name == "textgrad":
                textgrad_defaults: Dict[str, Any] = {
                    "validation_metric": "accuracy",
                }
                textgrad_defaults.update(algo_overrides)
                algo = TextGrad(
                    task_llm=task_llm,
                    gradient_llm=gradient_llm,
                    optimizer_llm=optimizer_llm,
                    loss_llm=loss_llm,
                    **common_params,
                    **textgrad_defaults,
                )
            else:
                raise ValueError(
                    f"Unknown algorithm: {algo_name!r}. "
                    f"Must be 'opro', 'gpo', or 'textgrad'."
                )

            results = algo.train(
                dataset=dataset,
                initial_prompt=initial_prompt,
                output_dir=output_dir,
            )

            markdown_content = _build_markdown_report(
                algo_name=algo_name,
                task_mode=task_mode,
                task_names=task_names,
                output_dir=output_dir,
                num_steps=STEPS,
            )
            md_dir = E2E_OUTPUT_DIR / f"{task_mode}_{algo_name}"
            md_dir.mkdir(parents=True, exist_ok=True)
            md_path = md_dir / f"{algo_name}_{task_mode}_artifacts.md"
            md_path.write_text(markdown_content)
            print(f"\nMarkdown artifacts written to: {md_path}")

            return {
                "status": "success",
                "results": results,
                "markdown_path": str(md_path),
            }

        finally:
            task_llm.stop()
            optimizer_llm.stop()
            gradient_llm.stop()
            loss_llm.stop()


# ---------------------------------------------------------------------------
# Validation: structural checks on logged artifacts
# ---------------------------------------------------------------------------
def _validate_run_artifacts(
    *,
    output_dir: str,
    algo_name: str,
    task_names: List[str],
    num_steps: int,
) -> None:
    """Validate the structural correctness of all logged artifacts.

    Args:
        output_dir: Path to the run's output directory.
        algo_name: Algorithm identifier.
        task_names: Expected task names.
        num_steps: Expected number of steps.

    Raises:
        AssertionError: If any structural validation fails.
    """
    from prompt_moo.observability import ObservabilityManager

    run_logs = ObservabilityManager.read_run_logs(output_dir)
    assert len(run_logs) == num_steps, (
        f"Expected {num_steps} logged steps, got {len(run_logs)}"
    )

    for step_num in range(1, num_steps + 1):
        matching = run_logs[run_logs["step"] == step_num]
        assert len(matching) == 1, (
            f"Expected 1 row for step {step_num}, got {len(matching)}"
        )
        row = matching.iloc[0]

        batch_data = _parse_json_field(row["batch"])
        assert batch_data["num_samples"] == BATCH_SIZE

        pred_data = _parse_json_field(row["predictions"])
        assert pred_data["num_predictions"] > 0, (
            f"Step {step_num}: no predictions logged"
        )
        first_pred = pred_data["predictions"][0]
        assert "task_outputs" in first_pred
        assert "raw_response" in first_pred
        assert "prompt" in first_pred

        fb_data = _parse_json_field(row["feedbacks"])
        assert "feedbacks" in fb_data
        assert fb_data["num_tasks"] > 0

        grad_data = _parse_json_field(row["gradients"])
        assert "gradients" in grad_data

        assert "old_prompt_template" in row and row["old_prompt_template"] is not None
        assert "meta_prompt" in row and row["meta_prompt"] is not None
        assert "new_prompt_template" in row and row["new_prompt_template"] is not None

        pu_data = _parse_json_field(row["prompt_update"])
        assert "old_instruction" in pu_data
        assert "new_instruction" in pu_data
        new_instr = pu_data["new_instruction"]
        for tn in task_names:
            assert tn in new_instr, (
                f"Step {step_num}: task '{tn}' missing from new_instruction. "
                f"Keys: {list(new_instr.keys())}"
            )

    summary_path = os.path.join(output_dir, "run_summary.json")
    assert os.path.exists(summary_path)
    with open(summary_path, "r") as f:
        summary = json.load(f)
    assert "completed_at" in summary, "Run should be marked as completed"
    assert summary["total_steps"] == num_steps


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------
@skip_no_api_key
class TestUnifiedE2E:
    """Unified E2E tests: one test_* per algorithm x task_mode, calling the common backbone.

    Each algorithm has two tests: ``single_task`` (coherence only) and
    ``all_task`` (all 4 SummEval tasks).  Use ``--algorithms=opro,gpo`` to
    run only selected algorithms, or ``-k "all_task"`` to run only the
    all-task mode across whichever algorithms are selected.
    """

    @pytest.fixture(scope="class")
    def api_key(self):
        key = os.environ.get("OPENROUTER_API_KEY")
        if key is None or len(key) == 0:
            pytest.skip("OPENROUTER_API_KEY not set")
        return key

    @pytest.fixture(scope="class")
    def summeval_dataset(self):
        from dataset import SummEval

        return SummEval(data_dir="expt")

    # ------------------------------------------------------------------
    # Per-algorithm test functions (selectable via pytest -k or --algorithms)
    # ------------------------------------------------------------------
    def _skip_unless_requested(self, request: Any, algo_name: str) -> None:
        """Skip this test if --algorithms was provided and does not include algo_name."""
        requested = _get_requested_algorithms(request)
        if algo_name not in requested:
            pytest.skip(
                f"Skipped: --algorithms={request.config.getoption('--algorithms')} "
                f"does not include '{algo_name}'"
            )

    @pytest.mark.timeout(3600 * 12)
    def test_opro_single_task_e2e(self, request, api_key, summeval_dataset):
        """OPRO: single-task (coherence), full artifact dump to markdown."""
        self._skip_unless_requested(request, "opro")
        coherence_task = [
            t for t in summeval_dataset.tasks if t.task_name == "coherence"
        ][0]
        self._run_single_mode(
            algo_name="opro",
            task_mode="single",
            tasks=[coherence_task],
            api_key=api_key,
            dataset=summeval_dataset,
        )

    @pytest.mark.timeout(3600 * 12)
    def test_opro_all_task_e2e(self, request, api_key, summeval_dataset):
        """OPRO: all-task (all 4 SummEval tasks), full artifact dump to markdown."""
        self._skip_unless_requested(request, "opro")
        self._run_single_mode(
            algo_name="opro",
            task_mode="all",
            tasks=summeval_dataset.tasks,
            api_key=api_key,
            dataset=summeval_dataset,
        )

    @pytest.mark.timeout(3600 * 12)
    def test_gpo_single_task_e2e(self, request, api_key, summeval_dataset):
        """GPO: single-task (coherence), full artifact dump to markdown."""
        self._skip_unless_requested(request, "gpo")
        coherence_task = [
            t for t in summeval_dataset.tasks if t.task_name == "coherence"
        ][0]
        self._run_single_mode(
            algo_name="gpo",
            task_mode="single",
            tasks=[coherence_task],
            api_key=api_key,
            dataset=summeval_dataset,
        )

    @pytest.mark.timeout(3600 * 12)
    def test_gpo_all_task_e2e(self, request, api_key, summeval_dataset):
        """GPO: all-task (all 4 SummEval tasks), full artifact dump to markdown."""
        self._skip_unless_requested(request, "gpo")
        self._run_single_mode(
            algo_name="gpo",
            task_mode="all",
            tasks=summeval_dataset.tasks,
            api_key=api_key,
            dataset=summeval_dataset,
        )

    @pytest.mark.timeout(3600 * 12)
    def test_textgrad_single_task_e2e(self, request, api_key, summeval_dataset):
        """TextGrad: single-task (coherence), full artifact dump to markdown."""
        self._skip_unless_requested(request, "textgrad")
        coherence_task = [
            t for t in summeval_dataset.tasks if t.task_name == "coherence"
        ][0]
        self._run_single_mode(
            algo_name="textgrad",
            task_mode="single",
            tasks=[coherence_task],
            api_key=api_key,
            dataset=summeval_dataset,
        )

    @pytest.mark.timeout(3600 * 12)
    def test_textgrad_all_task_e2e(self, request, api_key, summeval_dataset):
        """TextGrad: all-task (all 4 SummEval tasks), full artifact dump to markdown."""
        self._skip_unless_requested(request, "textgrad")
        self._run_single_mode(
            algo_name="textgrad",
            task_mode="all",
            tasks=summeval_dataset.tasks,
            api_key=api_key,
            dataset=summeval_dataset,
        )

    def _run_single_mode(
        self,
        *,
        algo_name: str,
        task_mode: str,
        tasks: List[Any],
        api_key: str,
        dataset: Any,
    ) -> None:
        """Run one algorithm in one task mode, validate, and dump markdown.

        Args:
            algo_name: Algorithm identifier.
            task_mode: "single" or "all".
            tasks: List of Task objects.
            api_key: OpenRouter API key.
            dataset: SummEval dataset instance.
        """
        task_names = [t.task_name for t in tasks]
        output_dir = str(E2E_OUTPUT_DIR / f"{task_mode}_{algo_name}" / "run")

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        print(f"\n{'#' * 70}")
        print(f"  {algo_name.upper()} — {task_mode}-task ({', '.join(task_names)})")
        print(f"{'#' * 70}")

        result = _run_algorithm_e2e(
            algo_name=algo_name,
            task_mode=task_mode,
            api_key=api_key,
            dataset=dataset,
            tasks=tasks,
            output_dir=output_dir,
        )

        assert result["status"] == "success", f"{algo_name} {task_mode}-task failed"
        assert "results" in result
        assert "markdown_path" in result
        assert os.path.exists(result["markdown_path"])

        _validate_run_artifacts(
            output_dir=output_dir,
            algo_name=algo_name,
            task_names=task_names,
            num_steps=STEPS,
        )

        print(f"  Artifacts: {result['markdown_path']}")
        print(f"  All validations passed for {algo_name} {task_mode}-task")
