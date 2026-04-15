"""
End-to-end test for TextGrad multi-task strategy configurations.

TextGrad supports 4 valid strategy combinations that control how tasks are
grouped at each LLM stage (Loss -> Gradient -> Optimizer).  The constraint
is: once a stage combines tasks, all downstream stages must also combine
(because combined output cannot be split back into per-task pieces).

Valid configurations::

    Config   Loss          Gradient      Optimizer     LLM calls (4 tasks, batch=N)
    ------   -----------   -----------   -----------   ----------------------------
    S/S/S    separate      separate      separate      Loss: 4N, Grad: 4N, Opt: 4
    S/S/C    separate      separate      combine       Loss: 4N, Grad: 4N, Opt: 1  [default]
    S/C/C    separate      combine       combine       Loss: 4N, Grad:  N, Opt: 1
    C/C/C    combine       combine       combine       Loss:  N, Grad:  N, Opt: 1

Invalid (enforced by TextGrad.post_initialize): C/S/*, C/C/S, S/C/S.

This test runs all 4 valid configs with real LLM calls (3 steps, batch=2,
4 SummEval tasks) and dumps a markdown report showing the exact prompt sent
to each LLM at every step.  The reports are written to:

    tests/e2e_outputs/textgrad_modes/{SSS,SSC,SCC,CCC}/textgrad_{config}_prompts.md

What to verify in each report:

    - **Loss (Step 2):** In separate mode, each loss prompt mentions ONLY one
      task's prediction + ground truth.  In combine mode, all tasks appear.
    - **Gradient (Step 3):** In separate mode, <LM_OUTPUT> and <VARIABLE>
      show one task; <OBJECTIVE_FUNCTION> carries one task's loss feedback.
      In combine mode, all tasks appear in all three tags.  Critical: in SCC
      mode, <OBJECTIVE_FUNCTION> must show all 4 tasks' feedback per sample
      (not just the first task from dict iteration order).
    - **Optimizer (Step 4):** In separate mode, <LM_OUTPUT> inside <CONTEXT>
      shows only the target task's prediction value (task_name_filter).
      In combine mode, all tasks appear.

Configuration via environment variables::

    E2E_LLM_FAMILY=qwen3    LLM family key (default: qwen3)

Usage::

    # Run all 4 configs (~25 min total):
    pytest tests/test_e2e_textgrad_modes.py -s -v --timeout=900

    # Run one specific config:
    pytest tests/test_e2e_textgrad_modes.py -s -v --timeout=900 -k "SSS"
    pytest tests/test_e2e_textgrad_modes.py -s -v --timeout=900 -k "SCC"

    # Use a different LLM family:
    E2E_LLM_FAMILY=llama3.1 pytest tests/test_e2e_textgrad_modes.py -s -v --timeout=900

Requires OPENROUTER_API_KEY in .env.

See also:
    - ``tests/test_textgrad_strategies.py`` for unit tests of valid/invalid
      strategy validation (no LLM calls, runs in <30s).
    - ``src/prompt_moo/algorithm/textgrad.py`` TextGrad class docstring for
      the strategy constraint rules.
"""

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "expt"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tests.conftest import skip_no_api_key

STEPS = 3
BATCH_SIZE = 2
EVAL_EVERY = 99999
LLM_FAMILY = os.environ.get("E2E_LLM_FAMILY", "qwen3")

OUTPUT_BASE = Path(__file__).parent / "e2e_outputs" / "textgrad_modes"

VALID_CONFIGS: List[Tuple[str, str, str, str]] = [
    ("SSS", "separate_tasks", "separate_tasks", "separate_tasks"),
    ("SSC", "separate_tasks", "separate_tasks", "combine_all_tasks"),
    ("SCC", "separate_tasks", "combine_all_tasks", "combine_all_tasks"),
    ("CCC", "combine_all_tasks", "combine_all_tasks", "combine_all_tasks"),
]


def _extract_prompts_from_run(*, output_dir: str, num_steps: int) -> str:
    """Read run_logs and extract the LLM prompts at each stage into markdown."""
    from prompt_moo.observability import ObservabilityManager

    run_logs = ObservabilityManager.read_run_logs(output_dir)
    lines: List[str] = []

    for step_num in range(1, num_steps + 1):
        matching = run_logs[run_logs["step"] == step_num]
        if len(matching) == 0:
            lines.append(f"## Step {step_num} — MISSING")
            continue

        row = matching.iloc[0]
        lines.append(f"## Step {step_num}")
        lines.append("")

        # Task LLM prompts (first prediction only, for brevity)
        if "predictions" in row and row["predictions"] is not None:
            pred_data = (
                json.loads(row["predictions"])
                if isinstance(row["predictions"], str)
                else row["predictions"]
            )
            preds = pred_data.get("predictions", [])
            if len(preds) > 0:
                p = preds[0]
                lines.append("### Task LLM (Step 1: Predict)")
                lines.append("")
                lines.append("**Prompt (sample 0):**")
                lines.append("```")
                lines.append(p.get("prompt", "N/A"))
                lines.append("```")
                lines.append(f"**Raw response:** `{p.get('raw_response', 'N/A')}`")
                lines.append(f"**Parsed:** `{json.dumps(p.get('task_outputs', {}))}`")
                lines.append("")

        # Loss LLM prompts (first feedback per task, for brevity)
        if "feedbacks" in row and row["feedbacks"] is not None:
            fb_data = (
                json.loads(row["feedbacks"])
                if isinstance(row["feedbacks"], str)
                else row["feedbacks"]
            )
            feedbacks_dict = fb_data.get("feedbacks", {})
            lines.append("### Loss LLM (Step 2: Compute Loss)")
            lines.append("")
            for task_name, fbs in feedbacks_dict.items():
                if len(fbs) > 0:
                    fb = fbs[0]
                    if "feedback_prompt" in fb and fb["feedback_prompt"] is not None:
                        lines.append(f"**Task: {task_name} — Loss prompt (sample 0):**")
                        lines.append("```")
                        lines.append(fb["feedback_prompt"])
                        lines.append("```")
                        lines.append(f"**Response:** {fb['feedback_text']}")
                        lines.append("")

        # Gradient LLM prompts (first gradient per task)
        if "gradients" in row and row["gradients"] is not None:
            grad_data = (
                json.loads(row["gradients"])
                if isinstance(row["gradients"], str)
                else row["gradients"]
            )
            gradients_dict = grad_data.get("gradients", {})
            lines.append("### Gradient LLM (Step 3: Compute Gradient)")
            lines.append("")
            for task_name, grads in gradients_dict.items():
                if len(grads) > 0:
                    g = grads[0]
                    if g.get("gradient_prompt") is not None:
                        lines.append(
                            f"**Task: {task_name} — Gradient prompt (sample 0):**"
                        )
                        lines.append("```")
                        lines.append(g["gradient_prompt"])
                        lines.append("```")
                        lines.append(f"**Response:** {g['gradient_text']}")
                        lines.append("")

        if "meta_prompt" in row and row["meta_prompt"] is not None:
            meta_prompt = row["meta_prompt"]
            if meta_prompt is not None:
                lines.append("### Optimizer LLM (Step 4: Optimize Prompt)")
                lines.append("")
                lines.append("**Meta-prompt:**")
                lines.append("```")
                lines.append(meta_prompt)
                lines.append("```")
                lines.append(f"**Response:** `{pu_data['optimizer_response']}`")
                lines.append("")
                lines.append("**New instructions:**")
                lines.append("```json")
                lines.append(json.dumps(pu_data.get("new_instruction", {}), indent=2))
                lines.append("```")
                lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def _run_one_config(
    *,
    config_name: str,
    loss_strategy: str,
    gradient_strategy: str,
    optimizer_strategy: str,
    api_key: str,
    dataset: Any,
    tasks: List[Any],
) -> str:
    """Run TextGrad with one config, return path to markdown artifact."""
    from runner import (
        create_gradient_llm,
        create_loss_llm,
        create_optimizer_llm,
        create_shared_limits,
        create_task_llm,
        get_initial_prompt,
    )
    from prompt_moo.algorithm import TextGrad
    from prompt_moo.config import temp_config

    output_dir = str(OUTPUT_BASE / config_name / "run")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    with temp_config(
        substep_delay=1.0,
        verbosity=2,
    ):
        initial_prompt = get_initial_prompt(dataset=dataset, tasks=tasks)
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

        try:
            algo = TextGrad(
                task_llm=task_llm,
                gradient_llm=gradient_llm,
                optimizer_llm=optimizer_llm,
                loss_llm=loss_llm,
                tasks=tasks,
                steps=STEPS,
                batch_size=BATCH_SIZE,
                eval_every=EVAL_EVERY,
                name=f"textgrad_modes_{config_name}",
                verbosity=2,
                validation_metric="accuracy",
                loss_task_strategy=loss_strategy,
                gradient_task_strategy=gradient_strategy,
                optimizer_task_strategy=optimizer_strategy,
            )

            algo.train(
                dataset=dataset,
                initial_prompt=initial_prompt,
                output_dir=output_dir,
            )

            report = _extract_prompts_from_run(output_dir=output_dir, num_steps=STEPS)
            header = (
                f"# TextGrad Mode: {config_name}\n\n"
                f"- loss_task_strategy: `{loss_strategy}`\n"
                f"- gradient_task_strategy: `{gradient_strategy}`\n"
                f"- optimizer_task_strategy: `{optimizer_strategy}`\n"
                f"- steps: {STEPS}, batch_size: {BATCH_SIZE}\n"
                f"- tasks: {', '.join(t.task_name for t in tasks)}\n\n"
            )
            md_path = OUTPUT_BASE / config_name / f"textgrad_{config_name}_prompts.md"
            md_path.write_text(header + report)
            print(f"\n  Artifact: {md_path}")
            return str(md_path)

        finally:
            task_llm.stop()
            optimizer_llm.stop()
            gradient_llm.stop()
            loss_llm.stop()


@skip_no_api_key
class TestTextGradModes:
    """Run all 4 valid TextGrad multi-task strategy configs."""

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

    @pytest.mark.timeout(900)
    @pytest.mark.parametrize(
        "config_name,loss_s,grad_s,opt_s",
        VALID_CONFIGS,
        ids=[c[0] for c in VALID_CONFIGS],
    )
    def test_textgrad_mode(
        self, config_name, loss_s, grad_s, opt_s, api_key, summeval_dataset
    ):
        tasks = summeval_dataset.tasks
        print(f"\n{'#' * 60}")
        print(f"  TextGrad config: {config_name} (L={loss_s}, G={grad_s}, O={opt_s})")
        print(f"{'#' * 60}")

        md_path = _run_one_config(
            config_name=config_name,
            loss_strategy=loss_s,
            gradient_strategy=grad_s,
            optimizer_strategy=opt_s,
            api_key=api_key,
            dataset=summeval_dataset,
            tasks=tasks,
        )

        assert os.path.exists(md_path)
        content = Path(md_path).read_text()
        assert len(content) > 100
        assert "## Step 0" in content
        assert "## Step 1" in content
        assert "## Step 2" in content
        print(f"  PASSED: {config_name}")
