"""
Tests for the GPO algorithm implementation (Tang et al., AAAI 2025).

Verifies that:

1. Default configuration matches the paper:
   - k=7, warmup_steps=0, final_step_size=5 (20% of initial 25)
   - use_textual_feedback=False (no Loss LLM, no Gradient LLM)
2. Step 2 (Loss): deterministic numeric scoring only, NO LLM calls
3. Step 3 (Gradient): deterministic score summary, NO LLM calls
4. Step 4 (Optimizer): meta-prompt contains trajectory (prompt, score) pairs
   + current prompt template + edit distance constraint, NO "gradient" section
5. Trajectory stores only (instructions, scores), NO textual gradients
6. Hybrid mode (use_textual_feedback=True) enables the full LLM pipeline
7. E2E: the full pipeline runs, produces correct parquet files, and every
   logged artifact matches paper expectations
"""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Union

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prompt_moo.data_input import Dataset
from prompt_moo.data_structures import (
    Batch,
    DatasetSample,
    NumericFeedback,
    PredictionResult,
    Task,
    TextGradient,
    TextualFeedback,
)
from prompt_moo.metrics import Accuracy
from prompt_moo.prompt_template import PromptTemplate
from prompt_moo.algorithm.gpo import GPOTrajectoryElement
from prompt_moo.prompt_trajectory import PromptTrajectory

TASKS = [
    Task(
        task_name="fluency",
        task_description="Evaluate fluency",
        task_instruction="Rate from 1 to 5.",
        gt_col="fluency",
    ),
    Task(
        task_name="consistency",
        task_description="Evaluate consistency",
        task_instruction="Rate from 1 to 5.",
        gt_col="consistency",
    ),
]

SKELETON = (
    "Evaluate the summary. Output JSON with the requested task scores. "
    "Do NOT include reasoning or explanations."
)

TASK_LOSSES = {"fluency": "accuracy", "consistency": "accuracy"}

INPUT_COL_LABELS = {
    "machine_summary": "Summary",
    "text": "Source Text",
}


def _make_prompt_template() -> PromptTemplate:
    return PromptTemplate(
        skeleton=SKELETON,
        instruction={t.task_name: t.task_instruction for t in TASKS},
        tasks=TASKS,
        input_col_labels=INPUT_COL_LABELS,
    )


def _make_samples() -> List[DatasetSample]:
    return [
        DatasetSample(
            sample_id="s0",
            inputs={"machine_summary": "The cat sat.", "text": "A story about a cat."},
            ground_truths={"fluency": 4, "consistency": 5},
        ),
        DatasetSample(
            sample_id="s1",
            inputs={"machine_summary": "Dogs are great.", "text": "Dogs are loyal."},
            ground_truths={"fluency": 3, "consistency": 2},
        ),
    ]


def _make_predictions() -> List[PredictionResult]:
    return [
        PredictionResult(
            sample_id="s0",
            prompt="test_prompt",
            task_outputs={"fluency": 3, "consistency": 5},
            raw_response='{"fluency": 3, "consistency": 5}',
        ),
        PredictionResult(
            sample_id="s1",
            prompt="test_prompt",
            task_outputs={"fluency": 5, "consistency": 4},
            raw_response='{"fluency": 5, "consistency": 4}',
        ),
    ]


# -----------------------------------------------------------------------
# 1. Paper defaults
# -----------------------------------------------------------------------


class TestGPOPaperDefaults:
    """GPO defaults must match the paper's winning configuration."""

    def test_k_equals_7(self):
        from prompt_moo.algorithm import GPO

        gpo = GPO(
            task_llm=None,
            optimizer_llm=None,
            tasks=TASKS,
            steps=1,
            batch_size=2,
            loss_batch_size=2,
            gradient_batch_size=2,
            eval_every=1,
            name="t",
            task_losses=TASK_LOSSES,
        )
        assert gpo.k == 7

    def test_warmup_steps_defaults_to_zero(self):
        from prompt_moo.algorithm import GPO

        gpo = GPO(
            task_llm=None,
            optimizer_llm=None,
            tasks=TASKS,
            steps=1,
            batch_size=2,
            loss_batch_size=2,
            gradient_batch_size=2,
            eval_every=1,
            name="t",
            task_losses=TASK_LOSSES,
        )
        assert gpo.warmup_steps == 0

    def test_final_step_size_is_20_percent_of_initial(self):
        from prompt_moo.algorithm import GPO

        gpo = GPO(
            task_llm=None,
            optimizer_llm=None,
            tasks=TASKS,
            steps=1,
            batch_size=2,
            loss_batch_size=2,
            gradient_batch_size=2,
            eval_every=1,
            name="t",
            task_losses=TASK_LOSSES,
        )
        assert gpo.initial_step_size == 25
        assert gpo.final_step_size == 5
        assert gpo.final_step_size == pytest.approx(gpo.initial_step_size * 0.20)

    def test_use_textual_feedback_defaults_false(self):
        from prompt_moo.algorithm import GPO

        gpo = GPO(
            task_llm=None,
            optimizer_llm=None,
            tasks=TASKS,
            steps=1,
            batch_size=2,
            loss_batch_size=2,
            gradient_batch_size=2,
            eval_every=1,
            name="t",
            task_losses=TASK_LOSSES,
        )
        assert gpo.use_textual_feedback is False

    def test_num_task_demonstrations_defaults_to_3(self):
        from prompt_moo.algorithm import GPO

        gpo = GPO(
            task_llm=None,
            optimizer_llm=None,
            tasks=TASKS,
            steps=1,
            batch_size=2,
            loss_batch_size=2,
            gradient_batch_size=2,
            eval_every=1,
            name="t",
            task_losses=TASK_LOSSES,
        )
        assert gpo.num_task_demonstrations == 3

    def test_algorithm_context_propagates_defaults(self):
        from prompt_moo.algorithm import GPO

        gpo = GPO(
            task_llm=None,
            optimizer_llm=None,
            tasks=TASKS,
            steps=10,
            batch_size=2,
            loss_batch_size=2,
            gradient_batch_size=2,
            eval_every=5,
            name="t",
            task_losses=TASK_LOSSES,
        )
        gpo._input_col_labels = {}
        gpo._task_demonstrations = []
        ctx = gpo._get_algorithm_context(step=0)
        assert ctx["use_textual_feedback"] is False
        for task_name, cfg in ctx["loss_functions"].items():
            assert cfg["use_textual"] is False


# -----------------------------------------------------------------------
# 2. Loss: numeric only
# -----------------------------------------------------------------------


class TestGPOLossComputation:
    """TaskLevelLossComputer produces ONLY numeric feedback by default (GPO context)."""

    def test_numeric_only_no_textual(self):
        from prompt_moo.loss_computer import TaskLevelLossComputer

        loss_comp = TaskLevelLossComputer()
        feedbacks = loss_comp.compute(
            predictions=_make_predictions(),
            ground_truths=_make_samples(),
            tasks=TASKS,
            prompt_template=_make_prompt_template(),
            llm_pool=None,
            loss_batch_size=4,
            loss_functions={
                "fluency": {"metric": "accuracy", "use_textual": False},
                "consistency": {"metric": "accuracy", "use_textual": False},
            },
        )
        for task, fb_list in feedbacks.items():
            for fb in fb_list:
                assert isinstance(fb, NumericFeedback), (
                    f"Expected NumericFeedback, got {type(fb).__name__}"
                )


# -----------------------------------------------------------------------
# 3. Gradient: deterministic score summary, no LLM
# -----------------------------------------------------------------------


class TestGPOGradientComputer:
    """GPO Gradient Computer produces a deterministic summary, no LLM call."""

    def _make_numeric_feedbacks(self) -> Dict[Task, list]:
        return {
            TASKS[0]: [
                NumericFeedback(
                    task_name="fluency",
                    metric=Accuracy(value=0.50),
                    aggregated_from_samples=["s0", "s1"],
                ),
            ],
            TASKS[1]: [
                NumericFeedback(
                    task_name="consistency",
                    metric=Accuracy(value=0.75),
                    aggregated_from_samples=["s0", "s1"],
                ),
            ],
        }

    def test_no_llm_call_needed(self):
        from prompt_moo.algorithm.gpo import GPOGradientComputer

        gradients = GPOGradientComputer().compute(
            feedbacks=self._make_numeric_feedbacks(),
            prompt_template=_make_prompt_template(),
            tasks=TASKS,
            llm_pool=None,
            gradient_batch_size=4,
            use_textual_feedback=False,
        )
        for task, grad_list in gradients.items():
            assert len(grad_list) == 1
            assert grad_list[0].gradient_prompt is None

    def test_gradient_text_contains_scores(self):
        from prompt_moo.algorithm.gpo import GPOGradientComputer

        gradients = GPOGradientComputer().compute(
            feedbacks=self._make_numeric_feedbacks(),
            prompt_template=_make_prompt_template(),
            tasks=TASKS,
            llm_pool=None,
            gradient_batch_size=4,
            use_textual_feedback=False,
        )
        assert "50.0" in gradients[TASKS[0]][0].gradient_text
        assert "75.0" in gradients[TASKS[1]][0].gradient_text


# -----------------------------------------------------------------------
# 4. Optimizer meta-prompt: trajectory + edit distance, no gradients
# -----------------------------------------------------------------------


class TestGPOOptimizerMetaPrompt:
    """GPO optimizer meta-prompt matches the paper: trajectory + edit distance."""

    def _build_meta_prompt(
        self, *, use_textual_feedback: bool = False, include_demos: bool = False
    ) -> str:
        from prompt_moo.algorithm.gpo import GPOOptimizer

        optimizer = GPOOptimizer()
        trajectory = PromptTrajectory(
            k=7, order="best_to_worst", metric_priority="accuracy"
        )
        trajectory.push(
            GPOTrajectoryElement(
                numeric_scores={
                    "fluency": [
                        NumericFeedback(
                            task_name="fluency",
                            metric=Accuracy(value=0.50),
                            aggregated_from_samples=[],
                        ),
                    ]
                },
                instructions={
                    "fluency": "Rate fluency.",
                    "consistency": "Rate consistency.",
                },
            )
        )

        grad_prompt = "some LLM prompt" if use_textual_feedback else None
        gradients = {
            TASKS[0]: [
                TextGradient(
                    task_name="fluency",
                    gradient_text="Add calibration guidance for fluency."
                    if use_textual_feedback
                    else "Scores: accuracy: 50.0",
                    based_on_feedbacks=[],
                    gradient_prompt=grad_prompt,
                )
            ],
        }

        kwargs = dict(
            gradients=gradients,
            current_prompt=_make_prompt_template(),
            tasks=TASKS,
            batch=Batch(step=5, samples=_make_samples()),
            trajectory=trajectory,
            trajectory_strategy="importance",
            warmup_steps=0,
            total_steps=100,
            initial_step_size=25,
            final_step_size=5,
            top_k_retrieve=7,
            use_textual_feedback=use_textual_feedback,
            input_col_labels=INPUT_COL_LABELS,
            loss_functions={t.task_name: {"metric": "accuracy"} for t in TASKS},
        )
        if include_demos:
            kwargs["task_demonstrations"] = _make_samples()[:2]

        return optimizer.create_meta_prompt(**kwargs)

    def test_default_omits_gradient_section(self):
        meta = self._build_meta_prompt(use_textual_feedback=False)
        assert "Gradients / Improvement suggestions" not in meta

    def test_contains_trajectory(self):
        meta = self._build_meta_prompt(use_textual_feedback=False)
        assert "previous instructions with their scores" in meta
        assert "Scores:" in meta or "Score:" in meta

    def test_contains_edit_distance_word_count(self):
        meta = self._build_meta_prompt(use_textual_feedback=False)
        assert "change up to" in meta
        assert "words" in meta

    def test_contains_analyze_instruction(self):
        meta = self._build_meta_prompt(use_textual_feedback=False)
        assert "Carefully analyze the previous instructions" in meta
        assert "<Instruction_" in meta

    def test_hybrid_includes_gradient_section(self):
        meta = self._build_meta_prompt(use_textual_feedback=True)
        assert "Gradients / Improvement suggestions" in meta
        assert "Add calibration guidance" in meta

    def test_meta_prompt_includes_task_demonstrations(self):
        meta = self._build_meta_prompt(include_demos=True)
        assert "exemplars" in meta.lower() or "example" in meta.lower()
        assert "The cat sat." in meta or "Cat sat." in meta
        assert "fluency:" in meta.lower()
        assert "consistency:" in meta.lower()

    def test_meta_prompt_without_demos_has_no_examples_section(self):
        meta = self._build_meta_prompt(include_demos=False)
        assert "exemplars" not in meta.lower() or "Example 1:" not in meta


# -----------------------------------------------------------------------
# 5. Trajectory: only (instructions, scores) by default
#    Uses GPOTrajectoryElement
# -----------------------------------------------------------------------


class TestGPOTrajectory:
    """Trajectory element display matches the paper's (Prompt, Score) format."""

    def test_default_str_shows_only_instructions_and_scores(self):
        elem = GPOTrajectoryElement(
            numeric_scores={
                "fluency": [
                    NumericFeedback(
                        task_name="fluency",
                        metric=Accuracy(value=0.65),
                        aggregated_from_samples=[],
                    ),
                ]
            },
            instructions={"fluency": "Rate from 1 to 5."},
        )
        s = str(elem)
        assert "Prompt:" in s
        assert "Score" in s
        assert "fluency" in s


# -----------------------------------------------------------------------
# 6. E2E: full pipeline run with mock dataset, validate parquet output
# -----------------------------------------------------------------------


class _MockDataset(Dataset):
    """Minimal Dataset subclass for testing without real data files."""

    _allow_subclass_override = True

    dataset_name: ClassVar[str] = "MockSummEval"
    train_size: ClassVar[int] = 4
    test_size: ClassVar[int] = 4
    input_cols: ClassVar[List[str]] = ["machine_summary", "text"]
    gt_cols: ClassVar[List[str]] = ["fluency", "consistency"]
    input_col_labels: ClassVar[Dict[str, str]] = INPUT_COL_LABELS
    task_output_formats: ClassVar[dict] = {
        "fluency": "1|2|3|4|5",
        "consistency": "1|2|3|4|5",
    }
    task_losses: ClassVar[Dict[str, str]] = TASK_LOSSES
    tasks: ClassVar[List[Task]] = TASKS
    num_classes: ClassVar[Dict[str, int]] = {}

    @classmethod
    def setup(cls, base_dir: str):
        pass

    def train(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "machine_summary": "Cat sat.",
                    "text": "Cat story.",
                    "fluency": 4,
                    "consistency": 5,
                },
                {
                    "machine_summary": "Dog ran.",
                    "text": "Dog story.",
                    "fluency": 3,
                    "consistency": 2,
                },
                {
                    "machine_summary": "Bird flew.",
                    "text": "Bird story.",
                    "fluency": 5,
                    "consistency": 4,
                },
                {
                    "machine_summary": "Fish swam.",
                    "text": "Fish story.",
                    "fluency": 2,
                    "consistency": 3,
                },
            ]
        )

    def test(self) -> pd.DataFrame:
        return self.train()


class _MockLLMPool:
    """Fake LLM pool that returns deterministic responses without network calls."""

    def __init__(self, *, role: str):
        self._role = role

    def call_llm_batch(self, prompts, verbosity=0, validator=None, **kwargs):
        responses = []
        for prompt in prompts:
            resp = self._generate(prompt)
            if validator is not None:
                resp = validator(resp)
            responses.append(resp)
        return _MockFuture(responses)

    def _generate(self, prompt: str) -> str:
        if self._role == "task":
            return '{"fluency": 3, "consistency": 4}'
        if self._role == "optimizer":
            return "Here is the improved instruction:\n" + json.dumps(
                {
                    "instructions": {
                        "fluency": "Improved fluency instruction from optimizer.",
                        "consistency": "Improved consistency instruction from optimizer.",
                    }
                }
            )
        return "Mock response"

    def stop(self):
        pass


class _MockFuture:
    def __init__(self, value):
        self._value = value

    def result(self, timeout=None):
        return self._value


class TestGPOE2E:
    """End-to-end test: runs the full GPO pipeline with mock LLMs,
    then reads every parquet file and validates each step's data."""

    @pytest.fixture()
    def output_dir(self):
        d = tempfile.mkdtemp(prefix="gpo_e2e_")
        yield d
        shutil.rmtree(d, ignore_errors=True)

    def test_e2e_default_mode(self, output_dir):
        from prompt_moo.algorithm import GPO
        from prompt_moo.config import temp_config
        from prompt_moo.observability import ObservabilityManager

        dataset = _MockDataset(data_dir=".")
        task_llm = _MockLLMPool(role="task")
        optimizer_llm = _MockLLMPool(role="optimizer")

        steps = 3
        with temp_config(substep_delay=0, verbosity=0):
            algo = GPO(
                task_llm=task_llm,
                optimizer_llm=optimizer_llm,
                tasks=TASKS,
                steps=steps,
                batch_size=2,
                loss_batch_size=2,
                gradient_batch_size=2,
                eval_every=99,
                name="e2e_test",
                task_losses=TASK_LOSSES,
                verbosity=2,
            )

            results = algo.train(
                dataset=dataset,
                initial_prompt=_make_prompt_template(),
                output_dir=output_dir,
            )

        assert results is not None
        assert "output_dir" in results
        assert "final_prompt" in results

        # --- Read and validate parquet files ---
        run_logs = ObservabilityManager.read_run_logs(output_dir)
        assert len(run_logs) == steps, f"Expected {steps} rows, got {len(run_logs)}"

        for step_num in range(1, steps + 1):
            row = run_logs[run_logs["step"] == step_num].iloc[0]

            # --- batch ---
            batch_data = (
                json.loads(row["batch"])
                if isinstance(row["batch"], str)
                else row["batch"]
            )
            assert batch_data["num_samples"] == 2

            # --- predictions ---
            pred_data = (
                json.loads(row["predictions"])
                if isinstance(row["predictions"], str)
                else row["predictions"]
            )
            assert pred_data["num_predictions"] > 0
            first_pred = pred_data["predictions"][0]
            assert "task_outputs" in first_pred
            assert "fluency" in first_pred["task_outputs"]
            assert "consistency" in first_pred["task_outputs"]
            assert "prompt" in first_pred
            prompt_text = first_pred["prompt"]
            assert "## Instructions:" in prompt_text
            assert "## Sample" in prompt_text

            # --- feedbacks: ONLY numeric, NO textual ---
            fb_data = (
                json.loads(row["feedbacks"])
                if isinstance(row["feedbacks"], str)
                else row["feedbacks"]
            )
            feedbacks_dict = fb_data["feedbacks"]
            assert "fluency" in feedbacks_dict
            assert "consistency" in feedbacks_dict
            for task_name, fbs in feedbacks_dict.items():
                for fb in fbs:
                    metric_dict = fb.get("metric")
                    assert (
                        metric_dict is not None and metric_dict.get("name") is not None
                    ), (
                        f"Step {step_num}, task {task_name}: feedback should be NumericFeedback "
                        f"(has metric.name), got keys: {list(fb.keys())}"
                    )
                    assert "feedback_text" not in fb, (
                        f"Step {step_num}, task {task_name}: GPO default should NOT produce "
                        f"TextualFeedback, but found 'feedback_text' key"
                    )

            # --- gradients: deterministic summary, no LLM prompt ---
            grad_data = (
                json.loads(row["gradients"])
                if isinstance(row["gradients"], str)
                else row["gradients"]
            )
            gradients_dict = grad_data["gradients"]
            for task_name, grads in gradients_dict.items():
                for g in grads:
                    assert g["gradient_prompt"] is None, (
                        f"Step {step_num}, task {task_name}: GPO default gradient should have "
                        f"gradient_prompt=None (no LLM call), got: {g['gradient_prompt']}"
                    )
                    assert "accuracy" in g["gradient_text"].lower()

            # --- prompt_update: meta-prompt has trajectory, no gradient section ---
            meta_prompt = row["meta_prompt"]
            assert meta_prompt is not None
            assert "previous instructions with their scores" in meta_prompt
            assert "Gradients / Improvement suggestions" not in meta_prompt, (
                f"Step {step_num}: GPO default meta-prompt should NOT contain gradient section"
            )
            assert "change up to" in meta_prompt, (
                "Meta-prompt should contain edit distance word count"
            )
            assert "Carefully analyze the previous instructions" in meta_prompt
            assert (
                "exemplar" in meta_prompt.lower() or "example" in meta_prompt.lower()
            ), f"Step {step_num}: GPO meta-prompt should contain task demonstrations"

            pu_data = (
                json.loads(row["prompt_update"])
                if isinstance(row["prompt_update"], str)
                else row["prompt_update"]
            )
            new_instr = pu_data["new_instruction"]
            assert "fluency" in new_instr
            assert "consistency" in new_instr

            # --- algorithm_context: use_textual_feedback=False ---
            ctx_data = (
                json.loads(row["algorithm_context"])
                if isinstance(row["algorithm_context"], str)
                else row["algorithm_context"]
            )
            assert ctx_data.get("use_textual_feedback") is False

            # --- algorithm_state: trajectory present ---
            state_data = (
                json.loads(row["algorithm_state"])
                if isinstance(row["algorithm_state"], str)
                else row["algorithm_state"]
            )
            if step_num > 1:
                assert "trajectory" in state_data
                traj_elements = state_data["trajectory"]
                assert len(traj_elements) > 0

        # --- Validate run_summary.json ---
        summary_path = os.path.join(output_dir, "run_summary.json")
        with open(summary_path) as f:
            summary = json.load(f)
        assert "completed_at" in summary
        assert summary["total_steps"] == steps

    def test_e2e_prompt_evolves_across_steps(self, output_dir):
        """The optimizer produces new instructions at each step; verify they change."""
        from prompt_moo.algorithm import GPO
        from prompt_moo.config import temp_config
        from prompt_moo.observability import ObservabilityManager

        dataset = _MockDataset(data_dir=".")
        task_llm = _MockLLMPool(role="task")
        optimizer_llm = _MockLLMPool(role="optimizer")

        with temp_config(substep_delay=0, verbosity=0):
            algo = GPO(
                task_llm=task_llm,
                optimizer_llm=optimizer_llm,
                tasks=TASKS,
                steps=2,
                batch_size=2,
                loss_batch_size=2,
                gradient_batch_size=2,
                eval_every=99,
                name="evolve",
                task_losses=TASK_LOSSES,
                verbosity=0,
            )
            results = algo.train(
                dataset=dataset,
                initial_prompt=_make_prompt_template(),
                output_dir=output_dir,
            )

        run_logs = ObservabilityManager.read_run_logs(output_dir)
        step1_pu = json.loads(run_logs.iloc[0]["prompt_update"])
        step2_pu = json.loads(run_logs.iloc[1]["prompt_update"])

        assert (
            step1_pu["new_instruction"]["fluency"]
            == "Improved fluency instruction from optimizer."
        )
        assert (
            step2_pu["old_instruction"]["fluency"]
            == "Improved fluency instruction from optimizer."
        )


# -----------------------------------------------------------------------
# 7. Cosine decay step-size schedule edge cases
# -----------------------------------------------------------------------


class TestGPOCosineDecay:
    """Verify the cosine-decay edit distance schedule matches the paper."""

    def _calc(self, *, step: int, total_steps: int, warmup_steps: int = 0) -> int:
        from prompt_moo.algorithm.gpo import GPOOptimizer

        return GPOOptimizer()._calculate_step_size(
            step=step,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            initial_step_size=25,
            final_step_size=5,
        )

    def test_step_0_gets_initial_step_size(self):
        assert self._calc(step=0, total_steps=100) == 25

    def test_last_step_gets_final_step_size(self):
        result = self._calc(step=99, total_steps=100)
        assert result == 5

    def test_midpoint_is_between_initial_and_final(self):
        result = self._calc(step=50, total_steps=100)
        assert 5 < result < 25

    def test_monotonically_decreasing_without_warmup(self):
        values = [self._calc(step=s, total_steps=20) for s in range(20)]
        for i in range(1, len(values)):
            assert values[i] <= values[i - 1], (
                f"Step {i}: {values[i]} > step {i - 1}: {values[i - 1]}"
            )

    def test_warmup_ramps_up_then_decays(self):
        values = [self._calc(step=s, total_steps=20, warmup_steps=5) for s in range(20)]
        assert values[0] < values[4], "Warmup should ramp up"
        assert values[4] >= values[19], "Post-warmup should decay"

    def test_warmup_0_means_no_warmup(self):
        step0 = self._calc(step=0, total_steps=10, warmup_steps=0)
        assert step0 == 25

    def test_minimum_is_1_word(self):
        from prompt_moo.algorithm.gpo import GPOOptimizer

        result = GPOOptimizer()._calculate_step_size(
            step=999,
            warmup_steps=0,
            total_steps=1000,
            initial_step_size=2,
            final_step_size=0,
        )
        assert result >= 1

    def test_single_step_total(self):
        result = self._calc(step=0, total_steps=1)
        assert result == 25


# -----------------------------------------------------------------------
# 8. GPO forces unused LLMs to None
# -----------------------------------------------------------------------


class TestGPOForcesUnusedLLMsToNone:
    """When use_textual_feedback=False, loss_llm and gradient_llm must be None."""

    def test_loss_llm_forced_to_none(self):
        from prompt_moo.algorithm import GPO

        gpo = GPO(
            task_llm="fake_task_llm",
            optimizer_llm="fake_opt_llm",
            loss_llm="should_be_removed",
            gradient_llm="should_be_removed",
            tasks=TASKS,
            steps=1,
            batch_size=2,
            eval_every=1,
            name="t",
            task_losses=TASK_LOSSES,
        )
        assert gpo.loss_llm is None
        assert gpo.gradient_llm is None

    def test_textual_mode_keeps_llms(self):
        from prompt_moo.algorithm import GPO

        gpo = GPO(
            task_llm="fake_task_llm",
            optimizer_llm="fake_opt_llm",
            loss_llm="kept",
            gradient_llm="kept",
            tasks=TASKS,
            steps=1,
            batch_size=2,
            eval_every=1,
            name="t",
            task_losses=TASK_LOSSES,
            use_textual_feedback=True,
        )
        assert gpo.loss_llm == "kept"
        assert gpo.gradient_llm == "kept"


# -----------------------------------------------------------------------
# 9. GPO post_initialize validation
# -----------------------------------------------------------------------


class TestGPOPostInitializeValidation:
    """post_initialize rejects mismatched batch sizes."""

    def test_mismatched_loss_batch_size_raises(self):
        from prompt_moo.algorithm import GPO

        with pytest.raises(ValueError, match="loss_batch_size == batch_size"):
            GPO(
                task_llm=None,
                optimizer_llm=None,
                tasks=TASKS,
                steps=1,
                batch_size=4,
                loss_batch_size=2,
                gradient_batch_size=4,
                eval_every=1,
                name="t",
                task_losses=TASK_LOSSES,
            )

    def test_mismatched_gradient_batch_size_raises(self):
        from prompt_moo.algorithm import GPO

        with pytest.raises(ValueError, match="gradient_batch_size == batch_size"):
            GPO(
                task_llm=None,
                optimizer_llm=None,
                tasks=TASKS,
                steps=1,
                batch_size=4,
                loss_batch_size=4,
                gradient_batch_size=2,
                eval_every=1,
                name="t",
                task_losses=TASK_LOSSES,
            )

    def test_auto_filled_batch_sizes_pass(self):
        from prompt_moo.algorithm import GPO

        gpo = GPO(
            task_llm=None,
            optimizer_llm=None,
            tasks=TASKS,
            steps=1,
            batch_size=8,
            eval_every=1,
            name="t",
            task_losses=TASK_LOSSES,
        )
        assert gpo.loss_batch_size == 8
        assert gpo.gradient_batch_size == 8


# -----------------------------------------------------------------------
# 10. TaskLevelLossComputer requires loss_functions (GPO context)
# -----------------------------------------------------------------------


class TestTaskLevelLossComputerValidation:
    """TaskLevelLossComputer raises on missing loss_functions."""

    def test_missing_loss_functions_raises(self):
        from prompt_moo.loss_computer import TaskLevelLossComputer

        with pytest.raises(ValueError, match="requires 'loss_functions'"):
            TaskLevelLossComputer().compute(
                predictions=_make_predictions(),
                ground_truths=_make_samples(),
                tasks=TASKS,
                prompt_template=_make_prompt_template(),
                llm_pool=None,
                loss_batch_size=4,
                loss_functions=None,
            )

    def test_raises_for_tasks_not_in_loss_functions(self):
        from prompt_moo.loss_computer import TaskLevelLossComputer

        with pytest.raises(ValueError, match="not found in loss_functions"):
            TaskLevelLossComputer().compute(
                predictions=_make_predictions(),
                ground_truths=_make_samples(),
                tasks=TASKS,
                prompt_template=_make_prompt_template(),
                llm_pool=None,
                loss_batch_size=4,
                loss_functions={
                    "fluency": {"metric": "accuracy", "use_textual": False}
                },
            )


# -----------------------------------------------------------------------
# 11. GPOOptimizer meta-prompt: missing required fields
# -----------------------------------------------------------------------


class TestGPOOptimizerValidation:
    """GPOOptimizer raises on missing required fields."""

    def test_missing_batch_raises(self):
        from prompt_moo.algorithm.gpo import GPOOptimizer

        with pytest.raises(ValueError, match="Missing required 'batch'"):
            GPOOptimizer().create_meta_prompt(
                gradients={},
                current_prompt=_make_prompt_template(),
                tasks=TASKS,
                batch=None,
                trajectory=PromptTrajectory(
                    k=7, order="worst_to_best", metric_priority="accuracy"
                ),
                total_steps=100,
                trajectory_strategy="importance",
                use_textual_feedback=False,
                warmup_steps=0,
                initial_step_size=25,
                final_step_size=5,
                top_k_retrieve=7,
                loss_functions={
                    "fluency": {"metric": "accuracy", "use_textual": False},
                    "consistency": {"metric": "accuracy", "use_textual": False},
                },
            )

    def test_missing_trajectory_raises(self):
        from prompt_moo.algorithm.gpo import GPOOptimizer

        with pytest.raises(ValueError, match="Missing required 'trajectory'"):
            GPOOptimizer().create_meta_prompt(
                gradients={},
                current_prompt=_make_prompt_template(),
                tasks=TASKS,
                batch=Batch(step=0, samples=[]),
                trajectory=None,
                total_steps=100,
                trajectory_strategy="importance",
                use_textual_feedback=False,
                warmup_steps=0,
                initial_step_size=25,
                final_step_size=5,
                top_k_retrieve=7,
                loss_functions={
                    "fluency": {"metric": "accuracy", "use_textual": False},
                    "consistency": {"metric": "accuracy", "use_textual": False},
                },
            )

    def test_invalid_trajectory_strategy_raises(self):
        from prompt_moo.algorithm.gpo import GPOOptimizer

        with pytest.raises(ValueError, match="Unknown trajectory_strategy"):
            GPOOptimizer().create_meta_prompt(
                gradients={},
                current_prompt=_make_prompt_template(),
                tasks=TASKS,
                batch=Batch(step=0, samples=[]),
                trajectory=PromptTrajectory(
                    k=7, order="worst_to_best", metric_priority="accuracy"
                ),
                trajectory_strategy="recency",
                warmup_steps=0,
                total_steps=10,
                initial_step_size=25,
                final_step_size=5,
                top_k_retrieve=7,
                use_textual_feedback=False,
                loss_functions={
                    "fluency": {"metric": "accuracy", "use_textual": False},
                    "consistency": {"metric": "accuracy", "use_textual": False},
                },
            )


# -----------------------------------------------------------------------
# 12. GPOTrajectoryElement: single-task and empty scores
# -----------------------------------------------------------------------


class TestGPOTrajectoryElementEdgeCases:
    """Edge cases for the GPOTrajectoryElement display format."""

    def test_single_task_str(self):
        elem = GPOTrajectoryElement(
            instructions={"coherence": "Rate coherence."},
            numeric_scores={
                "coherence": [
                    NumericFeedback(
                        task_name="coherence",
                        metric=Accuracy(value=0.33),
                        aggregated_from_samples=[],
                    ),
                ]
            },
        )
        s = str(elem)
        assert "Prompt:" in s
        assert "- coherence: Rate coherence." in s
        assert "Scores: coherence=33.0" in s

    def test_single_string_instruction(self):
        elem = GPOTrajectoryElement(
            instructions="Rate everything from 1 to 5.",
            numeric_scores={
                "task": [
                    NumericFeedback(
                        task_name="task",
                        metric=Accuracy(value=0.5),
                        aggregated_from_samples=[],
                    ),
                ]
            },
        )
        s = str(elem)
        assert "Prompt: Rate everything from 1 to 5." in s
        assert "Score: 50.0" in s

    def test_no_scores_for_task(self):
        elem = GPOTrajectoryElement(
            instructions={"fluency": "Rate fluency."},
            numeric_scores={},
        )
        s = str(elem)
        assert "Prompt:" in s
        assert "- fluency: Rate fluency." in s
        assert "Score" not in s
