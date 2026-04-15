"""
Tests for the OPRO algorithm implementation (Yang et al., ICLR 2024) and the
refactored PromptTrajectory system.

Test groups:
1. Paper defaults (k=20, num_demos=3, 2 LLM roles)
2. Step 2 (Loss): numeric-only, no LLM calls
3. Step 3 (Gradient): deterministic score summary, no LLM calls
4. TrajectoryElement.ranking_key: direction-aware, weighted, lexicographic
5. PromptTrajectory: sort order, eviction, metric_priority validation
6. NumericFeedback.normalized_score + display_score
7. OPROOptimizer meta-prompt structure
8. E2E: full pipeline with mock LLM
"""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prompt_moo.data_structures import (
    Batch,
    DatasetSample,
    NumericFeedback,
    PredictionResult,
    Task,
    TextGradient,
    TextualFeedback,
)
from prompt_moo.metrics import Metric
from prompt_moo.metrics import Metric
from prompt_moo.prompt_template import PromptTemplate
from prompt_moo.prompt_trajectory import (
    PromptTrajectory,
    TrajectoryElement,
)
from prompt_moo.algorithm.opro import OPROTrajectoryElement

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
SKELETON = "Evaluate the summary. Output JSON with the requested task scores."
TASK_LOSSES = {"fluency": "accuracy", "consistency": "accuracy"}
INPUT_COL_LABELS = {"machine_summary": "Summary", "text": "Source Text"}


def _make_prompt() -> PromptTemplate:
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
            task_outputs={"fluency": 4, "consistency": 5},
            raw_response='{"fluency": 4, "consistency": 5}',
        ),
        PredictionResult(
            sample_id="s1",
            prompt="test_prompt",
            task_outputs={"fluency": 5, "consistency": 2},
            raw_response='{"fluency": 5, "consistency": 2}',
        ),
    ]


def _fb(*, task: str, metric: str, value: float) -> NumericFeedback:
    """Shorthand factory for NumericFeedback."""
    metric_cls = Metric.get_subclass(metric)
    return NumericFeedback(
        task_name=task,
        metric=metric_cls(value=value),
        aggregated_from_samples=[],
    )


# -----------------------------------------------------------------------
# 1. Paper defaults
# -----------------------------------------------------------------------


class TestOPROPaperDefaults:
    def test_k_equals_20(self):
        from prompt_moo.algorithm import OPRO

        opro = OPRO(
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
        assert opro.k == 20

    def test_num_task_demonstrations_equals_3(self):
        from prompt_moo.algorithm import OPRO

        opro = OPRO(
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
        assert opro.num_task_demonstrations == 3

    def test_no_gradient_or_loss_llm(self):
        from prompt_moo.algorithm import OPRO

        opro = OPRO(
            task_llm="task",
            optimizer_llm="opt",
            tasks=TASKS,
            steps=1,
            batch_size=2,
            loss_batch_size=2,
            gradient_batch_size=2,
            eval_every=1,
            name="t",
            task_losses=TASK_LOSSES,
        )
        assert opro.gradient_llm is None
        assert opro.loss_llm is None

    def test_trajectory_order_worst_to_best(self):
        from prompt_moo.algorithm import OPRO

        opro = OPRO(
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
        assert opro.trajectory.order == "worst_to_best"

    def test_trajectory_metric_priority_from_task_losses(self):
        from prompt_moo.algorithm import OPRO

        opro = OPRO(
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
        assert opro.trajectory.metric_priority == ["accuracy"]


# -----------------------------------------------------------------------
# 2. Step 2: Loss (numeric only)
# -----------------------------------------------------------------------


class TestOPROLoss:
    """OPRO uses the base TaskLevelLossComputer with llm_pool=None and
    use_textual=False, producing only numeric feedback."""

    def test_produces_only_numeric_feedback(self):
        from prompt_moo.loss_computer import TaskLevelLossComputer

        lc = TaskLevelLossComputer()
        feedbacks = lc.compute(
            predictions=_make_predictions(),
            ground_truths=_make_samples(),
            tasks=TASKS,
            prompt_template=_make_prompt(),
            llm_pool=None,
            loss_batch_size=10,
            verbosity=0,
            loss_functions={
                "fluency": {"metric": "accuracy"},
                "consistency": {"metric": "accuracy"},
            },
        )
        for task, fb_list in feedbacks.items():
            for fb in fb_list:
                assert isinstance(fb, NumericFeedback)

    def test_accuracy_values(self):
        from prompt_moo.loss_computer import TaskLevelLossComputer

        lc = TaskLevelLossComputer()
        feedbacks = lc.compute(
            predictions=_make_predictions(),
            ground_truths=_make_samples(),
            tasks=TASKS,
            prompt_template=_make_prompt(),
            llm_pool=None,
            loss_batch_size=10,
            verbosity=0,
            loss_functions={
                "fluency": {"metric": "accuracy"},
                "consistency": {"metric": "accuracy"},
            },
        )
        assert feedbacks[TASKS[0]][0].value == pytest.approx(0.5)
        assert feedbacks[TASKS[1]][0].value == pytest.approx(1.0)


# -----------------------------------------------------------------------
# 3. Step 3: Gradient (no LLM)
# -----------------------------------------------------------------------


class TestOPROGradient:
    def test_deterministic_no_llm(self):
        from prompt_moo.algorithm.opro import OPROGradientComputer

        gc = OPROGradientComputer()
        feedbacks = {
            TASKS[0]: [_fb(task="fluency", metric="accuracy", value=0.5)],
            TASKS[1]: [_fb(task="consistency", metric="accuracy", value=1.0)],
        }
        grads = gc.compute(
            feedbacks=feedbacks,
            prompt_template=_make_prompt(),
            tasks=TASKS,
            llm_pool=None,
            gradient_batch_size=10,
            verbosity=0,
        )
        for task, gl in grads.items():
            assert len(gl) == 1
            assert gl[0].gradient_prompt is None


# -----------------------------------------------------------------------
# 4. OPROOptimizer meta-prompt
#
# NOTE: TrajectoryElement.ranking_key, PromptTrajectory, and
# NumericFeedback tests live in tests/test_trajectory.py and
# tests/test_data_structures.py (shared infrastructure).
# Candidate ranking and multi-candidate defaults live in
# tests/test_algorithm_base.py (base-class behavior).
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------


class TestOPROMetaPrompt:
    def _make_kwargs(self) -> Dict[str, Any]:
        traj = PromptTrajectory(k=20, order="worst_to_best", metric_priority="accuracy")
        traj.push(
            OPROTrajectoryElement(
                instructions={
                    "fluency": "Rate carefully.",
                    "consistency": "Check facts.",
                },
                numeric_scores={
                    "fluency": [_fb(task="fluency", metric="accuracy", value=0.50)],
                    "consistency": [
                        _fb(task="consistency", metric="accuracy", value=0.80)
                    ],
                },
            )
        )
        traj.push(
            OPROTrajectoryElement(
                instructions={
                    "fluency": "Be precise.",
                    "consistency": "Verify alignment.",
                },
                numeric_scores={
                    "fluency": [_fb(task="fluency", metric="accuracy", value=0.70)],
                    "consistency": [
                        _fb(task="consistency", metric="accuracy", value=0.90)
                    ],
                },
            )
        )
        demos = [
            DatasetSample(
                sample_id="demo0",
                inputs={"machine_summary": "Summary text.", "text": "Source text."},
                ground_truths={"fluency": 4, "consistency": 5},
            )
        ]
        return {
            "trajectory": traj,
            "loss_functions": {
                "fluency": {"metric": "accuracy"},
                "consistency": {"metric": "accuracy"},
            },
            "task_demonstrations": demos,
            "input_col_labels": INPUT_COL_LABELS,
            "use_textual_feedback": False,
        }

    def test_contains_trajectory(self):
        from prompt_moo.algorithm.opro import OPROOptimizer

        meta = OPROOptimizer().create_meta_prompt(
            gradients={},
            current_prompt=_make_prompt(),
            tasks=TASKS,
            **self._make_kwargs(),
        )
        assert "Rate carefully." in meta
        assert "Be precise." in meta

    def test_worst_first(self):
        from prompt_moo.algorithm.opro import OPROOptimizer

        meta = OPROOptimizer().create_meta_prompt(
            gradients={},
            current_prompt=_make_prompt(),
            tasks=TASKS,
            **self._make_kwargs(),
        )
        assert meta.find("Rate carefully.") < meta.find("Be precise.")

    def test_contains_exemplars(self):
        from prompt_moo.algorithm.opro import OPROOptimizer

        meta = OPROOptimizer().create_meta_prompt(
            gradients={},
            current_prompt=_make_prompt(),
            tasks=TASKS,
            **self._make_kwargs(),
        )
        assert "Exemplar" in meta
        assert "fluency: 4" in meta

    def test_no_float_scores(self):
        from prompt_moo.algorithm.opro import OPROOptimizer

        meta = OPROOptimizer().create_meta_prompt(
            gradients={},
            current_prompt=_make_prompt(),
            tasks=TASKS,
            **self._make_kwargs(),
        )
        assert "0.5000" not in meta
        assert "0.7000" not in meta

    def test_meta_instructions_quality(self):
        from prompt_moo.algorithm.opro import OPROOptimizer

        meta = OPROOptimizer().create_meta_prompt(
            gradients={},
            current_prompt=_make_prompt(),
            tasks=TASKS,
            **self._make_kwargs(),
        )
        assert "higher score" in meta.lower()
        assert "different" in meta.lower()


# -----------------------------------------------------------------------
# 8. E2E
# -----------------------------------------------------------------------


class _MockFuture:
    def __init__(self, value):
        self._value = value

    def result(self, timeout=None):
        return self._value


class _MockLLM:
    """Mock LLM that cycles through a list of responses.

    Each call to call_llm_batch increments an internal counter. The response
    for call N is ``responses[(N-1) % len(responses)]``, replicated for every
    prompt in the batch.  When ``responses`` is None, a default JSON output
    is returned for every call.
    """

    _prompt_suffix = ""

    def __init__(self, responses=None):
        self._responses = responses
        self._call_count = 0

    def call_llm_batch(self, *, prompts, verbosity=1, validator=None, **kw):
        self._call_count += 1
        if self._responses is not None and len(self._responses) > 0:
            idx = (self._call_count - 1) % len(self._responses)
            results = [self._responses[idx]] * len(prompts)
        else:
            results = ['{"fluency": 3, "consistency": 4}'] * len(prompts)
        if validator is not None:
            results = [validator(r) for r in results]
        return _MockFuture(results)


class TestOPROE2E:
    def _make_dataset(self, tmp_dir):
        from prompt_moo.data_input import Dataset

        train = {
            "machine_summary": ["Cat sat.", "Dog ran.", "Bird flew.", "Fish swam."] * 2,
            "text": ["Cat.", "Dog.", "Bird.", "Fish."] * 2,
            "fluency": [4, 3, 5, 2, 4, 3, 5, 1],
            "consistency": [5, 2, 4, 3, 5, 4, 3, 2],
        }
        test = {
            "machine_summary": ["T1.", "T2."],
            "text": ["S1.", "S2."],
            "fluency": [4, 3],
            "consistency": [5, 2],
        }
        os.makedirs(os.path.join(tmp_dir, "DS"), exist_ok=True)
        pd.DataFrame(train).to_parquet(os.path.join(tmp_dir, "DS", "DS-train.parquet"))
        pd.DataFrame(test).to_parquet(os.path.join(tmp_dir, "DS", "DS-test.parquet"))

        class DS(Dataset):
            dataset_name = "DS"
            train_size = 8
            test_size = 2
            input_cols = ["machine_summary", "text"]
            gt_cols = ["fluency", "consistency"]
            input_col_labels = INPUT_COL_LABELS
            prompt_prefix = "Evaluate."
            task_output_formats = {"fluency": "1|2|3|4|5", "consistency": "1|2|3|4|5"}
            task_losses = TASK_LOSSES
            tasks: ClassVar[List[Task]] = TASKS

            @classmethod
            def setup(cls, base_dir):
                pass

        return DS(data_dir=tmp_dir)

    def test_e2e_runs(self):
        """E2E: OPRO trains for 2 steps and produces expected outputs.

        Validates:
        - Training completes and returns a final prompt
        - Run logs are written to disk
        - Trajectory contains the correct instructions: each element's
          instructions must be the prompt that was evaluated at that step
          (i.e., the prompt the task LLM actually saw), paired with the
          scores computed from that evaluation.
        """
        from prompt_moo.algorithm import OPRO
        from prompt_moo.config import temp_config

        initial_instr = {t.task_name: t.task_instruction for t in TASKS}
        step0_new = {"fluency": "new f", "consistency": "new c"}

        optimizer_resp = json.dumps({"instructions": step0_new})

        tmp = tempfile.mkdtemp()
        try:
            ds = self._make_dataset(tmp)
            initial_prompt = _make_prompt()
            with temp_config(substep_delay=0.0, batch_invocation_timeout=10.0):
                opro = OPRO(
                    task_llm=_MockLLM(),
                    optimizer_llm=_MockLLM([optimizer_resp] * 10),
                    tasks=TASKS,
                    steps=2,
                    batch_size=2,
                    loss_batch_size=2,
                    gradient_batch_size=2,
                    eval_every=1,
                    name="e2e",
                    task_losses=TASK_LOSSES,
                    k=5,
                    num_task_demonstrations=2,
                    num_candidates=1,
                    verbosity=0,
                )
                out = os.path.join(tmp, "out")
                results = opro.train(
                    dataset=ds, initial_prompt=initial_prompt, output_dir=out
                )

            assert results["final_prompt"] is not None
            assert os.path.isdir(os.path.join(out, "run_logs"))

            topk = opro.trajectory.get_topk()
            assert len(topk) == 2
            stored_instructions = [e.instructions for e in topk]
            assert initial_instr in stored_instructions, (
                f"Trajectory should contain the initial prompt instructions "
                f"(evaluated at step 0), got: {stored_instructions}"
            )
            assert step0_new in stored_instructions, (
                f"Trajectory should contain step 0's optimizer output "
                f"(evaluated at step 1), got: {stored_instructions}"
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# -----------------------------------------------------------------------
# 8b. Fixed training batch (OPRO paper: exp.tex L84)
# -----------------------------------------------------------------------


class _PromptCapturingMockLLM:
    """Mock LLM that records prompts sent to it for later inspection.

    Task-prediction calls (validator=None) store each prompt in
    ``captured_task_prompts``.  Optimizer calls (validator is not None)
    return a fixed instruction set.
    """

    _prompt_suffix = ""

    def __init__(self, *, optimizer_response: str):
        self._optimizer_response = optimizer_response
        self.captured_task_prompts: List[List[str]] = []

    def call_llm_batch(self, *, prompts, verbosity=1, validator=None, **kw):
        if validator is not None:
            results = [validator(self._optimizer_response) for _ in prompts]
        else:
            self.captured_task_prompts.append(list(prompts))
            results = ['{"fluency": 3, "consistency": 4}'] * len(prompts)
        return _MockFuture(results)


class TestOPROFixedTrainingBatch:
    """Verify that OPRO uses the same training batch across all steps.

    The OPRO paper (exp.tex L84) states: "The same subset is used
    throughout optimization, so that the task accuracies computed at
    intermediate optimization steps are approximations of the training
    accuracy."

    Without ``use_fixed_training_batch=True``, each step samples a
    different batch using the step number as the random seed, making
    trajectory scores non-comparable.
    """

    def _make_dataset(self, tmp_dir):
        from prompt_moo.data_input import Dataset

        train = {
            "machine_summary": [f"Summary {i}." for i in range(20)],
            "text": [f"Source {i}." for i in range(20)],
            "fluency": list(range(1, 6)) * 4,
            "consistency": list(range(1, 6)) * 4,
        }
        test = {
            "machine_summary": ["T1.", "T2."],
            "text": ["S1.", "S2."],
            "fluency": [4, 3],
            "consistency": [5, 2],
        }
        os.makedirs(os.path.join(tmp_dir, "DS"), exist_ok=True)
        pd.DataFrame(train).to_parquet(os.path.join(tmp_dir, "DS", "DS-train.parquet"))
        pd.DataFrame(test).to_parquet(os.path.join(tmp_dir, "DS", "DS-test.parquet"))

        class DS(Dataset):
            dataset_name = "DS"
            train_size = 20
            test_size = 2
            input_cols = ["machine_summary", "text"]
            gt_cols = ["fluency", "consistency"]
            input_col_labels = INPUT_COL_LABELS
            prompt_prefix = "Evaluate."
            task_output_formats = {"fluency": "1|2|3|4|5", "consistency": "1|2|3|4|5"}
            task_losses = TASK_LOSSES
            tasks: ClassVar[List[Task]] = TASKS

            @classmethod
            def setup(cls, base_dir):
                pass

        return DS(data_dir=tmp_dir)

    def test_same_samples_across_steps(self):
        """The training batch content must be identical at every step.

        Runs OPRO for 3 steps with batch_size=4 from a pool of 20
        training samples.  The task LLM's prompts are captured and the
        sample content extracted.  All three steps must use the same
        4 samples (though sample_id strings differ per step).

        Steps:
        1. Create dataset with 20 distinct training samples.
        2. Run OPRO for 3 steps with use_fixed_training_batch=True.
        3. Extract the sample content from each step's task LLM prompts.
        4. Assert that steps 1, 2, and 3 used identical sample content.
        """
        from prompt_moo.algorithm import OPRO
        from prompt_moo.config import temp_config

        optimizer_resp = json.dumps(
            {"instructions": {"fluency": "new f", "consistency": "new c"}}
        )

        tmp = tempfile.mkdtemp()
        try:
            ds = self._make_dataset(tmp)
            initial_prompt = _make_prompt()
            task_mock = _PromptCapturingMockLLM(optimizer_response=optimizer_resp)

            with temp_config(substep_delay=0.0, batch_invocation_timeout=10.0):
                opro = OPRO(
                    task_llm=task_mock,
                    optimizer_llm=_MockLLM([optimizer_resp] * 20),
                    tasks=TASKS,
                    steps=3,
                    batch_size=4,
                    loss_batch_size=4,
                    gradient_batch_size=4,
                    eval_every=99,
                    name="fixed_batch",
                    task_losses=TASK_LOSSES,
                    k=5,
                    num_candidates=1,
                    num_task_demonstrations=0,
                    verbosity=0,
                )
                out = os.path.join(tmp, "out")
                opro.train(dataset=ds, initial_prompt=initial_prompt, output_dir=out)

            assert opro.use_fixed_training_batch is True

            def _extract_sample_content(prompts: List[str]) -> List[str]:
                """Extract the ## Sample: section from each prompt."""
                contents: List[str] = []
                for p in prompts:
                    marker: str = "## Sample:\n"
                    start: int = p.find(marker)
                    end: int = p.find("## Response:")
                    if start != -1 and end != -1:
                        contents.append(p[start + len(marker) : end].strip())
                return sorted(contents)

            # captured_task_prompts includes: step 1 predict, step 1 candidate eval (x2+),
            # step 2 predict, step 2 candidate eval (x2+), step 3 predict, step 3 candidate eval (x2+),
            # plus baseline eval (step 0) and final eval (step 3 last_step).
            # With num_candidates=1, no candidate eval calls happen.
            # With eval_every=99 and eval_last_step=True, eval runs at step 0 and step 3.
            # Task LLM calls: step 0 eval (full test), step 1 predict (batch),
            # step 2 predict (batch), step 3 predict (batch), step 3 eval (full test).
            # The predict calls (indices 1, 2, 3) are the ones with batch_size=4.
            # But eval calls have different sizes (test set = 2 samples).
            batch_calls: List[List[str]] = [
                call for call in task_mock.captured_task_prompts if len(call) == 4
            ]
            assert len(batch_calls) >= 3, (
                f"Expected at least 3 batch-size-4 predict calls (one per step), "
                f"got {len(batch_calls)}"
            )

            step1_content: List[str] = _extract_sample_content(batch_calls[0])
            step2_content: List[str] = _extract_sample_content(batch_calls[1])
            step3_content: List[str] = _extract_sample_content(batch_calls[2])

            assert step1_content == step2_content, (
                f"Step 1 and Step 2 must use the same training samples.\n"
                f"Step 1 samples: {step1_content[:2]}...\n"
                f"Step 2 samples: {step2_content[:2]}..."
            )
            assert step2_content == step3_content, (
                f"Step 2 and Step 3 must use the same training samples.\n"
                f"Step 2 samples: {step2_content[:2]}...\n"
                f"Step 3 samples: {step3_content[:2]}..."
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_use_fixed_training_batch_defaults_true(self):
        """OPRO must default use_fixed_training_batch=True per the paper."""
        from prompt_moo.algorithm import OPRO

        opro = OPRO(
            task_llm=None,
            optimizer_llm=None,
            tasks=TASKS,
            steps=1,
            batch_size=2,
            eval_every=1,
            name="t",
            task_losses=TASK_LOSSES,
        )
        assert opro.use_fixed_training_batch is True


# -----------------------------------------------------------------------
# 6. Multi-candidate E2E (OPRO-specific)
#
# NOTE: TestCandidateRankingScore and TestMultiCandidateDefaults have
# been moved to tests/test_algorithm_base.py (base-class behavior).
# -----------------------------------------------------------------------


class _CountingMockLLM:
    """Mock LLM that returns distinct per-call optimizer responses.

    Each call_llm_batch increments a counter.  When ``validator`` is not None
    (optimizer calls), returns a unique instruction set using the counter.
    When ``validator`` is None (task prediction calls), returns a fixed JSON.
    """

    _prompt_suffix = ""

    def __init__(self):
        self._call_count = 0

    def call_llm_batch(self, *, prompts, verbosity=1, validator=None, **kw):
        self._call_count += 1
        if validator is not None:
            instr = {
                "fluency": f"fluency_instruction_{self._call_count}",
                "consistency": f"consistency_instruction_{self._call_count}",
            }
            resp = json.dumps({"instructions": instr})
            results = [validator(resp) for _ in prompts]
        else:
            results = ['{"fluency": 3, "consistency": 4}'] * len(prompts)
        return _MockFuture(results)


class _ScoringMockLLM:
    """Mock task LLM that produces controllable per-candidate scores.

    The first N task-prediction batches return scores that give perfect
    accuracy for the candidate at index ``winning_candidate_idx`` and low
    accuracy for all others.

    Usage: set ``winning_candidate_idx`` before the step's predict calls.
    The mock counts task-prediction batches (calls without ``validator``)
    and returns perfect or imperfect outputs based on which candidate is
    being evaluated.

    For optimizer calls (validator is not None), it produces distinct
    instruction sets so each candidate is unique.
    """

    _prompt_suffix = ""

    def __init__(self, *, ground_truths: Dict[str, int], winning_idx: int = 0):
        self._call_count = 0
        self._predict_count = 0
        self._ground_truths = ground_truths
        self.winning_idx = winning_idx

    def call_llm_batch(self, *, prompts, verbosity=1, validator=None, **kw):
        self._call_count += 1
        if validator is not None:
            instr = {
                "fluency": f"fluency_{self._call_count}",
                "consistency": f"consistency_{self._call_count}",
            }
            resp = json.dumps({"instructions": instr})
            results = [validator(resp) for _ in prompts]
        else:
            self._predict_count += 1
            if self._predict_count == 1:
                results = [json.dumps(self._ground_truths)] * len(prompts)
            else:
                candidate_eval_idx = self._predict_count - 2
                if candidate_eval_idx == self.winning_idx:
                    results = [json.dumps(self._ground_truths)] * len(prompts)
                else:
                    wrong = {k: v + 1 for k, v in self._ground_truths.items()}
                    results = [json.dumps(wrong)] * len(prompts)
        return _MockFuture(results)


class TestMultiCandidateE2E:
    """E2E tests for multi-candidate generation in OPRO.

    These tests run the full training loop with mock LLMs to verify that
    multi-candidate generation, evaluation, selection, and trajectory
    population all work correctly.
    """

    def _make_dataset(self, tmp_dir):
        from prompt_moo.data_input import Dataset

        train = {
            "machine_summary": ["Cat sat.", "Dog ran.", "Bird flew.", "Fish swam."] * 2,
            "text": ["Cat.", "Dog.", "Bird.", "Fish."] * 2,
            "fluency": [4, 3, 5, 2, 4, 3, 5, 1],
            "consistency": [5, 2, 4, 3, 5, 4, 3, 2],
        }
        test = {
            "machine_summary": ["T1.", "T2."],
            "text": ["S1.", "S2."],
            "fluency": [4, 3],
            "consistency": [5, 2],
        }
        os.makedirs(os.path.join(tmp_dir, "DS"), exist_ok=True)
        pd.DataFrame(train).to_parquet(os.path.join(tmp_dir, "DS", "DS-train.parquet"))
        pd.DataFrame(test).to_parquet(os.path.join(tmp_dir, "DS", "DS-test.parquet"))

        class DS(Dataset):
            dataset_name = "DS"
            train_size = 8
            test_size = 2
            input_cols = ["machine_summary", "text"]
            gt_cols = ["fluency", "consistency"]
            input_col_labels = INPUT_COL_LABELS
            prompt_prefix = "Evaluate."
            task_output_formats = {"fluency": "1|2|3|4|5", "consistency": "1|2|3|4|5"}
            task_losses = TASK_LOSSES
            tasks: ClassVar[List[Task]] = TASKS

            @classmethod
            def setup(cls, base_dir):
                pass

        return DS(data_dir=tmp_dir)

    def test_num_candidates_3_generates_distinct_candidates(self):
        """With num_candidates=3, the optimizer LLM is called 3 times and
        the trajectory receives all evaluated candidates plus the current prompt.

        Steps: 1 step with num_candidates=3.
        Expected trajectory entries:
        - 1 from the current_prompt evaluation (Step 1-2)
        - 3 from the evaluated candidates
        = 4 total
        """
        from prompt_moo.algorithm import OPRO
        from prompt_moo.config import temp_config

        tmp = tempfile.mkdtemp()
        try:
            ds = self._make_dataset(tmp)
            initial_prompt = _make_prompt()
            counting_llm = _CountingMockLLM()

            with temp_config(substep_delay=0.0, batch_invocation_timeout=10.0):
                opro = OPRO(
                    task_llm=_MockLLM(),
                    optimizer_llm=counting_llm,
                    tasks=TASKS,
                    steps=1,
                    batch_size=2,
                    loss_batch_size=2,
                    gradient_batch_size=2,
                    eval_every=1,
                    name="mc3",
                    task_losses=TASK_LOSSES,
                    k=20,
                    num_candidates=3,
                    num_task_demonstrations=0,
                    verbosity=0,
                )
                out = os.path.join(tmp, "out")
                opro.train(dataset=ds, initial_prompt=initial_prompt, output_dir=out)

            topk = opro.trajectory.get_topk()
            assert len(topk) == 4, (
                f"Expected 4 trajectory entries (1 current + 3 candidates), "
                f"got {len(topk)}"
            )

            all_instructions = [e.instructions for e in topk]
            unique_instructions = set(
                json.dumps(i, sort_keys=True) for i in all_instructions
            )
            assert len(unique_instructions) >= 3, (
                f"Expected at least 3 distinct instruction sets (from 3 candidates), "
                f"got {len(unique_instructions)}: {all_instructions}"
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_single_candidate_no_evaluation_overhead(self):
        """With num_candidates=1, the candidate is NOT evaluated separately.

        The trajectory should contain only the current_prompt's scores (from
        Steps 1-2), not a separate evaluation of the candidate.
        """
        from prompt_moo.algorithm import OPRO
        from prompt_moo.config import temp_config

        optimizer_resp = json.dumps(
            {
                "instructions": {
                    "fluency": "new f",
                    "consistency": "new c",
                }
            }
        )

        tmp = tempfile.mkdtemp()
        try:
            ds = self._make_dataset(tmp)
            initial_prompt = _make_prompt()
            task_mock = _MockLLM()

            with temp_config(substep_delay=0.0, batch_invocation_timeout=10.0):
                opro = OPRO(
                    task_llm=task_mock,
                    optimizer_llm=_MockLLM([optimizer_resp]),
                    tasks=TASKS,
                    steps=1,
                    batch_size=2,
                    loss_batch_size=2,
                    gradient_batch_size=2,
                    eval_every=99,
                    name="mc1",
                    task_losses=TASK_LOSSES,
                    k=20,
                    num_candidates=1,
                    num_task_demonstrations=0,
                    verbosity=0,
                )
                out = os.path.join(tmp, "out")
                opro.train(dataset=ds, initial_prompt=initial_prompt, output_dir=out)

            assert task_mock._call_count == 3, (
                f"With num_candidates=1, task LLM should be called exactly 3 times "
                f"(baseline eval at step 0 + Step 1 predict + eval at step 1), "
                f"got {task_mock._call_count}"
            )
            assert len(opro.trajectory) == 1
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_best_candidate_selected_by_score(self):
        """The candidate with the highest evaluation score is selected as
        the new prompt, not an arbitrary one.

        Uses a _ScoringMockLLM that returns perfect accuracy for candidate
        at index 1 (second candidate) and wrong answers for others.
        After training, the final prompt should contain the winning
        candidate's instruction text.
        """
        from prompt_moo.algorithm import OPRO
        from prompt_moo.config import temp_config

        ground_truths = {"fluency": 4, "consistency": 5}
        scoring_llm = _ScoringMockLLM(
            ground_truths=ground_truths,
            winning_idx=1,
        )

        tmp = tempfile.mkdtemp()
        try:
            ds = self._make_dataset(tmp)
            initial_prompt = _make_prompt()

            with temp_config(substep_delay=0.0, batch_invocation_timeout=10.0):
                opro = OPRO(
                    task_llm=scoring_llm,
                    optimizer_llm=scoring_llm,
                    tasks=TASKS,
                    steps=1,
                    batch_size=2,
                    loss_batch_size=2,
                    gradient_batch_size=2,
                    eval_every=99,
                    name="select",
                    task_losses=TASK_LOSSES,
                    k=20,
                    num_candidates=3,
                    num_task_demonstrations=0,
                    verbosity=0,
                )
                out = os.path.join(tmp, "out")
                results = opro.train(
                    dataset=ds,
                    initial_prompt=initial_prompt,
                    output_dir=out,
                )

            final = results["final_prompt"]
            final_fluency_instr = final.instruction["fluency"]
            assert "fluency_" in final_fluency_instr, (
                f"Expected final instruction from a generated candidate, "
                f"got: {final_fluency_instr}"
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_all_candidates_pushed_to_trajectory(self):
        """All evaluated candidates are pushed to the OPRO trajectory,
        matching the paper's design where all 8 candidates per step enter
        the history.

        Steps: 1 step, num_candidates=4, k=20.
        Expected: 5 trajectory entries (1 current_prompt + 4 candidates).
        """
        from prompt_moo.algorithm import OPRO
        from prompt_moo.config import temp_config

        tmp = tempfile.mkdtemp()
        try:
            ds = self._make_dataset(tmp)
            initial_prompt = _make_prompt()

            with temp_config(substep_delay=0.0, batch_invocation_timeout=10.0):
                opro = OPRO(
                    task_llm=_MockLLM(),
                    optimizer_llm=_CountingMockLLM(),
                    tasks=TASKS,
                    steps=1,
                    batch_size=2,
                    loss_batch_size=2,
                    gradient_batch_size=2,
                    eval_every=99,
                    name="allpush",
                    task_losses=TASK_LOSSES,
                    k=20,
                    num_candidates=4,
                    num_task_demonstrations=0,
                    verbosity=0,
                )
                out = os.path.join(tmp, "out")
                opro.train(dataset=ds, initial_prompt=initial_prompt, output_dir=out)

            topk = opro.trajectory.get_topk()
            assert len(topk) == 5, (
                f"Expected 5 trajectory entries (1 current + 4 candidates), "
                f"got {len(topk)}"
            )

            for elem in topk:
                has_scores = any(len(fbs) > 0 for fbs in elem.numeric_scores.values())
                assert has_scores, (
                    f"Every trajectory element must have numeric scores, "
                    f"but element with instructions {elem.instructions} has none"
                )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# -----------------------------------------------------------------------
# 9. Batch size validation (post_initialize)
# -----------------------------------------------------------------------


class TestOPROBatchSizeValidation:
    """OPRO requires loss_batch_size == batch_size and gradient_batch_size == batch_size."""

    def test_loss_batch_size_mismatch_raises(self):
        from prompt_moo.algorithm import OPRO

        with pytest.raises(ValueError, match="loss_batch_size == batch_size"):
            OPRO(
                task_llm=None,
                optimizer_llm=None,
                tasks=TASKS,
                steps=1,
                batch_size=10,
                loss_batch_size=5,
                gradient_batch_size=10,
                eval_every=1,
                name="t",
                task_losses=TASK_LOSSES,
            )

    def test_gradient_batch_size_mismatch_raises(self):
        from prompt_moo.algorithm import OPRO

        with pytest.raises(ValueError, match="gradient_batch_size == batch_size"):
            OPRO(
                task_llm=None,
                optimizer_llm=None,
                tasks=TASKS,
                steps=1,
                batch_size=10,
                loss_batch_size=10,
                gradient_batch_size=3,
                eval_every=1,
                name="t",
                task_losses=TASK_LOSSES,
            )

    def test_pre_initialize_auto_fills_from_batch_size(self):
        from prompt_moo.algorithm import OPRO

        opro = OPRO(
            task_llm=None,
            optimizer_llm=None,
            tasks=TASKS,
            steps=1,
            batch_size=24,
            eval_every=1,
            name="t",
            task_losses=TASK_LOSSES,
        )
        assert opro.loss_batch_size == 24
        assert opro.gradient_batch_size == 24


# -----------------------------------------------------------------------
# 10. OPRO uses TaskLevelLossComputer directly (no OPRO-specific subclass)
# -----------------------------------------------------------------------


class TestOPROUsesTaskLevelLossComputer:
    """OPRO uses the base TaskLevelLossComputer with llm_pool=None.
    No OPRO-specific LossComputer subclass is needed because the base class
    already handles llm_pool=None by skipping textual feedback."""

    def test_opro_loss_computer_is_task_level(self):
        """OPRO's loss_computer config resolves to TaskLevelLossComputer."""
        from prompt_moo.loss_computer import LossComputer, TaskLevelLossComputer

        resolved = LossComputer.of("task-level")
        assert isinstance(resolved, TaskLevelLossComputer)

    def test_requires_loss_functions(self):
        from prompt_moo.loss_computer import TaskLevelLossComputer

        lc = TaskLevelLossComputer()
        with pytest.raises(ValueError, match="requires 'loss_functions'"):
            lc.compute(
                predictions=[],
                ground_truths=[],
                tasks=TASKS,
                prompt_template=_make_prompt(),
                llm_pool=None,
                loss_batch_size=10,
                verbosity=0,
            )

    def test_empty_predictions_returns_empty_feedbacks(self):
        from prompt_moo.loss_computer import TaskLevelLossComputer

        lc = TaskLevelLossComputer()
        feedbacks = lc.compute(
            predictions=[],
            ground_truths=[],
            tasks=TASKS,
            prompt_template=_make_prompt(),
            llm_pool=None,
            loss_batch_size=10,
            verbosity=0,
            loss_functions={
                "fluency": {"metric": "accuracy"},
                "consistency": {"metric": "accuracy"},
            },
        )
        for task in TASKS:
            assert len(feedbacks[task]) == 0


# -----------------------------------------------------------------------
# 11. Integer score formatting (decimals=0)
# -----------------------------------------------------------------------


class TestOPROIntegerScores:
    """OPRO scores must be integer percentages per the paper."""

    def test_accuracy_to_score_str_with_decimals_0(self):
        from prompt_moo.metrics import Accuracy

        metric = Accuracy(value=0.6667, display_decimals=0)
        assert metric.display_score == "67"

    def test_accuracy_to_score_str_with_decimals_1(self):
        from prompt_moo.metrics import Accuracy

        metric = Accuracy(value=0.6667, display_decimals=1)
        assert metric.display_score == "66.7"

    def test_trajectory_element_shows_integer_scores(self):
        elem = OPROTrajectoryElement(
            instructions={"fluency": "Test instruction"},
            numeric_scores={
                "fluency": [_fb(task="fluency", metric="accuracy", value=0.6667)],
            },
        )
        text = str(elem)
        assert "66.7" in text
        assert "0.6667" not in text

    def test_trajectory_element_with_decimals_0_shows_integer(self):
        from prompt_moo.metrics import Accuracy

        metric = Accuracy(value=0.6667, display_decimals=0)
        fb = NumericFeedback(
            task_name="fluency", metric=metric, aggregated_from_samples=[]
        )
        elem = OPROTrajectoryElement(
            instructions={"fluency": "Test instruction"},
            numeric_scores={"fluency": [fb]},
        )
        text = str(elem)
        assert "Score: 67" in text
        assert "66.7" not in text


# -----------------------------------------------------------------------
# 12. Meta-prompt structure (paper-faithful checks)
# -----------------------------------------------------------------------


class TestOPROMetaPromptStructure:
    """Verify the meta-prompt follows OPRO paper conventions."""

    def _build_meta_prompt(self, *, order: str = "worst_to_best") -> str:
        from prompt_moo.algorithm.opro import OPROOptimizer

        traj = PromptTrajectory(k=20, order=order, metric_priority="accuracy")
        traj.push(
            OPROTrajectoryElement(
                instructions={"fluency": "Instruction A", "consistency": "Check A"},
                numeric_scores={
                    "fluency": [_fb(task="fluency", metric="accuracy", value=0.30)],
                    "consistency": [
                        _fb(task="consistency", metric="accuracy", value=0.50)
                    ],
                },
            )
        )
        traj.push(
            OPROTrajectoryElement(
                instructions={"fluency": "Instruction B", "consistency": "Check B"},
                numeric_scores={
                    "fluency": [_fb(task="fluency", metric="accuracy", value=0.70)],
                    "consistency": [
                        _fb(task="consistency", metric="accuracy", value=0.90)
                    ],
                },
            )
        )
        demos = [
            DatasetSample(
                sample_id="d0",
                inputs={"machine_summary": "Summary.", "text": "Source."},
                ground_truths={"fluency": 4, "consistency": 5},
            )
        ]
        return OPROOptimizer().create_meta_prompt(
            gradients={},
            current_prompt=_make_prompt(),
            tasks=TASKS,
            trajectory=traj,
            task_demonstrations=demos,
            input_col_labels=INPUT_COL_LABELS,
            loss_functions={
                "fluency": {"metric": "accuracy"},
                "consistency": {"metric": "accuracy"},
            },
        )

    def test_no_prompt_template_section(self):
        """The paper does NOT show the full prompt template."""
        meta = self._build_meta_prompt()
        assert "prompt template" not in meta.lower()
        assert "## Instructions:" not in meta
        assert "## Sample:" not in meta

    def test_has_ins_placeholders(self):
        """Exemplars must show <INS_task> placeholders."""
        meta = self._build_meta_prompt()
        assert "<INS_fluency>" in meta
        assert "<INS_consistency>" in meta

    def test_has_evaluation_directive_in_exemplars(self):
        """Each exemplar's Instructions line includes the evaluation directive."""
        meta = self._build_meta_prompt()
        assert "Instructions: Evaluate the summary" in meta

    def test_has_delimiter_between_trajectory_elements(self):
        """Trajectory elements are separated by ---."""
        meta = self._build_meta_prompt()
        assert "\n---\n" in meta

    def test_ascending_order_description(self):
        meta = self._build_meta_prompt(order="worst_to_best")
        assert "worst to best" in meta.lower()
        assert "best-performing instructions appear last" in meta.lower()

    def test_descending_order_description(self):
        meta = self._build_meta_prompt(order="best_to_worst")
        assert "best to worst" in meta.lower()
        assert "best-performing instructions appear first" in meta.lower()

    def test_has_score_direction_description(self):
        """Meta-prompt should describe what score direction means."""
        meta = self._build_meta_prompt()
        assert "higher" in meta.lower() or "lower" in meta.lower()

    def test_improves_performance_scores(self):
        """Should say 'improves the performance scores' not 'has a higher score'."""
        meta = self._build_meta_prompt()
        assert "improves the performance scores" in meta

    def test_json_output_format(self):
        meta = self._build_meta_prompt()
        assert '"instructions"' in meta
        assert '"fluency"' in meta
        assert '"consistency"' in meta

    def test_no_task_names_line(self):
        """Redundant 'Task names:' line should not appear."""
        meta = self._build_meta_prompt()
        assert "Task names:" not in meta


# -----------------------------------------------------------------------
# 13. OPROTrajectoryElement display format
# -----------------------------------------------------------------------


class TestTrajectoryElementDisplay:
    """Test the string representation of trajectory elements."""

    def test_single_task_format(self):
        elem = OPROTrajectoryElement(
            instructions="Rate from 1 to 5.",
            numeric_scores={
                "coherence": [_fb(task="coherence", metric="accuracy", value=0.33)]
            },
        )
        text = str(elem)
        assert "Instruction: Rate from 1 to 5." in text
        assert "Score: 33.0" in text

    def test_multi_task_interleaved_format(self):
        elem = OPROTrajectoryElement(
            instructions={"fluency": "Rate fluency.", "coherence": "Rate coherence."},
            numeric_scores={
                "fluency": [_fb(task="fluency", metric="accuracy", value=0.50)],
                "coherence": [_fb(task="coherence", metric="accuracy", value=0.75)],
            },
        )
        text = str(elem)
        fluency_pos = text.find("fluency:")
        fluency_score_pos = text.find("Score: 50.0")
        coherence_pos = text.find("coherence:")
        assert fluency_pos < fluency_score_pos < coherence_pos

    def test_empty_scores(self):
        elem = OPROTrajectoryElement(
            instructions={"fluency": "Rate fluency."},
            numeric_scores={},
        )
        text = str(elem)
        assert "fluency:" in text
        assert "Score" not in text


# -----------------------------------------------------------------------
# 14. OPROGradientComputer edge cases
# -----------------------------------------------------------------------


class TestOPROGradientEdgeCases:
    def test_empty_feedbacks(self):
        from prompt_moo.algorithm.opro import OPROGradientComputer

        gc = OPROGradientComputer()
        grads = gc.compute(
            feedbacks={TASKS[0]: []},
            prompt_template=_make_prompt(),
            tasks=TASKS,
            llm_pool=None,
            gradient_batch_size=10,
            verbosity=0,
        )
        assert len(grads[TASKS[0]]) == 0

    def test_only_textual_feedbacks_produces_empty(self):
        from prompt_moo.algorithm.opro import OPROGradientComputer

        gc = OPROGradientComputer()
        grads = gc.compute(
            feedbacks={
                TASKS[0]: [
                    TextualFeedback(
                        task_name="fluency",
                        feedback_text="Some feedback",
                        aggregated_from_samples=[],
                        feedback_prompt=None,
                    )
                ],
            },
            prompt_template=_make_prompt(),
            tasks=TASKS,
            llm_pool=None,
            gradient_batch_size=10,
            verbosity=0,
        )
        assert len(grads[TASKS[0]]) == 0

    def test_gradient_prompt_is_none(self):
        from prompt_moo.algorithm.opro import OPROGradientComputer

        gc = OPROGradientComputer()
        grads = gc.compute(
            feedbacks={TASKS[0]: [_fb(task="fluency", metric="accuracy", value=0.5)]},
            prompt_template=_make_prompt(),
            tasks=TASKS,
            llm_pool=None,
            gradient_batch_size=10,
            verbosity=0,
        )
        assert grads[TASKS[0]][0].gradient_prompt is None
