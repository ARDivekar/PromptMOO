"""
Tests for base PromptAlgorithm functionality shared across all algorithms.

Covers:
- _candidate_ranking_score: direction-aware, multi-task averaging
- num_candidates defaults per algorithm (OPRO, GPO, TextGrad, base)

Extracted from tests/test_opro.py where these tests were mixed in with
OPRO-specific tests despite testing base-class and cross-algorithm behavior.
"""

import pytest

from typing import Any

from prompt_moo.algorithm import GPO, OPRO, TextGrad
from prompt_moo.prompt_algorithm import PromptAlgorithm
from prompt_moo.data_structures import NumericFeedback, Task
from prompt_moo.metrics import Metric


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
TASK_LOSSES = {"fluency": "accuracy", "consistency": "accuracy"}


def _fb(
    *, task: str, metric: str, value: float, **metric_kwargs: Any
) -> NumericFeedback:
    metric_cls = Metric.get_subclass(metric)
    return NumericFeedback(
        task_name=task,
        metric=metric_cls(value=value, **metric_kwargs),
        aggregated_from_samples=[],
    )


class TestCandidateRankingScore:
    """Unit tests for PromptAlgorithm._candidate_ranking_score."""

    def test_empty_scores_returns_neg_inf(self):
        assert PromptAlgorithm._candidate_ranking_score({}) == float("-inf")

    def test_single_maximize_metric(self):
        scores = {
            TASKS[0]: [_fb(task="fluency", metric="accuracy", value=0.80)],
        }
        result = PromptAlgorithm._candidate_ranking_score(scores)
        assert result == pytest.approx(0.80)

    def test_single_minimize_metric_negated(self):
        scores = {
            TASKS[0]: [_fb(task="fluency", metric="lce", value=2.0, num_classes=5)],
        }
        result = PromptAlgorithm._candidate_ranking_score(scores)
        assert result == pytest.approx(-2.0)

    def test_multi_task_averages(self):
        scores = {
            TASKS[0]: [_fb(task="fluency", metric="accuracy", value=0.60)],
            TASKS[1]: [_fb(task="consistency", metric="accuracy", value=1.00)],
        }
        result = PromptAlgorithm._candidate_ranking_score(scores)
        assert result == pytest.approx(0.80)

    def test_multi_batch_per_task(self):
        scores = {
            TASKS[0]: [
                _fb(task="fluency", metric="accuracy", value=0.50),
                _fb(task="fluency", metric="accuracy", value=0.70),
            ],
        }
        result = PromptAlgorithm._candidate_ranking_score(scores)
        assert result == pytest.approx(0.60)

    def test_higher_accuracy_candidate_wins(self):
        scores_good = {
            TASKS[0]: [_fb(task="fluency", metric="accuracy", value=0.90)],
            TASKS[1]: [_fb(task="consistency", metric="accuracy", value=0.80)],
        }
        scores_bad = {
            TASKS[0]: [_fb(task="fluency", metric="accuracy", value=0.40)],
            TASKS[1]: [_fb(task="consistency", metric="accuracy", value=0.30)],
        }
        assert PromptAlgorithm._candidate_ranking_score(
            scores_good
        ) > PromptAlgorithm._candidate_ranking_score(scores_bad)

    def test_lower_loss_candidate_wins(self):
        scores_good = {
            TASKS[0]: [_fb(task="fluency", metric="lce", value=0.5, num_classes=5)],
        }
        scores_bad = {
            TASKS[0]: [_fb(task="fluency", metric="lce", value=3.0, num_classes=5)],
        }
        assert PromptAlgorithm._candidate_ranking_score(
            scores_good
        ) > PromptAlgorithm._candidate_ranking_score(scores_bad)


class TestMultiCandidateDefaults:
    """Verify num_candidates defaults per algorithm."""

    def test_opro_default_8(self):
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
        assert opro.num_candidates == 8

    def test_gpo_default_8(self):
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
        assert gpo.num_candidates == 8

    def test_textgrad_default_1(self):
        tg = TextGrad(
            task_llm=None,
            optimizer_llm=None,
            loss_llm=None,
            gradient_llm=None,
            tasks=TASKS,
            steps=1,
            batch_size=2,
            loss_batch_size=1,
            gradient_batch_size=1,
            eval_every=1,
            name="t",
            validation_metric="accuracy",
        )
        assert tg.num_candidates == 1

    def test_base_default_1(self):
        assert PromptAlgorithm.model_fields["num_candidates"].default == 1
