"""
Unit tests for TextGrad multi-task strategy validation.

Tests:
1. All 4 valid strategy configurations construct without error
2. All invalid configurations raise ValueError with clear messages
3. Batch size constraints are enforced (loss_batch_size=1, gradient_batch_size=1)

These are fast unit tests (no LLM calls, <30s).  For end-to-end tests that
verify the actual LLM prompts produced by each mode with real API calls, see
``tests/test_e2e_textgrad_modes.py``.
"""

import os
import sys
from typing import Any, Dict, List

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from prompt_moo.algorithm.textgrad import TextGrad
from prompt_moo.data_structures import Task


def _make_tasks() -> List[Task]:
    return [
        Task(
            task_name="fluency",
            task_description="Rate fluency",
            task_instruction="Rate from 1 to 5.",
            gt_col="fluency",
        ),
        Task(
            task_name="coherence",
            task_description="Rate coherence",
            task_instruction="Rate from 1 to 5.",
            gt_col="coherence",
        ),
    ]


class _FakeLLM:
    """Minimal stub satisfying the LLMPool protocol."""

    def call_llm_batch(self, *, prompts, verbosity=1, validator=None):
        raise NotImplementedError("Stub — not for actual LLM calls")

    def stop(self) -> None:
        pass


def _base_params() -> Dict[str, Any]:
    """Common params shared by all TextGrad construction tests."""
    fake = _FakeLLM()
    return {
        "tasks": _make_tasks(),
        "steps": 1,
        "batch_size": 2,
        "eval_every": 99999,
        "name": "test",
        "task_llm": fake,
        "gradient_llm": fake,
        "optimizer_llm": fake,
        "loss_llm": fake,
        "validation_metric": "accuracy",
    }


class TestTextGradStrategyValidation:
    """Validate that only the 4 legal strategy combinations are accepted."""

    # ------------------------------------------------------------------
    # Valid configurations: must construct without error
    # ------------------------------------------------------------------
    @pytest.mark.parametrize(
        "loss_s,grad_s,opt_s",
        [
            ("separate_tasks", "separate_tasks", "separate_tasks"),
            ("separate_tasks", "separate_tasks", "combine_all_tasks"),
            ("separate_tasks", "combine_all_tasks", "combine_all_tasks"),
            ("combine_all_tasks", "combine_all_tasks", "combine_all_tasks"),
        ],
        ids=["SSS", "SSC", "SCC", "CCC"],
    )
    def test_valid_strategy_constructs(self, loss_s, grad_s, opt_s):
        """Valid strategy combos must construct without error."""
        algo = TextGrad(
            loss_task_strategy=loss_s,
            gradient_task_strategy=grad_s,
            optimizer_task_strategy=opt_s,
            **_base_params(),
        )
        assert algo.loss_task_strategy == loss_s
        assert algo.gradient_task_strategy == grad_s
        assert algo.optimizer_task_strategy == opt_s

    # ------------------------------------------------------------------
    # Invalid configurations: must raise ValueError
    # ------------------------------------------------------------------
    @pytest.mark.parametrize(
        "loss_s,grad_s,opt_s,expected_fragment",
        [
            (
                "combine_all_tasks",
                "separate_tasks",
                "separate_tasks",
                "loss_task_strategy='combine_all_tasks'",
            ),
            (
                "combine_all_tasks",
                "separate_tasks",
                "combine_all_tasks",
                "loss_task_strategy='combine_all_tasks'",
            ),
            (
                "separate_tasks",
                "combine_all_tasks",
                "separate_tasks",
                "gradient_task_strategy='combine_all_tasks'",
            ),
            (
                "combine_all_tasks",
                "combine_all_tasks",
                "separate_tasks",
                "gradient_task_strategy='combine_all_tasks'",
            ),
        ],
        ids=["C/S/S", "C/S/C", "S/C/S", "C/C/S"],
    )
    def test_invalid_strategy_raises(self, loss_s, grad_s, opt_s, expected_fragment):
        """Invalid combos must raise ValueError with descriptive message."""
        with pytest.raises(ValueError, match=expected_fragment):
            TextGrad(
                loss_task_strategy=loss_s,
                gradient_task_strategy=grad_s,
                optimizer_task_strategy=opt_s,
                **_base_params(),
            )

    # ------------------------------------------------------------------
    # Batch size constraints
    # ------------------------------------------------------------------
    def test_loss_batch_size_must_be_1(self):
        """TextGrad requires loss_batch_size=1."""
        with pytest.raises(ValueError, match="loss_batch_size=1"):
            TextGrad(
                loss_batch_size=3,
                gradient_batch_size=1,
                **_base_params(),
            )

    def test_gradient_batch_size_must_be_1(self):
        """TextGrad requires gradient_batch_size=1."""
        with pytest.raises(ValueError, match="gradient_batch_size=1"):
            TextGrad(
                loss_batch_size=1,
                gradient_batch_size=5,
                **_base_params(),
            )

    def test_batch_sizes_auto_default_to_1(self):
        """When omitted, loss_batch_size and gradient_batch_size default to 1."""
        algo = TextGrad(**_base_params())
        assert algo.loss_batch_size == 1
        assert algo.gradient_batch_size == 1
