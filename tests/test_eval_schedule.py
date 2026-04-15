"""Tests for should_evaluate_at_step eval schedule logic."""

import pytest
from pydantic import ValidationError

from prompt_moo.prompt_algorithm import should_evaluate_at_step


class TestShouldEvaluateAtStep:
    """Rigorous tests for the eval schedule function."""

    def test_step0_baseline_default(self):
        assert should_evaluate_at_step(step=0, total_steps=20, eval_every=5) is True

    def test_step0_baseline_disabled(self):
        assert (
            should_evaluate_at_step(
                step=0, total_steps=20, eval_every=5, eval_initial_prompt=False
            )
            is False
        )

    def test_step1_first_step_default(self):
        assert should_evaluate_at_step(step=1, total_steps=20, eval_every=5) is True

    def test_step1_first_step_disabled(self):
        assert (
            should_evaluate_at_step(
                step=1, total_steps=20, eval_every=5, eval_first_step=False
            )
            is False
        )

    def test_last_step_default(self):
        assert should_evaluate_at_step(step=20, total_steps=20, eval_every=5) is True

    def test_last_step_disabled(self):
        assert (
            should_evaluate_at_step(
                step=20, total_steps=20, eval_every=5, eval_last_step=False
            )
            is True
        )

    def test_last_step_disabled_not_multiple(self):
        assert (
            should_evaluate_at_step(
                step=23, total_steps=23, eval_every=5, eval_last_step=False
            )
            is False
        )

    def test_eval_every_multiples(self):
        for step in [5, 10, 15, 20]:
            assert (
                should_evaluate_at_step(step=step, total_steps=20, eval_every=5) is True
            ), f"step={step} should evaluate (multiple of eval_every=5)"

    def test_non_multiples_not_evaluated(self):
        for step in [2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19]:
            assert (
                should_evaluate_at_step(
                    step=step,
                    total_steps=20,
                    eval_every=5,
                    eval_first_step=False,
                    eval_last_step=False,
                )
                is False
            ), f"step={step} should NOT evaluate"

    def test_user_example_steps23_eval5(self):
        """User's example: steps=23, eval_every=5.

        Expected: step 0, 1, 5, 10, 15, 20, 23.
        """
        expected_eval: set = {0, 1, 5, 10, 15, 20, 23}
        for step in range(0, 24):
            result: bool = should_evaluate_at_step(
                step=step, total_steps=23, eval_every=5
            )
            if step in expected_eval:
                assert result is True, f"step={step} should evaluate"
            else:
                assert result is False, f"step={step} should NOT evaluate"

    def test_user_example_steps23_eval5_no_initial(self):
        """steps=23, eval_every=5, eval_initial_prompt=False.

        Expected: step 1, 5, 10, 15, 20, 23 (no step 0).
        """
        expected_eval: set = {1, 5, 10, 15, 20, 23}
        for step in range(0, 24):
            result: bool = should_evaluate_at_step(
                step=step,
                total_steps=23,
                eval_every=5,
                eval_initial_prompt=False,
            )
            if step in expected_eval:
                assert result is True, f"step={step} should evaluate"
            else:
                assert result is False, f"step={step} should NOT evaluate"

    def test_user_example_last_disabled_not_multiple(self):
        """steps=23, eval_every=5, eval_last_step=False.

        23 is not a multiple of 5, so it should NOT eval at step 23.
        Expected: step 0, 1, 5, 10, 15, 20.
        """
        expected_eval: set = {0, 1, 5, 10, 15, 20}
        for step in range(0, 24):
            result: bool = should_evaluate_at_step(
                step=step,
                total_steps=23,
                eval_every=5,
                eval_last_step=False,
            )
            if step in expected_eval:
                assert result is True, f"step={step} should evaluate"
            else:
                assert result is False, f"step={step} should NOT evaluate"

    def test_user_example_last_disabled_is_multiple(self):
        """steps=20, eval_every=5, eval_last_step=False.

        20 IS a multiple of 5, so it should still eval (eval_every overrides).
        Expected: step 0, 1, 5, 10, 15, 20.
        """
        expected_eval: set = {0, 1, 5, 10, 15, 20}
        for step in range(0, 21):
            result: bool = should_evaluate_at_step(
                step=step,
                total_steps=20,
                eval_every=5,
                eval_last_step=False,
            )
            if step in expected_eval:
                assert result is True, f"step={step} should evaluate"
            else:
                assert result is False, f"step={step} should NOT evaluate"

    def test_eval_every_1_evaluates_all(self):
        """eval_every=1 should evaluate every step regardless of flags."""
        for step in range(0, 11):
            result: bool = should_evaluate_at_step(
                step=step,
                total_steps=10,
                eval_every=1,
                eval_initial_prompt=True,
                eval_first_step=False,
                eval_last_step=False,
            )
            assert result is True, f"step={step} should evaluate with eval_every=1"

    def test_eval_every_1_no_initial(self):
        """eval_every=1, eval_initial_prompt=False: all steps 1..N evaluate."""
        assert (
            should_evaluate_at_step(
                step=0, total_steps=10, eval_every=1, eval_initial_prompt=False
            )
            is False
        )
        for step in range(1, 11):
            assert (
                should_evaluate_at_step(
                    step=step, total_steps=10, eval_every=1, eval_initial_prompt=False
                )
                is True
            )

    def test_all_flags_false_only_eval_every_multiples(self):
        """All flags False: only eval_every multiples evaluate."""
        expected_eval: set = {5, 10, 15, 20}
        for step in range(0, 21):
            result: bool = should_evaluate_at_step(
                step=step,
                total_steps=20,
                eval_every=5,
                eval_initial_prompt=False,
                eval_first_step=False,
                eval_last_step=False,
            )
            if step in expected_eval:
                assert result is True, f"step={step} should evaluate"
            else:
                assert result is False, f"step={step} should NOT evaluate"

    def test_large_eval_every_only_flags_fire(self):
        """eval_every=99999, steps=5: only flag-based steps evaluate."""
        expected_eval: set = {0, 1, 5}
        for step in range(0, 6):
            result: bool = should_evaluate_at_step(
                step=step, total_steps=5, eval_every=99999
            )
            if step in expected_eval:
                assert result is True, f"step={step} should evaluate"
            else:
                assert result is False, f"step={step} should NOT evaluate"

    def test_raises_on_negative_step(self):
        with pytest.raises(ValidationError):
            should_evaluate_at_step(step=-1, total_steps=10, eval_every=5)

    def test_raises_on_zero_total_steps(self):
        with pytest.raises(ValidationError):
            should_evaluate_at_step(step=0, total_steps=0, eval_every=5)

    def test_raises_on_zero_eval_every(self):
        with pytest.raises(ValidationError):
            should_evaluate_at_step(step=0, total_steps=10, eval_every=0)

    def test_single_step_run(self):
        """steps=1: step 0 is baseline, step 1 is both first and last."""
        assert should_evaluate_at_step(step=0, total_steps=1, eval_every=5) is True
        assert should_evaluate_at_step(step=1, total_steps=1, eval_every=5) is True
