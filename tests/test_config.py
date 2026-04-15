"""
Unit tests for prompt_moo.config — centralized configuration system.

Covers:
- Default values are correct
- Mutation works via attribute assignment
- temp_config scopes and restores correctly
- Unknown keys in temp_config raise ValueError
- reset_to_defaults restores original values
- Pydantic validation rejects invalid values
"""

import pytest

from prompt_moo.config import (
    PromptMOOConfig,
    PromptMOODefaults,
    promptmoo_config,
    temp_config,
)


@pytest.mark.unit
class TestPromptMOODefaults:
    """Verify default values match the migration plan."""

    def test_batch_invocation_timeout_default(self):
        d = PromptMOODefaults()
        assert d.batch_invocation_timeout == 3600.0

    def test_per_role_timeout_defaults(self):
        d = PromptMOODefaults()
        assert d.task_llm_timeout == 60.0
        assert d.optimizer_llm_timeout == 600.0
        assert d.gradient_llm_timeout == 600.0
        assert d.loss_llm_timeout == 60.0

    def test_per_role_max_tokens_defaults(self):
        d = PromptMOODefaults()
        assert d.task_llm_max_tokens == 256
        assert d.optimizer_llm_max_tokens == 4096
        assert d.gradient_llm_max_tokens == 2048
        assert d.loss_llm_max_tokens == 256

    def test_per_role_temperature_defaults(self):
        d = PromptMOODefaults()
        assert d.task_llm_temperature == 0.1
        assert d.optimizer_llm_temperature == 0.7
        assert d.gradient_llm_temperature == 0.7
        assert d.loss_llm_temperature == 0.7

    def test_rate_limit_defaults(self):
        d = PromptMOODefaults()
        assert d.max_parallel_calls == 20
        assert d.max_rpm == 1000
        assert d.max_input_tpm == 10_000_000
        assert d.max_output_tpm == 1_000_000

    def test_retry_defaults(self):
        d = PromptMOODefaults()
        assert d.num_retries == 10
        assert d.retry_algorithm == "Fibonacci"
        assert d.retry_wait == 2.0
        assert d.retry_jitter == 0.7

    def test_training_defaults(self):
        d = PromptMOODefaults()
        assert d.substep_delay == 1.5
        assert d.verbosity == 1


@pytest.mark.unit
class TestPromptMOOConfigMutation:
    """Verify config can be mutated and read at call sites."""

    def test_mutate_and_read(self):
        """Direct mutation of defaults is reflected immediately."""
        original = promptmoo_config.defaults.task_llm_timeout
        try:
            promptmoo_config.defaults.task_llm_timeout = 999.0
            assert promptmoo_config.defaults.task_llm_timeout == 999.0
        finally:
            promptmoo_config.defaults.task_llm_timeout = original

    def test_validation_rejects_negative_timeout(self):
        """Pydantic validation prevents invalid values."""
        with pytest.raises(Exception):
            promptmoo_config.defaults.task_llm_timeout = -1.0

    def test_validation_rejects_temperature_above_2(self):
        with pytest.raises(Exception):
            promptmoo_config.defaults.task_llm_temperature = 3.0

    def test_extra_fields_forbidden(self):
        """PromptMOODefaults rejects unknown fields at construction time."""
        with pytest.raises(Exception):
            PromptMOODefaults(nonexistent_field=42)


@pytest.mark.unit
class TestTempConfig:
    """Verify temp_config scopes and restores correctly."""

    def test_overrides_within_scope(self):
        original_timeout = promptmoo_config.defaults.batch_invocation_timeout
        with temp_config(batch_invocation_timeout=123.0):
            assert promptmoo_config.defaults.batch_invocation_timeout == 123.0
        assert promptmoo_config.defaults.batch_invocation_timeout == original_timeout

    def test_multiple_overrides(self):
        orig_timeout = promptmoo_config.defaults.task_llm_timeout
        orig_verbosity = promptmoo_config.defaults.verbosity
        with temp_config(task_llm_timeout=42.0, verbosity=0):
            assert promptmoo_config.defaults.task_llm_timeout == 42.0
            assert promptmoo_config.defaults.verbosity == 0
        assert promptmoo_config.defaults.task_llm_timeout == orig_timeout
        assert promptmoo_config.defaults.verbosity == orig_verbosity

    def test_restores_on_exception(self):
        """Values are restored even if the block raises."""
        original = promptmoo_config.defaults.num_retries
        with pytest.raises(RuntimeError):
            with temp_config(num_retries=99):
                assert promptmoo_config.defaults.num_retries == 99
                raise RuntimeError("boom")
        assert promptmoo_config.defaults.num_retries == original

    def test_unknown_key_raises(self):
        with pytest.raises(ValueError, match="Unknown config key"):
            with temp_config(totally_fake_key=42):
                pass

    def test_nested_temp_config(self):
        """Nested temp_config scopes restore independently."""
        original = promptmoo_config.defaults.max_rpm
        with temp_config(max_rpm=100):
            assert promptmoo_config.defaults.max_rpm == 100
            with temp_config(max_rpm=50):
                assert promptmoo_config.defaults.max_rpm == 50
            assert promptmoo_config.defaults.max_rpm == 100
        assert promptmoo_config.defaults.max_rpm == original


@pytest.mark.unit
class TestResetToDefaults:
    """Verify reset_to_defaults restores original values."""

    def test_reset_after_mutation(self):
        promptmoo_config.defaults.task_llm_timeout = 999.0
        promptmoo_config.defaults.verbosity = 99
        promptmoo_config.reset_to_defaults()

        assert promptmoo_config.defaults.task_llm_timeout == 60.0
        assert promptmoo_config.defaults.verbosity == 1
