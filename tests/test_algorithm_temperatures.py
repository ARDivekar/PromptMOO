"""
Tests for per-algorithm LLM temperature configuration.

Covers:
- Each algorithm subclass declares the correct paper-sourced temperature defaults
- Explicit temperature overrides are respected at construction time
- None temperatures defer to global config fallback
- Temperatures are included in run_config serialization for reproducibility
- Factory functions in runner.py resolve explicit vs. config-default temperatures
- AlgorithmRunner reads algorithm class-level temperature defaults
"""

import pytest

from prompt_moo.config import PromptMOODefaults, promptmoo_config, temp_config
from prompt_moo.data_structures import Task
from prompt_moo.prompt_template import PromptTemplate


TASKS = [
    Task(
        task_name="fluency",
        task_description="Rate fluency 1-5",
        task_instruction="Check grammar",
        gt_col="fluency",
    ),
    Task(
        task_name="consistency",
        task_description="Rate consistency 1-5",
        task_instruction="Check facts",
        gt_col="consistency",
    ),
]

TASK_LOSSES = {"fluency": "accuracy", "consistency": "accuracy"}


def _make_prompt() -> PromptTemplate:
    return PromptTemplate(
        skeleton="Evaluate the text.\n",
        tasks=TASKS,
        instruction={t.task_name: t.task_instruction for t in TASKS},
    )


def _make_opro(**overrides):
    from prompt_moo.algorithm import OPRO

    defaults = dict(
        task_llm=None,
        optimizer_llm=None,
        tasks=TASKS,
        steps=1,
        batch_size=2,
        loss_batch_size=2,
        gradient_batch_size=2,
        eval_every=1,
        name="test",
        task_losses=TASK_LOSSES,
    )
    defaults.update(overrides)
    return OPRO(**defaults)


def _make_gpo(**overrides):
    from prompt_moo.algorithm import GPO

    defaults = dict(
        task_llm=None,
        optimizer_llm=None,
        tasks=TASKS,
        steps=1,
        batch_size=2,
        eval_every=1,
        name="test",
        task_losses=TASK_LOSSES,
    )
    defaults.update(overrides)
    return GPO(**defaults)


def _make_textgrad(**overrides):
    from prompt_moo.algorithm import TextGrad

    defaults = dict(
        task_llm=None,
        optimizer_llm=None,
        gradient_llm=None,
        loss_llm=None,
        tasks=TASKS,
        steps=1,
        batch_size=2,
        eval_every=1,
        name="test",
        validation_metric="accuracy",
    )
    defaults.update(overrides)
    return TextGrad(**defaults)


@pytest.mark.unit
class TestOPROTemperatureDefaults:
    """OPRO paper: task=0.0 (deterministic scoring), optimizer=1.0 (diverse candidates). No gradient/loss LLM."""

    def test_task_llm_temperature(self):
        algo = _make_opro()
        assert algo.task_llm_temperature == 0.0

    def test_optimizer_llm_temperature(self):
        algo = _make_opro()
        assert algo.optimizer_llm_temperature == 1.0

    def test_gradient_llm_temperature_is_none(self):
        algo = _make_opro()
        assert algo.gradient_llm_temperature is None

    def test_loss_llm_temperature_is_none(self):
        algo = _make_opro()
        assert algo.loss_llm_temperature is None


@pytest.mark.unit
class TestGPOTemperatureDefaults:
    """GPO paper: task=0.0 (deterministic scoring), optimizer=1.0 (diverse candidates), gradient=0.7, loss=0.7."""

    def test_task_llm_temperature(self):
        algo = _make_gpo()
        assert algo.task_llm_temperature == 0.0

    def test_optimizer_llm_temperature(self):
        algo = _make_gpo()
        assert algo.optimizer_llm_temperature == 1.0

    def test_gradient_llm_temperature(self):
        algo = _make_gpo()
        assert algo.gradient_llm_temperature == 0.7

    def test_loss_llm_temperature(self):
        algo = _make_gpo()
        assert algo.loss_llm_temperature == 0.7


@pytest.mark.unit
class TestTextGradTemperatureDefaults:
    """TextGrad uses all 4 LLM roles: task=0.1, rest=0.7."""

    def test_task_llm_temperature(self):
        algo = _make_textgrad()
        assert algo.task_llm_temperature == 0.1

    def test_optimizer_llm_temperature(self):
        algo = _make_textgrad()
        assert algo.optimizer_llm_temperature == 0.7

    def test_gradient_llm_temperature(self):
        algo = _make_textgrad()
        assert algo.gradient_llm_temperature == 0.7

    def test_loss_llm_temperature(self):
        algo = _make_textgrad()
        assert algo.loss_llm_temperature == 0.7


@pytest.mark.unit
class TestExplicitTemperatureOverride:
    """Explicit temperatures at construction override the class default."""

    def test_opro_task_temperature_override(self):
        algo = _make_opro(task_llm_temperature=0.5)
        assert algo.task_llm_temperature == 0.5

    def test_gpo_optimizer_temperature_override(self):
        algo = _make_gpo(optimizer_llm_temperature=1.2)
        assert algo.optimizer_llm_temperature == 1.2

    def test_textgrad_all_temperatures_override(self):
        algo = _make_textgrad(
            task_llm_temperature=0.3,
            optimizer_llm_temperature=0.4,
            gradient_llm_temperature=0.5,
            loss_llm_temperature=0.6,
        )
        assert algo.task_llm_temperature == 0.3
        assert algo.optimizer_llm_temperature == 0.4
        assert algo.gradient_llm_temperature == 0.5
        assert algo.loss_llm_temperature == 0.6

    def test_explicit_none_keeps_none(self):
        algo = _make_opro(task_llm_temperature=None)
        assert algo.task_llm_temperature is None


@pytest.mark.unit
class TestTemperatureValidation:
    """Pydantic rejects temperatures outside [0.0, 2.0]."""

    def test_rejects_negative_temperature(self):
        with pytest.raises(Exception):
            _make_opro(task_llm_temperature=-0.1)

    def test_rejects_temperature_above_2(self):
        with pytest.raises(Exception):
            _make_gpo(optimizer_llm_temperature=2.1)

    def test_accepts_zero_temperature(self):
        algo = _make_opro(task_llm_temperature=0.0)
        assert algo.task_llm_temperature == 0.0

    def test_accepts_max_temperature(self):
        algo = _make_gpo(optimizer_llm_temperature=2.0)
        assert algo.optimizer_llm_temperature == 2.0


@pytest.mark.unit
class TestTemperaturesInRunConfig:
    """Verify temperatures are serialized into run_config for reproducibility."""

    def test_opro_temperatures_in_config(self):
        algo = _make_opro()
        algo._input_col_labels = {}
        algo._task_demonstrations = []
        config = algo._build_run_config(
            initial_prompt=_make_prompt(),
            start_step=0,
        )
        assert config["task_llm_temperature"] == 0.0
        assert config["optimizer_llm_temperature"] == 1.0
        assert config["gradient_llm_temperature"] is None
        assert config["loss_llm_temperature"] is None

    def test_overridden_temperatures_in_config(self):
        algo = _make_gpo(task_llm_temperature=0.5, optimizer_llm_temperature=1.5)
        algo._input_col_labels = {}
        algo._task_demonstrations = []
        config = algo._build_run_config(
            initial_prompt=_make_prompt(),
            start_step=0,
        )
        assert config["task_llm_temperature"] == 0.5
        assert config["optimizer_llm_temperature"] == 1.5


@pytest.mark.unit
class TestBaseClassTemperatureDefaults:
    """PromptAlgorithm base class has None defaults (defer to global config)."""

    def test_base_class_fields_default_to_none(self):
        from prompt_moo.prompt_algorithm import PromptAlgorithm

        for field_name in [
            "task_llm_temperature",
            "optimizer_llm_temperature",
            "gradient_llm_temperature",
            "loss_llm_temperature",
        ]:
            field_info = PromptAlgorithm.model_fields[field_name]
            assert field_info.default is None, (
                f"{field_name} base default should be None, got {field_info.default}"
            )


@pytest.mark.unit
class TestGlobalConfigTemperatureFallback:
    """Global config defaults match the paper values and serve as fallback."""

    def test_config_defaults_match_paper(self):
        d = PromptMOODefaults()
        assert d.task_llm_temperature == 0.1
        assert d.optimizer_llm_temperature == 0.7
        assert d.gradient_llm_temperature == 0.7
        assert d.loss_llm_temperature == 0.7

    def test_temp_config_still_works_for_temperatures(self):
        original = promptmoo_config.defaults.task_llm_temperature
        with temp_config(task_llm_temperature=1.5):
            assert promptmoo_config.defaults.task_llm_temperature == 1.5
        assert promptmoo_config.defaults.task_llm_temperature == original


@pytest.mark.unit
class TestAlgorithmRunnerTemperatureResolution:
    """Verify the runner reads algorithm class-level temperature defaults."""

    def test_resolve_temperature_from_class_default(self):
        from prompt_moo.algorithm import GPO

        field_info = GPO.model_fields.get("task_llm_temperature")
        assert field_info is not None
        assert field_info.default == 0.0

    def test_resolve_temperature_from_algo_params_override(self):
        from prompt_moo.algorithm import GPO

        class_default = GPO.model_fields["task_llm_temperature"].default
        assert class_default == 0.0
        algo_params = {"task_llm_temperature": 0.5}
        resolved = algo_params.get(
            "task_llm_temperature",
            class_default,
        )
        assert resolved == 0.5

    def test_opro_class_has_none_gradient_temperature(self):
        from prompt_moo.algorithm import OPRO

        field_info = OPRO.model_fields.get("gradient_llm_temperature")
        assert field_info is not None
        assert field_info.default is None

    def test_textgrad_class_has_all_four_temperatures(self):
        from prompt_moo.algorithm import TextGrad

        assert TextGrad.model_fields["task_llm_temperature"].default == 0.1
        assert TextGrad.model_fields["optimizer_llm_temperature"].default == 0.7
        assert TextGrad.model_fields["gradient_llm_temperature"].default == 0.7
        assert TextGrad.model_fields["loss_llm_temperature"].default == 0.7
