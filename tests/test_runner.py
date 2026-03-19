"""
Tests for runner.py: LLM factory functions, prompt building, algorithm instantiation.

Mix of unit tests (no API calls) and integration tests (require API key).
"""
import pytest

from prompt_moo.data_structures import Task


FLUENCY_TASK = Task(
    task_name="fluency",
    task_description="1-5 grammar score",
    task_instruction="Check grammar, clarity, and readability",
    gt_col="fluency",
)


@pytest.mark.unit
class TestPromptSkeleton:
    """Tests for build_prompt_skeleton and get_initial_prompt."""

    def test_single_task_skeleton(self):
        from runner import build_prompt_skeleton
        skeleton = build_prompt_skeleton(
            dataset_name="SummEval",
            tasks=[FLUENCY_TASK],
        )
        assert "fluency" in skeleton
        assert "coherence" not in skeleton
        assert "JSON" not in skeleton or "format" in skeleton.lower()

    def test_multi_task_skeleton(self):
        from runner import build_prompt_skeleton
        coherence_task = Task(
            task_name="coherence",
            task_description="1-5 flow",
            task_instruction="Assess logical flow",
            gt_col="coherence",
        )
        skeleton = build_prompt_skeleton(
            dataset_name="SummEval",
            tasks=[FLUENCY_TASK, coherence_task],
        )
        assert "fluency" in skeleton
        assert "coherence" in skeleton

    def test_initial_prompt_returns_prompt_template(self):
        from runner import get_initial_prompt
        from prompt_moo.prompt_template_utils import PromptTemplate
        prompt = get_initial_prompt(
            dataset_name="SummEval",
            tasks=[FLUENCY_TASK],
        )
        assert isinstance(prompt, PromptTemplate)
        prompt_str = prompt.to_str()
        assert "fluency" in prompt_str

    def test_unknown_dataset_raises(self):
        from runner import build_prompt_skeleton
        with pytest.raises(KeyError):
            build_prompt_skeleton(dataset_name="NonexistentDataset", tasks=[FLUENCY_TASK])


@pytest.mark.unit
class TestTaskLosses:
    """Tests for get_task_losses."""

    def test_get_all_losses(self):
        from runner import get_task_losses
        losses = get_task_losses(dataset_name="SummEval")
        assert "fluency" in losses
        assert "coherence" in losses
        assert losses["fluency"] == "accuracy"

    def test_get_filtered_losses(self):
        from runner import get_task_losses
        losses = get_task_losses(dataset_name="SummEval", tasks=[FLUENCY_TASK])
        assert "fluency" in losses
        assert "coherence" not in losses


@pytest.mark.unit
class TestLLMConfigs:
    """Tests for LLM_CONFIGS dictionary."""

    def test_llama_config_exists(self):
        from runner import LLM_CONFIGS
        assert "llama3.1" in LLM_CONFIGS
        assert "task_model" in LLM_CONFIGS["llama3.1"]
        assert "other_model" in LLM_CONFIGS["llama3.1"]

    def test_all_configs_have_required_keys(self):
        from runner import LLM_CONFIGS
        for name, config in LLM_CONFIGS.items():
            assert "task_model" in config, f"{name} missing task_model"
            assert "other_model" in config, f"{name} missing other_model"
            assert "provider_order" in config, f"{name} missing provider_order"

    def test_model_names_include_provider_prefix(self):
        from runner import LLM_CONFIGS
        for name, config in LLM_CONFIGS.items():
            assert "openrouter/" in config["task_model"], (
                f"{name} task_model missing provider prefix: {config['task_model']}"
            )
            assert "openrouter/" in config["other_model"], (
                f"{name} other_model missing provider prefix: {config['other_model']}"
            )


@pytest.mark.integration
class TestLLMCreation:
    """Integration tests: create actual LLM workers."""

    def test_create_task_llm(self, api_key, shared_limits):
        from runner import create_task_llm
        llm = create_task_llm(llm="llama3.1", api_key=api_key, limits=shared_limits)
        assert llm is not None
        llm.stop()

    def test_create_optimizer_llm(self, api_key, shared_limits):
        from runner import create_optimizer_llm
        llm = create_optimizer_llm(llm="llama3.1", api_key=api_key, limits=shared_limits)
        assert llm is not None
        llm.stop()

    def test_create_gradient_llm(self, api_key, shared_limits):
        from runner import create_gradient_llm
        llm = create_gradient_llm(llm="llama3.1", api_key=api_key, limits=shared_limits)
        assert llm is not None
        llm.stop()

    def test_create_loss_llm(self, api_key, shared_limits):
        from runner import create_loss_llm
        llm = create_loss_llm(llm="llama3.1", api_key=api_key, limits=shared_limits)
        assert llm is not None
        llm.stop()

    def test_unknown_llm_raises(self, api_key, shared_limits):
        from runner import create_task_llm
        with pytest.raises(ValueError, match="Unknown LLM"):
            create_task_llm(llm="nonexistent-model", api_key=api_key, limits=shared_limits)


@pytest.mark.integration
class TestAlgorithmInstantiation:
    """Integration tests: instantiate algorithm objects (no training)."""

    def test_opro_instantiation(self, api_key, shared_limits):
        from runner import create_task_llm, create_optimizer_llm, get_task_losses
        from prompt_moo.algorithm import OPRO

        task_llm = create_task_llm(llm="llama3.1", api_key=api_key, limits=shared_limits)
        optimizer_llm = create_optimizer_llm(llm="llama3.1", api_key=api_key, limits=shared_limits)
        task_losses = get_task_losses(dataset_name="SummEval", tasks=[FLUENCY_TASK])

        algo = OPRO(
            task_llm=task_llm,
            optimizer_llm=optimizer_llm,
            task_losses=task_losses,
            tasks=[FLUENCY_TASK],
            steps=2,
            batch_size=5,
            loss_batch_size=2,
            gradient_batch_size=1,
            eval_every=1,
            name="test_opro",
            k=3,
        )
        assert algo is not None

        task_llm.stop()
        optimizer_llm.stop()
