"""
Tests for runner.py: LLM factory functions, prompt building, algorithm instantiation.

Mix of unit tests (no API calls) and integration tests (require API key).
"""

import pytest

from dataset import SummEval
from prompt_moo.data_structures import Task
from prompt_moo.prompt_template import PromptTemplate


FLUENCY_TASK = Task(
    task_name="fluency",
    task_description="1-5 grammar score",
    task_instruction="Check grammar, clarity, and readability",
    gt_col="fluency",
)

SUMMEVAL_DATASET = SummEval(data_dir="expt")


@pytest.mark.unit
class TestPromptSkeleton:
    """Tests for build_prompt_skeleton and get_initial_prompt.

    After the output-format extraction fix, build_prompt_skeleton returns
    the evaluation directive + output constraints WITHOUT the per-task
    ``## Output format`` section.  The per-task format specs live in
    ``task_output_formats`` on PromptTemplate and are rendered dynamically
    by ``render_instructions()``.
    """

    def test_skeleton_does_not_contain_output_format_section(self):
        """The skeleton must NOT bake in the ## Output format section.
        It should contain only the evaluation directive and constraints."""
        from runner import build_prompt_skeleton

        skeleton: str = build_prompt_skeleton(
            dataset=SUMMEVAL_DATASET,
            tasks=[FLUENCY_TASK],
        )
        assert "## Output format" not in skeleton, (
            "Skeleton must not contain ## Output format — it is now rendered "
            "dynamically from task_output_formats on PromptTemplate"
        )

    def test_skeleton_contains_evaluation_directive(self):
        """The skeleton must still contain the evaluation directive text."""
        from runner import build_prompt_skeleton

        skeleton: str = build_prompt_skeleton(
            dataset=SUMMEVAL_DATASET,
            tasks=[FLUENCY_TASK],
        )
        assert "Do NOT include reasoning" in skeleton

    def test_initial_prompt_returns_prompt_template(self):
        from runner import get_initial_prompt
        from prompt_moo.prompt_template import PromptTemplate

        prompt: PromptTemplate = get_initial_prompt(
            dataset=SUMMEVAL_DATASET,
            tasks=[FLUENCY_TASK],
        )
        assert isinstance(prompt, PromptTemplate)
        prompt_str: str = prompt.render_instructions()
        assert "fluency" in prompt_str

    def test_initial_prompt_populates_task_output_formats(self):
        """get_initial_prompt must populate task_output_formats on the
        PromptTemplate so that render_instructions(task_filter=...) can
        filter the output format to a single task."""
        from runner import get_initial_prompt

        prompt: PromptTemplate = get_initial_prompt(
            dataset=SUMMEVAL_DATASET,
            tasks=[FLUENCY_TASK],
        )
        assert len(prompt.task_output_formats) > 0, (
            "get_initial_prompt must populate task_output_formats"
        )
        assert "fluency" in prompt.task_output_formats, (
            "task_output_formats must contain the 'fluency' task"
        )

    def test_initial_prompt_multi_task_output_formats(self):
        """Multi-task initial prompt must have all task entries in
        task_output_formats."""
        from runner import get_initial_prompt

        coherence_task: Task = Task(
            task_name="coherence",
            task_description="1-5 flow",
            task_instruction="Assess logical flow",
            gt_col="coherence",
        )
        prompt: PromptTemplate = get_initial_prompt(
            dataset=SUMMEVAL_DATASET,
            tasks=[FLUENCY_TASK, coherence_task],
        )
        assert "fluency" in prompt.task_output_formats
        assert "coherence" in prompt.task_output_formats

    def test_initial_prompt_render_all_tasks_includes_output_format(self):
        """render_instructions() with no filter must still show ## Output format
        with all tasks when task_output_formats is populated."""
        from runner import get_initial_prompt

        coherence_task: Task = Task(
            task_name="coherence",
            task_description="1-5 flow",
            task_instruction="Assess logical flow",
            gt_col="coherence",
        )
        prompt: PromptTemplate = get_initial_prompt(
            dataset=SUMMEVAL_DATASET,
            tasks=[FLUENCY_TASK, coherence_task],
        )
        rendered: str = prompt.render_instructions()
        assert "## Output format" in rendered
        assert '"fluency"' in rendered
        assert '"coherence"' in rendered

    def test_initial_prompt_render_filtered_excludes_other_tasks(self):
        """render_instructions(task_filter=...) must show ONLY the target
        task in both output format and instructions sections."""
        from runner import get_initial_prompt

        coherence_task: Task = Task(
            task_name="coherence",
            task_description="1-5 flow",
            task_instruction="Assess logical flow",
            gt_col="coherence",
        )
        prompt: PromptTemplate = get_initial_prompt(
            dataset=SUMMEVAL_DATASET,
            tasks=[FLUENCY_TASK, coherence_task],
        )
        rendered: str = prompt.render_instructions(task_filter="fluency")
        assert '"fluency"' in rendered
        assert '"coherence"' not in rendered, (
            "task_filter='fluency' must exclude coherence from output format"
        )
        assert "Assess logical flow" not in rendered

    def test_unknown_task_raises(self):
        from runner import build_prompt_skeleton

        unknown_task = Task(
            task_name="nonexistent_metric",
            task_description="Does not exist",
            task_instruction="N/A",
            gt_col="nonexistent_metric",
        )
        with pytest.raises(ValueError, match="not found in task_output_specs"):
            build_prompt_skeleton(dataset=SUMMEVAL_DATASET, tasks=[unknown_task])


@pytest.mark.unit
class TestTaskLosses:
    """Tests for get_task_losses."""

    def test_get_all_losses(self):
        from runner import get_task_losses

        losses = get_task_losses(dataset=SUMMEVAL_DATASET)
        assert "fluency" in losses
        assert "coherence" in losses
        assert losses["fluency"] == "accuracy"

    def test_get_filtered_losses(self):
        from runner import get_task_losses

        losses = get_task_losses(dataset=SUMMEVAL_DATASET, tasks=[FLUENCY_TASK])
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

        llm = create_optimizer_llm(
            llm="llama3.1", api_key=api_key, limits=shared_limits
        )
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
            create_task_llm(
                llm="nonexistent-model", api_key=api_key, limits=shared_limits
            )


@pytest.mark.integration
class TestAlgorithmInstantiation:
    """Integration tests: instantiate algorithm objects (no training)."""

    def test_opro_instantiation(self, api_key, shared_limits):
        from runner import create_task_llm, create_optimizer_llm, get_task_losses
        from prompt_moo.algorithm import OPRO

        task_llm = create_task_llm(
            llm="llama3.1", api_key=api_key, limits=shared_limits
        )
        optimizer_llm = create_optimizer_llm(
            llm="llama3.1", api_key=api_key, limits=shared_limits
        )
        task_losses = get_task_losses(dataset=SUMMEVAL_DATASET, tasks=[FLUENCY_TASK])

        algo = OPRO(
            task_llm=task_llm,
            optimizer_llm=optimizer_llm,
            task_losses=task_losses,
            tasks=[FLUENCY_TASK],
            steps=2,
            batch_size=5,
            eval_every=1,
            name="test_opro",
            k=3,
        )
        assert algo is not None

        task_llm.stop()
        optimizer_llm.stop()
