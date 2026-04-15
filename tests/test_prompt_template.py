"""
Unit tests for prompt template rendering after structural cleanup.

Validates:
- Instruction is rendered exactly ONCE in the task LLM prompt
- task_description does NOT appear in the task LLM prompt
- render_for_optimizer() produces a version with placeholders (no real data)
- Singular/plural task header is correct
- Skeleton output format uses explicit valid values (no invalid-format priming)
"""

import pytest

from prompt_moo.data_structures import Task
from prompt_moo.prompt_template import PromptTemplate


COHERENCE_TASK = Task(
    task_name="coherence",
    task_description="Evaluate the logical structure and organization of the summary",
    task_instruction="Rate coherence from 1 to 5.",
    gt_col="coherence",
)

FLUENCY_TASK = Task(
    task_name="fluency",
    task_description="Evaluate the fluency and readability of the summary",
    task_instruction="Rate fluency from 1 to 5.",
    gt_col="fluency",
)

SKELETON_SINGLE = (
    "Evaluate the summary. Output JSON with the requested task score. "
    "Do NOT include reasoning or explanations.\n"
    'Output format: {"coherence": 1 | 2 | 3 | 4 | 5}\n'
)

SKELETON_MULTI = (
    "Evaluate the summary. Output JSON with the requested task scores. "
    "Do NOT include reasoning or explanations.\n"
    'Output format: {"coherence": ..., "fluency": ...}\n'
)


def _make_single_template() -> PromptTemplate:
    return PromptTemplate(
        skeleton=SKELETON_SINGLE,
        instruction={"coherence": COHERENCE_TASK.task_instruction},
        tasks=[COHERENCE_TASK],
    )


def _make_multi_template() -> PromptTemplate:
    return PromptTemplate(
        skeleton=SKELETON_MULTI,
        instruction={
            "coherence": COHERENCE_TASK.task_instruction,
            "fluency": FLUENCY_TASK.task_instruction,
        },
        tasks=[COHERENCE_TASK, FLUENCY_TASK],
    )


@pytest.mark.unit
class TestMultiObjectivePromptTemplateToStr:
    """Validate the task LLM prompt rendered by render_instructions()."""

    def test_instruction_appears_exactly_once(self):
        """The mutable instruction text must appear exactly once."""
        tmpl = _make_single_template()
        prompt = tmpl.render_instructions()
        count = prompt.count(COHERENCE_TASK.task_instruction)
        assert count == 1, (
            f"Instruction should appear exactly once, found {count} times.\n"
            f"Full prompt:\n{prompt}"
        )

    def test_task_description_not_in_prompt(self):
        """task_description must NOT appear in the task LLM prompt."""
        tmpl = _make_single_template()
        prompt = tmpl.render_instructions()
        assert COHERENCE_TASK.task_description not in prompt, (
            f"task_description should not be in the task LLM prompt.\n"
            f"Found: {COHERENCE_TASK.task_description!r}\n"
            f"Full prompt:\n{prompt}"
        )

    def test_multi_task_instructions_each_appear_once(self):
        """In multi-task mode, each instruction appears exactly once."""
        tmpl = _make_multi_template()
        prompt = tmpl.render_instructions()
        for task in [COHERENCE_TASK, FLUENCY_TASK]:
            count = prompt.count(task.task_instruction)
            assert count == 1, f"{task.task_name} instruction appeared {count} times"

    def test_multi_task_descriptions_not_in_prompt(self):
        """No task_description from any task in multi-task prompt."""
        tmpl = _make_multi_template()
        prompt = tmpl.render_instructions()
        for task in [COHERENCE_TASK, FLUENCY_TASK]:
            assert task.task_description not in prompt

    def test_single_task_header_singular(self):
        """Single-task template uses '## Instructions:' header."""
        tmpl = _make_single_template()
        prompt = tmpl.render_instructions()
        assert "## Instructions:" in prompt
        assert "coherence" in prompt

    def test_multi_task_header_plural(self):
        """Multi-task template uses '## Instructions:' header with all tasks."""
        tmpl = _make_multi_template()
        prompt = tmpl.render_instructions()
        assert "## Instructions:" in prompt
        assert "coherence" in prompt
        assert "fluency" in prompt

    def test_skeleton_preserved(self):
        """The frozen skeleton appears in the output."""
        tmpl = _make_single_template()
        prompt = tmpl.render_instructions()
        assert "Evaluate the summary" in prompt

    def test_instructions_section_present(self):
        """The ## Instructions: heading is present."""
        tmpl = _make_single_template()
        prompt = tmpl.render_instructions()
        assert "## Instructions:" in prompt


@pytest.mark.unit
class TestMultiObjectivePromptTemplateToTemplateStr:
    """Validate render_for_optimizer() for optimizer context."""

    def test_template_str_contains_placeholder(self):
        """render_for_optimizer() must include a sample placeholder."""
        tmpl = _make_single_template()
        template = tmpl.render_for_optimizer()
        assert "## Sample" in template
        assert "input data will be inserted here" in template

    def test_template_str_contains_instruction(self):
        """render_for_optimizer() includes the current instruction."""
        tmpl = _make_single_template()
        template = tmpl.render_for_optimizer()
        assert COHERENCE_TASK.task_instruction in template

    def test_template_str_no_real_data(self):
        """render_for_optimizer() must not contain any actual data values."""
        tmpl = _make_single_template()
        template = tmpl.render_for_optimizer()
        assert "chelsea" not in template.lower()
        assert "cat sat" not in template.lower()


@pytest.mark.unit
class TestBuildPromptSkeletonIntegration:
    """Integration tests verifying skeleton content comes from dataset.py."""

    def test_skeleton_no_invalid_format_priming(self):
        """Skeleton should NOT contain invalid format examples like '4/5'."""
        import sys
        import os

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "expt"))
        from dataset import SummEval, WildGuard, BRIGHTER

        for ds_cls in [SummEval, WildGuard, BRIGHTER]:
            prefix = ds_cls.prompt_prefix
            name = ds_cls.dataset_name
            assert "4/5" not in prefix, f"{name} skeleton primes with '4/5'"
            assert "4|5" not in prefix, f"{name} skeleton primes with '4|5'"
            assert "0/3" not in prefix, f"{name} skeleton primes with '0/3'"
            assert "0|3" not in prefix, f"{name} skeleton primes with '0|3'"

    def test_skeleton_uses_task_not_metric(self):
        """Skeleton should say 'task' not 'metric'."""
        import sys
        import os

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "expt"))
        from dataset import SummEval, WildGuard, BRIGHTER

        for ds_cls in [SummEval, WildGuard, BRIGHTER]:
            prefix = ds_cls.prompt_prefix
            name = ds_cls.dataset_name
            assert "metric" not in prefix.lower(), (
                f"{name} skeleton still uses 'metric': {prefix!r}"
            )

    def test_output_formats_show_valid_values(self):
        """task_output_formats should indicate the valid value range."""
        import sys
        import os

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "expt"))
        from dataset import SummEval, WildGuard, BRIGHTER

        for task_name, fmt in SummEval.task_output_formats.items():
            assert "1" in fmt and "5" in fmt, (
                f"SummEval {task_name} format should reference the 1-5 scale: {fmt!r}"
            )

        for task_name, fmt in WildGuard.task_output_formats.items():
            assert "|" in fmt, (
                f"WildGuard {task_name} format should use '|' separators: {fmt!r}"
            )

        for task_name, fmt in BRIGHTER.task_output_formats.items():
            assert "|" in fmt, (
                f"BRIGHTER {task_name} format should use '|' separators: {fmt!r}"
            )
            assert "none" in fmt.lower() or "0" in fmt, (
                f"BRIGHTER {task_name} format should label the low end: {fmt!r}"
            )
            assert "intense" in fmt.lower() or "3" in fmt, (
                f"BRIGHTER {task_name} format should label the high end: {fmt!r}"
            )

    def test_no_dataset_configs_in_runner(self):
        """runner.py should NOT have a DATASET_CONFIGS dict."""
        import sys
        import os

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "expt"))
        import runner

        assert not hasattr(runner, "DATASET_CONFIGS"), (
            "DATASET_CONFIGS still exists in runner.py. "
            "All dataset-specific content should live in dataset.py."
        )


@pytest.mark.unit
class TestSeedInstructionsMinimal:
    """Verify seed instructions do not contain semantic anchors."""

    def test_summeval_no_semantic_anchors(self):
        """SummEval initial instructions should not define what scores mean."""
        import sys
        import os

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "expt"))
        from dataset import SummEval

        banned_phrases = [
            "incoherent",
            "very coherent",
            "not relevant",
            "highly relevant",
            "very poor",
            "excellent",
            "inconsistent",
            "fully consistent",
            "Consider grammar",
            "Consider logical flow",
            "Consider if it captures",
            "Check for factual alignment",
        ]
        for task in SummEval.tasks:
            for phrase in banned_phrases:
                assert phrase not in task.task_instruction, (
                    f"Task '{task.task_name}' instruction contains semantic anchor "
                    f"'{phrase}' that the optimizer should discover.\n"
                    f"Instruction: {task.task_instruction!r}"
                )

    def test_summeval_instructions_contain_scale(self):
        """SummEval instructions must still reference the 1-5 scale."""
        import sys
        import os

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "expt"))
        from dataset import SummEval

        for task in SummEval.tasks:
            assert "1" in task.task_instruction and "5" in task.task_instruction, (
                f"Task '{task.task_name}' instruction must reference 1-5 scale.\n"
                f"Instruction: {task.task_instruction!r}"
            )


@pytest.mark.unit
class TestDatasetInputColLabels:
    """Verify datasets define input_col_labels."""

    def test_summeval_has_labels(self):
        import sys
        import os

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "expt"))
        from dataset import SummEval

        assert len(SummEval.input_col_labels) > 0
        for col in SummEval.input_cols:
            assert col in SummEval.input_col_labels, (
                f"SummEval missing label for input column '{col}'"
            )

    def test_wildguard_has_labels(self):
        import sys
        import os

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "expt"))
        from dataset import WildGuard

        assert len(WildGuard.input_col_labels) > 0
        for col in WildGuard.input_cols:
            assert col in WildGuard.input_col_labels

    def test_brighter_has_labels(self):
        import sys
        import os

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "expt"))
        from dataset import BRIGHTER

        assert len(BRIGHTER.input_col_labels) > 0
        for col in BRIGHTER.input_cols:
            assert col in BRIGHTER.input_col_labels

    def test_labels_are_human_readable(self):
        """Labels should not be raw column names (snake_case)."""
        import sys
        import os

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "expt"))
        from dataset import SummEval

        for col, label in SummEval.input_col_labels.items():
            assert label != col, (
                f"Label for '{col}' is just the raw column name. "
                f"Use a human-readable label instead."
            )


class TestPromptTemplateOutputFormatFiltering:
    """Verify that task_output_formats are filtered by task_filter in render_instructions."""

    def _make_multi_task_template(self):
        from prompt_moo.data_structures import Task
        from prompt_moo.prompt_template import PromptTemplate

        tasks = [
            Task(
                task_name="fluency",
                task_description="desc",
                task_instruction="Rate fluency.",
                gt_col="fluency",
            ),
            Task(
                task_name="consistency",
                task_description="desc",
                task_instruction="Rate consistency.",
                gt_col="consistency",
            ),
        ]
        return PromptTemplate(
            skeleton="Evaluate the summary. Output JSON.",
            tasks=tasks,
            instruction={t.task_name: t.task_instruction for t in tasks},
            input_col_labels={},
            task_output_formats={"fluency": "1|2|3|4|5", "consistency": "1|2|3|4|5"},
        )

    def test_render_all_tasks_shows_full_output_format(self):
        """Without task_filter, render_instructions shows all tasks in output format."""
        pt = self._make_multi_task_template()
        rendered: str = pt.render_instructions()
        assert '"fluency"' in rendered
        assert '"consistency"' in rendered
        assert "## Output format" in rendered

    def test_render_filtered_shows_only_target_task_format(self):
        """With task_filter, render_instructions shows only the target task in output format."""
        pt = self._make_multi_task_template()
        rendered: str = pt.render_instructions(task_filter="fluency")
        assert '"fluency"' in rendered
        assert '"consistency"' not in rendered

    def test_render_filtered_still_has_output_format_section(self):
        """Even with task_filter, the ## Output format section should be present."""
        pt = self._make_multi_task_template()
        rendered: str = pt.render_instructions(task_filter="fluency")
        assert "## Output format" in rendered

    def test_render_no_output_formats_uses_skeleton_verbatim(self):
        """When task_output_formats is empty, skeleton is passed through as-is
        (backward compatibility with tests that bake the format into the skeleton)."""
        from prompt_moo.data_structures import Task
        from prompt_moo.prompt_template import PromptTemplate

        tasks = [
            Task(
                task_name="fluency",
                task_description="desc",
                task_instruction="Rate fluency.",
                gt_col="fluency",
            ),
        ]
        skeleton_with_format: str = (
            "Evaluate the summary.\n\n"
            "## Output format (follow this EXACTLY):\n"
            '{\n  "fluency": 1|2|3|4|5\n}'
        )
        pt = PromptTemplate(
            skeleton=skeleton_with_format,
            tasks=tasks,
            instruction={"fluency": "Rate fluency."},
            input_col_labels={},
        )
        rendered: str = pt.render_instructions()
        assert '"fluency": 1|2|3|4|5' in rendered
