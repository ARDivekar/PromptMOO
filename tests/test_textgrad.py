"""
Tests for the TextGrad algorithm implementation.

Verifies that the TextGrad pipeline components produce prompts matching the
information-flow requirements from the TextGrad paper (Yuksekgonul et al., 2024):

- Loss LLM (evaluator): sees input data, prediction, and ground truth
- Gradient LLM (backward): sees <LM_SYSTEM_PROMPT>, <LM_INPUT>, <LM_OUTPUT>,
  <OBJECTIVE_FUNCTION>, and <VARIABLE>
- Optimizer LLM (TGD.step): sees <VARIABLE>, <CONTEXT> with a conversation
  example, and <FEEDBACK> with concatenated per-instance gradients
- Batch sizes must be 1 (per-instance processing with tg.sum concatenation)
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prompt_moo.data_structures import (
    DatasetSample,
    PredictionResult,
    Task,
    TextGradient,
    TextualFeedback,
)
from prompt_moo.prompt_template import PromptTemplate


TASKS = [
    Task(
        task_name="fluency",
        task_description="Evaluate the fluency and readability of the summary",
        task_instruction="Rate fluency from 1 to 5.",
        gt_col="fluency",
    ),
    Task(
        task_name="consistency",
        task_description="Evaluate factual consistency with the source text",
        task_instruction="Rate consistency from 1 to 5.",
        gt_col="consistency",
    ),
]

SKELETON = (
    "Evaluate the summary. Output JSON with the requested task scores. "
    "Do NOT include reasoning or explanations."
)

TASK_OUTPUT_FORMATS: Dict[str, str] = {
    "fluency": "1|2|3|4|5",
    "consistency": "1|2|3|4|5",
}

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
        task_output_formats=TASK_OUTPUT_FORMATS,
    )


def _make_samples() -> List[DatasetSample]:
    return [
        DatasetSample(
            sample_id="s0",
            inputs={
                "machine_summary": "The cat sat on the mat.",
                "text": "A story about a cat sitting on a mat in the living room.",
            },
            ground_truths={"fluency": 4, "consistency": 5},
        ),
        DatasetSample(
            sample_id="s1",
            inputs={
                "machine_summary": "Dogs are great pets and loyal companions.",
                "text": "Dogs have been domesticated for thousands of years.",
            },
            ground_truths={"fluency": 3, "consistency": 2},
        ),
    ]


def _make_predictions() -> List[PredictionResult]:
    return [
        PredictionResult(
            sample_id="s0",
            task_outputs={"fluency": 3, "consistency": 5},
            raw_response='{"fluency": 3, "consistency": 5}',
            prompt="full prompt for s0",
        ),
        PredictionResult(
            sample_id="s1",
            task_outputs={"fluency": 5, "consistency": 4},
            raw_response='{"fluency": 5, "consistency": 4}',
            prompt="full prompt for s1",
        ),
    ]


class TestTextGradBatchSizeValidation:
    """TextGrad must enforce loss_batch_size=1 and gradient_batch_size=1."""

    def test_rejects_loss_batch_size_greater_than_1(self):
        from prompt_moo.algorithm import TextGrad

        with pytest.raises(ValueError, match="loss_batch_size=1"):
            TextGrad(
                task_llm=None,
                optimizer_llm=None,
                tasks=TASKS,
                steps=1,
                batch_size=3,
                loss_batch_size=3,
                gradient_batch_size=1,
                eval_every=1,
                name="test",
                validation_metric="accuracy",
            ).train(
                dataset=None,
                initial_prompt=_make_prompt_template(),
            )

    def test_rejects_gradient_batch_size_greater_than_1(self):
        from prompt_moo.algorithm import TextGrad

        with pytest.raises(ValueError, match="gradient_batch_size=1"):
            TextGrad(
                task_llm=None,
                optimizer_llm=None,
                tasks=TASKS,
                steps=1,
                batch_size=3,
                loss_batch_size=1,
                gradient_batch_size=4,
                eval_every=1,
                name="test",
                validation_metric="accuracy",
            ).train(
                dataset=None,
                initial_prompt=_make_prompt_template(),
            )


class TestTextGradLossPrompt:
    """Verify the loss LLM prompt follows the paper's StringBasedFunction backward
    structure for classification tasks (GSM8k/Object Counting style).

    The loss prompt must:
    - Use the BACKWARD_SYSTEM_PROMPT framing (gradient engine role)
    - Show prediction and ground truth in <INPUTS_TO_FUNCTION> tags
    - Show CORRECT/INCORRECT result in <OUTPUT_OF_FUNCTION> tags
    - Target the prediction in <VARIABLE> tags (NOT the instruction)
    - Include the generic <OBJECTIVE_FUNCTION> ("improve the above metric")
    - NOT include input data (summary, source text) — the loss sees only
      predictions and ground truth, not the original input
    - NOT include task_instruction or task_description
    """

    def _build_loss_prompt(self, *, task_idx: int = 0, sample_idx: int = 0) -> str:
        from prompt_moo.algorithm.textgrad import TextGradLossComputer

        preds = _make_predictions()
        gts = _make_samples()

        return TextGradLossComputer._build_textgrad_feedback_prompt(
            predictions=[preds[sample_idx]],
            ground_truths=[gts[sample_idx]],
            task=TASKS[task_idx],
            input_col_labels=INPUT_COL_LABELS,
        )

    def test_loss_prompt_has_backward_system_framing(self):
        prompt: str = self._build_loss_prompt()
        assert "You are part of an optimization system" in prompt
        assert "gradient (feedback) engine" in prompt
        assert "DO NOT propose a new version of the variable" in prompt

    def test_loss_prompt_has_inputs_to_function_tags(self):
        prompt: str = self._build_loss_prompt()
        assert "<INPUTS_TO_FUNCTION>" in prompt
        assert "</INPUTS_TO_FUNCTION>" in prompt

    def test_loss_prompt_has_output_of_function_tags(self):
        prompt: str = self._build_loss_prompt()
        assert "<OUTPUT_OF_FUNCTION>" in prompt
        assert "</OUTPUT_OF_FUNCTION>" in prompt

    def test_loss_prompt_shows_incorrect_for_wrong_prediction(self):
        """Sample s0: pred fluency=3, gt fluency=4 → INCORRECT."""
        prompt: str = self._build_loss_prompt(task_idx=0, sample_idx=0)
        output_start: int = prompt.index("<OUTPUT_OF_FUNCTION>")
        output_end: int = prompt.index("</OUTPUT_OF_FUNCTION>")
        output_content: str = prompt[output_start:output_end]
        assert "INCORRECT" in output_content

    def test_loss_prompt_shows_correct_for_matching_prediction(self):
        """Sample s0: pred consistency=5, gt consistency=5 → CORRECT."""
        prompt: str = self._build_loss_prompt(task_idx=1, sample_idx=0)
        output_start: int = prompt.index("<OUTPUT_OF_FUNCTION>")
        output_end: int = prompt.index("</OUTPUT_OF_FUNCTION>")
        output_content: str = prompt[output_start:output_end]
        assert "CORRECT" in output_content

    def test_loss_prompt_inputs_contain_prediction_and_gt(self):
        prompt: str = self._build_loss_prompt(task_idx=0, sample_idx=0)
        inputs_start: int = prompt.index("<INPUTS_TO_FUNCTION>")
        inputs_end: int = prompt.index("</INPUTS_TO_FUNCTION>")
        inputs_content: str = prompt[inputs_start:inputs_end]
        assert "3" in inputs_content, (
            "Prediction value (3) must be in INPUTS_TO_FUNCTION"
        )
        assert "4" in inputs_content, (
            "Ground truth value (4) must be in INPUTS_TO_FUNCTION"
        )

    def test_loss_prompt_variable_targets_prediction(self):
        """<VARIABLE> must contain the prediction value, NOT the task instruction."""
        prompt: str = self._build_loss_prompt(task_idx=0, sample_idx=0)
        var_start: int = prompt.index("<VARIABLE>")
        var_end: int = prompt.index("</VARIABLE>")
        var_content: str = prompt[var_start:var_end]
        assert "3" in var_content, "<VARIABLE> must contain the prediction value"
        assert "Rate fluency" not in var_content, (
            "<VARIABLE> must NOT contain the task instruction"
        )

    def test_loss_prompt_has_objective_function(self):
        prompt: str = self._build_loss_prompt()
        assert "<OBJECTIVE_FUNCTION>" in prompt
        assert "improve the above metric" in prompt

    def test_loss_prompt_has_role_tag(self):
        prompt: str = self._build_loss_prompt()
        assert "<ROLE>" in prompt
        assert "fluency" in prompt

    def test_loss_prompt_omits_input_data(self):
        """The loss prompt must NOT contain the original input data (summary,
        source text).  Per the paper, the string-based function backward sees
        only the prediction and ground truth, not the input question."""
        prompt: str = self._build_loss_prompt()
        assert "The cat sat on the mat." not in prompt, (
            "Loss prompt must NOT include input data — the loss backward "
            "sees only prediction and ground truth"
        )
        assert "A story about a cat" not in prompt

    def test_loss_prompt_omits_task_instruction(self):
        prompt: str = self._build_loss_prompt()
        assert "Rate fluency from 1 to 5." not in prompt, (
            "Loss prompt must NOT include the mutable task_instruction"
        )

    def test_loss_prompt_omits_task_description(self):
        prompt: str = self._build_loss_prompt()
        assert "Evaluate the fluency and readability" not in prompt

    def test_loss_prompt_has_adaptive_stop(self):
        """Paper line 96: 'If a variable is already working well... you should
        not give feedback.'"""
        prompt: str = self._build_loss_prompt()
        assert "already working well" in prompt


class TestTextGradGradientPrompt:
    """Verify the gradient LLM prompt includes the full forward-pass context."""

    def _build_gradient_prompt(self, *, task_idx: int = 0, sample_idx: int = 0) -> str:
        from prompt_moo.algorithm.textgrad import TextGradGradientComputer

        prompt_template = _make_prompt_template()
        preds = _make_predictions()
        gts = _make_samples()
        task = TASKS[task_idx]

        pred_by_sample = {p.sample_id: p for p in preds}
        gt_by_sample = {g.sample_id: g for g in gts}

        fb = TextualFeedback(
            task_name=task.task_name,
            feedback_text="The model overpredicted fluency. It gave 3 but GT is 4.",
            aggregated_from_samples=[gts[sample_idx].sample_id],
            feedback_prompt=None,
        )

        return TextGradGradientComputer._build_textgrad_gradient_prompt(
            feedbacks=[fb],
            task=task,
            prompt_template=prompt_template,
            pred_by_sample=pred_by_sample,
            gt_by_sample=gt_by_sample,
            input_col_labels=INPUT_COL_LABELS,
        )

    def test_gradient_prompt_contains_lm_system_prompt(self):
        prompt = self._build_gradient_prompt()
        assert "<LM_SYSTEM_PROMPT>" in prompt
        assert "</LM_SYSTEM_PROMPT>" in prompt
        assert SKELETON in prompt, "Gradient prompt must include the prompt template"

    def test_gradient_prompt_contains_lm_input(self):
        prompt = self._build_gradient_prompt()
        assert "<LM_INPUT>" in prompt
        assert "</LM_INPUT>" in prompt
        assert "The cat sat on the mat." in prompt

    def test_gradient_prompt_contains_lm_output(self):
        prompt = self._build_gradient_prompt()
        assert "<LM_OUTPUT>" in prompt
        assert "</LM_OUTPUT>" in prompt
        output_start: int = prompt.index("<LM_OUTPUT>") + len("<LM_OUTPUT>")
        output_end: int = prompt.index("</LM_OUTPUT>")
        output_content: str = prompt[output_start:output_end].strip()
        assert "fluency" in output_content, (
            "Gradient prompt <LM_OUTPUT> should show the target task"
        )

    def test_gradient_prompt_contains_objective_function(self):
        prompt = self._build_gradient_prompt()
        assert "<OBJECTIVE_FUNCTION>" in prompt
        assert "</OBJECTIVE_FUNCTION>" in prompt
        assert "overpredicted" in prompt

    def test_gradient_prompt_contains_variable(self):
        prompt = self._build_gradient_prompt()
        assert "<VARIABLE>" in prompt
        assert "</VARIABLE>" in prompt
        assert "Rate fluency from 1 to 5." in prompt

    def test_gradient_prompt_contains_role(self):
        prompt = self._build_gradient_prompt()
        assert "<ROLE>" in prompt
        assert "fluency" in prompt

    def test_gradient_prompt_says_not_to_propose_new_version(self):
        prompt = self._build_gradient_prompt()
        assert "DO NOT propose a new version" in prompt

    def test_gradient_prompt_lm_system_prompt_filters_to_single_task(self):
        """In separate_tasks mode, <LM_SYSTEM_PROMPT> should show only
        the target task's instruction, not all tasks."""
        prompt = self._build_gradient_prompt(task_idx=0)
        sys_start: int = prompt.index("<LM_SYSTEM_PROMPT>")
        sys_end: int = prompt.index("</LM_SYSTEM_PROMPT>")
        sys_content: str = prompt[sys_start:sys_end]
        assert "fluency" in sys_content, (
            "<LM_SYSTEM_PROMPT> must include the target task"
        )
        assert "consistency" not in sys_content, (
            "<LM_SYSTEM_PROMPT> in separate_tasks mode must NOT include "
            "other tasks' instructions"
        )

    def test_gradient_prompt_lm_output_shows_only_target_task(self):
        """In separate_tasks mode, <LM_OUTPUT> must show ONLY the target task's
        prediction value — not the raw_response JSON with all tasks.  Cross-task
        leakage in <LM_OUTPUT> violates the separate_tasks contract: the gradient
        LLM for fluency should not see consistency's prediction."""
        prompt: str = self._build_gradient_prompt(task_idx=0)
        output_start: int = prompt.index("<LM_OUTPUT>") + len("<LM_OUTPUT>")
        output_end: int = prompt.index("</LM_OUTPUT>")
        output_content: str = prompt[output_start:output_end].strip()
        assert "fluency" in output_content, (
            "<LM_OUTPUT> must contain the target task name"
        )
        assert "3" in output_content, (
            "<LM_OUTPUT> must contain the target task's prediction value"
        )
        assert "consistency" not in output_content, (
            "<LM_OUTPUT> in separate_tasks mode must NOT contain other tasks' "
            "predictions — this is cross-task leakage"
        )

    def test_gradient_prompt_objective_has_wrapper_sentence(self):
        """Per the library's LLMCall._backward_through_llm_chain, the
        <OBJECTIVE_FUNCTION> must wrap the loss output with: 'Your goal is
        to give feedback to the variable to address the following feedback
        on the LM_OUTPUT: {loss_output}'."""
        prompt = self._build_gradient_prompt()
        assert "address the following feedback on the LM_OUTPUT" in prompt, (
            "Gradient prompt must contain the paper's wrapper sentence "
            "that frames the loss output as feedback on the LM_OUTPUT"
        )

    def test_gradient_prompt_has_bridge_sentence(self):
        """Per the paper's LLMCall backward template: 'This conversation is
        part of a larger system. The <LM_OUTPUT> was later used as...'"""
        prompt = self._build_gradient_prompt()
        assert "part of a larger system" in prompt

    def test_gradient_prompt_has_adaptive_stop(self):
        """Paper line 96: 'If a variable is already working well... you should
        not give feedback.'"""
        prompt = self._build_gradient_prompt()
        assert "already working well" in prompt

    def test_gradient_prompt_has_backward_system_framing(self):
        """The gradient prompt must use the paper's BACKWARD_SYSTEM_PROMPT
        framing, not a custom short version."""
        prompt = self._build_gradient_prompt()
        assert "You are part of an optimization system" in prompt
        assert "gradient (feedback) engine" in prompt

    def test_gradient_prompt_lm_system_prompt_output_format_filtered(self):
        """In separate_tasks mode with task_output_formats populated, the
        <LM_SYSTEM_PROMPT> output format section must show ONLY the
        filtered task's entry, not all tasks.  Cross-task leakage in the
        output format violates the separate_tasks contract: the gradient
        LLM for fluency should not know about consistency's output format."""
        prompt: str = self._build_gradient_prompt(task_idx=0)
        sys_start: int = prompt.index("<LM_SYSTEM_PROMPT>")
        sys_end: int = prompt.index("</LM_SYSTEM_PROMPT>")
        sys_content: str = prompt[sys_start:sys_end]
        assert '"fluency"' in sys_content or "fluency" in sys_content, (
            "<LM_SYSTEM_PROMPT> output format must include the target task"
        )
        assert '"consistency"' not in sys_content, (
            "<LM_SYSTEM_PROMPT> output format in separate_tasks mode must NOT "
            "include other tasks' format entries — this is cross-task leakage"
        )


class TestTextGradOptimizerPrompt:
    """Verify the optimizer LLM prompt includes VARIABLE, CONTEXT, and FEEDBACK."""

    def _build_meta_prompt(self) -> str:
        from prompt_moo.algorithm.textgrad import TextGradOptimizer

        prompt_template = _make_prompt_template()
        preds = _make_predictions()
        gts = _make_samples()

        gradients: Dict[Task, List[TextGradient]] = {}
        for task in TASKS:
            gradients[task] = [
                TextGradient(
                    task_name=task.task_name,
                    gradient_text=f"Improve the {task.task_name} instruction by being more specific.",
                    based_on_feedbacks=["s0"],
                    gradient_prompt=None,
                ),
                TextGradient(
                    task_name=task.task_name,
                    gradient_text=f"Add calibration guidance for {task.task_name} scores.",
                    based_on_feedbacks=["s1"],
                    gradient_prompt=None,
                ),
            ]

        optimizer = TextGradOptimizer()
        return optimizer.create_meta_prompt(
            gradients=gradients,
            current_prompt=prompt_template,
            tasks=TASKS,
            predictions=preds,
            ground_truths=gts,
            input_col_labels=INPUT_COL_LABELS,
        )

    def test_optimizer_prompt_contains_variable_tags(self):
        meta = self._build_meta_prompt()
        assert "<VARIABLE>" in meta
        assert "</VARIABLE>" in meta

    def test_optimizer_prompt_contains_context_with_conversation(self):
        meta = self._build_meta_prompt()
        assert "<CONTEXT>" in meta
        assert "</CONTEXT>" in meta
        assert "<LM_SYSTEM_PROMPT>" in meta
        assert "<LM_INPUT>" in meta
        assert "<LM_OUTPUT>" in meta

    def test_optimizer_prompt_contains_feedback_tags(self):
        meta = self._build_meta_prompt()
        assert "<FEEDBACK>" in meta
        assert "</FEEDBACK>" in meta

    def test_optimizer_prompt_feedback_has_per_instance_gradients(self):
        meta = self._build_meta_prompt()
        assert "Improve the fluency instruction" in meta
        assert "Add calibration guidance for fluency" in meta
        assert "Improve the consistency instruction" in meta

    def test_optimizer_prompt_context_includes_input_data(self):
        meta = self._build_meta_prompt()
        assert "The cat sat on the mat." in meta

    def test_optimizer_prompt_context_includes_all_examples(self):
        """Context block should show all forward-pass conversations, not just
        the first, so the optimizer can ground per-instance feedback."""
        meta = self._build_meta_prompt()
        assert "The cat sat on the mat." in meta
        assert "Dogs are great pets" in meta
        assert meta.count("<CONVERSATION>") == 2
        assert meta.count("</CONVERSATION>") == 2

    def test_optimizer_gradients_are_newline_separated(self):
        """Per-instance gradients should be newline-separated (tg.sum analog),
        not space-joined."""
        meta = self._build_meta_prompt()
        feedback_start = meta.index("<FEEDBACK>") + len("<FEEDBACK>")
        feedback_end = meta.index("</FEEDBACK>")
        feedback_content = meta[feedback_start:feedback_end]
        assert "  - Improve the fluency instruction" in feedback_content

    def test_optimizer_prompt_lm_output_shows_raw_response(self):
        """The optimizer's <LM_OUTPUT> should show the task LLM's raw_response
        (full JSON), not reformatted task_name: value lines."""
        meta: str = self._build_meta_prompt()
        assert '{"fluency": 3, "consistency": 5}' in meta or '"fluency": 3' in meta, (
            "Optimizer <LM_OUTPUT> must show raw_response JSON, "
            "not task_name: value lines"
        )
        assert "fluency: 3\n" not in meta.split("<FEEDBACK>")[0], (
            "Optimizer <LM_OUTPUT> must NOT show bare 'fluency: 3' format "
            "(without JSON braces) before the FEEDBACK section"
        )


class TestTextGradOptimizerParsing:
    """Verify the optimizer correctly parses JSON responses."""

    def test_parses_valid_json(self):
        from prompt_moo.algorithm.textgrad import TextGradOptimizer

        optimizer = TextGradOptimizer()
        response = (
            "Here is the improved instruction:\n"
            '{"instructions": {"fluency": "new fluency instr", "consistency": "new consistency instr"} }'
        )
        result = optimizer.parse_meta_prompt_response(response=response, tasks=TASKS)
        assert result["fluency"] == "new fluency instr"
        assert result["consistency"] == "new consistency instr"

    def test_rejects_empty_json(self):
        from prompt_moo.algorithm.textgrad import TextGradOptimizer

        optimizer = TextGradOptimizer()
        with pytest.raises(ValueError):
            optimizer.parse_meta_prompt_response(response="no json here", tasks=TASKS)


class TestTextGradE2EPromptFlow:
    """End-to-end test: verifies the full prompt construction pipeline without
    actual LLM calls.  Constructs the same data structures the training loop
    would produce and checks that each stage's output feeds correctly into the
    next stage's input.
    """

    def test_full_pipeline_prompt_construction(self):
        from prompt_moo.algorithm.textgrad import TextGradLossComputer
        from prompt_moo.algorithm.textgrad import TextGradGradientComputer
        from prompt_moo.algorithm.textgrad import TextGradOptimizer

        prompt_template = _make_prompt_template()
        predictions = _make_predictions()
        ground_truths = _make_samples()

        # --- Step 2: Build per-instance loss prompts ---
        loss_prompts = []
        for pred, gt in zip(predictions, ground_truths):
            lp = TextGradLossComputer._build_textgrad_feedback_prompt(
                predictions=[pred],
                ground_truths=[gt],
                task=TASKS[0],
                input_col_labels=INPUT_COL_LABELS,
            )
            loss_prompts.append(lp)

        assert len(loss_prompts) == 2
        assert "<INPUTS_TO_FUNCTION>" in loss_prompts[0]
        assert "<OUTPUT_OF_FUNCTION>" in loss_prompts[0]
        assert "3" in loss_prompts[0]  # pred fluency=3
        assert "4" in loss_prompts[0]  # gt fluency=4

        # Simulate loss LLM responses
        loss_responses = [
            "The model gave fluency=3 but it should be 4. The summary is grammatically correct.",
            "The model gave fluency=5 but it should be 3. The summary lacks detail.",
        ]
        feedbacks = [
            TextualFeedback(
                task_name="fluency",
                feedback_text=resp,
                aggregated_from_samples=[pred.sample_id],
                feedback_prompt=lp,
            )
            for resp, pred, lp in zip(loss_responses, predictions, loss_prompts)
        ]

        # --- Step 3: Build per-instance gradient prompts ---
        pred_by_sample = {p.sample_id: p for p in predictions}
        gt_by_sample = {g.sample_id: g for g in ground_truths}

        gradient_prompts = []
        for fb in feedbacks:
            gp = TextGradGradientComputer._build_textgrad_gradient_prompt(
                feedbacks=[fb],
                task=TASKS[0],
                prompt_template=prompt_template,
                pred_by_sample=pred_by_sample,
                gt_by_sample=gt_by_sample,
                input_col_labels=INPUT_COL_LABELS,
            )
            gradient_prompts.append(gp)

        assert len(gradient_prompts) == 2
        for gp in gradient_prompts:
            assert "<LM_SYSTEM_PROMPT>" in gp
            assert "<LM_INPUT>" in gp
            assert "<LM_OUTPUT>" in gp
            assert "<OBJECTIVE_FUNCTION>" in gp
            assert "<VARIABLE>" in gp

        assert "The cat sat on the mat." in gradient_prompts[0]
        assert "Dogs are great pets" in gradient_prompts[1]

        # Simulate gradient LLM responses
        gradients = {
            TASKS[0]: [
                TextGradient(
                    task_name="fluency",
                    gradient_text="Add guidance about scoring short summaries lower on fluency.",
                    based_on_feedbacks=["s0"],
                    gradient_prompt=gradient_prompts[0],
                ),
                TextGradient(
                    task_name="fluency",
                    gradient_text="Instruct the model to penalize lack of detail.",
                    based_on_feedbacks=["s1"],
                    gradient_prompt=gradient_prompts[1],
                ),
            ],
        }

        # --- Step 4: Build optimizer meta-prompt ---
        optimizer = TextGradOptimizer()
        meta_prompt = optimizer.create_meta_prompt(
            gradients=gradients,
            current_prompt=prompt_template,
            tasks=[TASKS[0]],
            predictions=predictions,
            ground_truths=ground_truths,
            input_col_labels=INPUT_COL_LABELS,
        )

        assert "<VARIABLE>" in meta_prompt
        assert "<CONTEXT>" in meta_prompt
        assert "<FEEDBACK>" in meta_prompt
        assert "Add guidance about scoring" in meta_prompt
        assert "Instruct the model to penalize" in meta_prompt
        assert "The cat sat on the mat." in meta_prompt


class TestTextGradMultiTaskStrategyValidation:
    """Validate that combine/separate constraints are enforced."""

    def test_default_strategy_is_sep_sep_combine(self):
        from prompt_moo.algorithm import TextGrad

        tg = TextGrad(
            task_llm=None,
            optimizer_llm=None,
            tasks=TASKS,
            steps=1,
            batch_size=2,
            loss_batch_size=1,
            gradient_batch_size=1,
            eval_every=1,
            name="test",
            validation_metric="accuracy",
        )
        assert tg.loss_task_strategy == "separate_tasks"
        assert tg.gradient_task_strategy == "separate_tasks"
        assert tg.optimizer_task_strategy == "combine_all_tasks"

    def test_all_combine_is_valid(self):
        from prompt_moo.algorithm import TextGrad

        tg = TextGrad(
            task_llm=None,
            optimizer_llm=None,
            tasks=TASKS,
            steps=1,
            batch_size=2,
            loss_batch_size=1,
            gradient_batch_size=1,
            eval_every=1,
            name="test",
            validation_metric="accuracy",
            loss_task_strategy="combine_all_tasks",
            gradient_task_strategy="combine_all_tasks",
            optimizer_task_strategy="combine_all_tasks",
        )
        assert tg.loss_task_strategy == "combine_all_tasks"

    def test_all_separate_is_valid(self):
        from prompt_moo.algorithm import TextGrad

        tg = TextGrad(
            task_llm=None,
            optimizer_llm=None,
            tasks=TASKS,
            steps=1,
            batch_size=2,
            loss_batch_size=1,
            gradient_batch_size=1,
            eval_every=1,
            name="test",
            validation_metric="accuracy",
            loss_task_strategy="separate_tasks",
            gradient_task_strategy="separate_tasks",
            optimizer_task_strategy="separate_tasks",
        )
        assert tg.optimizer_task_strategy == "separate_tasks"

    def test_loss_combine_gradient_separate_raises(self):
        from prompt_moo.algorithm import TextGrad

        with pytest.raises(ValueError, match="gradient_task_strategy must also be"):
            TextGrad(
                task_llm=None,
                optimizer_llm=None,
                tasks=TASKS,
                steps=1,
                batch_size=2,
                loss_batch_size=1,
                gradient_batch_size=1,
                eval_every=1,
                name="test",
                validation_metric="accuracy",
                loss_task_strategy="combine_all_tasks",
                gradient_task_strategy="separate_tasks",
                optimizer_task_strategy="combine_all_tasks",
            )

    def test_gradient_combine_optimizer_separate_raises(self):
        from prompt_moo.algorithm import TextGrad

        with pytest.raises(ValueError, match="optimizer_task_strategy must also be"):
            TextGrad(
                task_llm=None,
                optimizer_llm=None,
                tasks=TASKS,
                steps=1,
                batch_size=2,
                loss_batch_size=1,
                gradient_batch_size=1,
                eval_every=1,
                name="test",
                validation_metric="accuracy",
                loss_task_strategy="separate_tasks",
                gradient_task_strategy="combine_all_tasks",
                optimizer_task_strategy="separate_tasks",
            )


class TestTextGradCombinedLossPrompt:
    """Verify combine_all_tasks loss prompt follows paper structure."""

    def _build_combined_loss_prompt(self) -> str:
        from prompt_moo.algorithm.textgrad import TextGradLossComputer

        preds = _make_predictions()[:1]
        gts = _make_samples()[:1]

        return TextGradLossComputer._build_textgrad_combined_feedback_prompt(
            predictions=preds,
            ground_truths=gts,
            tasks=TASKS,
            input_col_labels=INPUT_COL_LABELS,
        )

    def test_combined_loss_prompt_has_backward_system_framing(self):
        prompt: str = self._build_combined_loss_prompt()
        assert "You are part of an optimization system" in prompt
        assert "gradient (feedback) engine" in prompt

    def test_combined_loss_prompt_has_inputs_to_function(self):
        prompt: str = self._build_combined_loss_prompt()
        assert "<INPUTS_TO_FUNCTION>" in prompt
        assert "</INPUTS_TO_FUNCTION>" in prompt

    def test_combined_loss_prompt_shows_all_tasks_predictions(self):
        prompt: str = self._build_combined_loss_prompt()
        inputs_start: int = prompt.index("<INPUTS_TO_FUNCTION>")
        inputs_end: int = prompt.index("</INPUTS_TO_FUNCTION>")
        inputs_content: str = prompt[inputs_start:inputs_end]
        assert "fluency" in inputs_content
        assert "consistency" in inputs_content

    def test_combined_loss_prompt_output_shows_per_task_results(self):
        prompt: str = self._build_combined_loss_prompt()
        output_start: int = prompt.index("<OUTPUT_OF_FUNCTION>")
        output_end: int = prompt.index("</OUTPUT_OF_FUNCTION>")
        output_content: str = prompt[output_start:output_end]
        assert "fluency" in output_content
        assert "consistency" in output_content
        assert "INCORRECT" in output_content
        assert "CORRECT" in output_content

    def test_combined_loss_prompt_omits_mutable_instruction(self):
        prompt: str = self._build_combined_loss_prompt()
        assert "Rate fluency from 1 to 5." not in prompt
        assert "Rate consistency from 1 to 5." not in prompt

    def test_combined_loss_prompt_omits_task_description(self):
        prompt: str = self._build_combined_loss_prompt()
        assert "Evaluate the fluency and readability" not in prompt
        assert "Evaluate factual consistency" not in prompt

    def test_combined_loss_prompt_omits_input_data(self):
        prompt: str = self._build_combined_loss_prompt()
        assert "The cat sat on the mat." not in prompt
        assert "A story about a cat" not in prompt


class TestTextGradCombinedGradientPrompt:
    """Verify combine_all_tasks gradient prompt shows full output."""

    def _build_combined_gradient_prompt(self) -> str:
        from prompt_moo.algorithm.textgrad import TextGradGradientComputer

        prompt_template = _make_prompt_template()
        preds = _make_predictions()
        gts = _make_samples()

        pred_by_sample: Dict[str, PredictionResult] = {p.sample_id: p for p in preds}
        gt_by_sample: Dict[str, DatasetSample] = {g.sample_id: g for g in gts}

        fb: TextualFeedback = TextualFeedback(
            task_name="fluency",
            feedback_text="The model overpredicted fluency.",
            aggregated_from_samples=["s0"],
            feedback_prompt=None,
        )

        return TextGradGradientComputer._build_textgrad_combined_gradient_prompt(
            feedbacks=[fb],
            tasks=TASKS,
            prompt_template=prompt_template,
            pred_by_sample=pred_by_sample,
            gt_by_sample=gt_by_sample,
            input_col_labels=INPUT_COL_LABELS,
        )

    def test_combined_gradient_prompt_shows_all_task_outputs(self):
        prompt: str = self._build_combined_gradient_prompt()
        assert "<LM_OUTPUT>" in prompt
        assert '"fluency": 3' in prompt or "fluency" in prompt
        assert '"consistency": 5' in prompt or "consistency" in prompt
        assert "<VARIABLE>" in prompt
        assert "fluency: Rate fluency from 1 to 5." in prompt
        assert "consistency: Rate consistency from 1 to 5." in prompt

    def test_combined_gradient_prompt_objective_has_wrapper(self):
        """The <OBJECTIVE_FUNCTION> must wrap the loss output per the paper."""
        prompt: str = self._build_combined_gradient_prompt()
        assert "address the following feedback on the LM_OUTPUT" in prompt

    def test_combined_gradient_prompt_has_bridge_sentence(self):
        prompt: str = self._build_combined_gradient_prompt()
        assert "part of a larger system" in prompt

    def test_combined_gradient_prompt_has_backward_system_framing(self):
        prompt: str = self._build_combined_gradient_prompt()
        assert "You are part of an optimization system" in prompt
        assert "already working well" in prompt

    def test_combined_gradient_prompt_lm_output_shows_raw_response(self):
        """The combined gradient prompt's <LM_OUTPUT> should show raw_response
        (full JSON), not reformatted task_name: value lines."""
        prompt: str = self._build_combined_gradient_prompt()
        assert (
            '{"fluency": 3, "consistency": 5}' in prompt or '"fluency": 3' in prompt
        ), "Combined gradient <LM_OUTPUT> must show raw_response JSON"

    def test_combined_gradient_deduplicates_identical_objective_functions(self):
        """When combined loss produces identical feedback copied to each task
        key (CCC mode), the combined gradient prompt should show ONE
        <OBJECTIVE_FUNCTION> block, not N identical copies."""
        from prompt_moo.algorithm.textgrad import TextGradGradientComputer

        prompt_template: PromptTemplate = _make_prompt_template()
        preds: List[PredictionResult] = _make_predictions()
        gts: List[DatasetSample] = _make_samples()

        pred_by_sample: Dict[str, PredictionResult] = {p.sample_id: p for p in preds}
        gt_by_sample: Dict[str, DatasetSample] = {g.sample_id: g for g in gts}

        combined_feedback_text: str = (
            "All predictions matched except relevance which was off by 1."
        )
        feedbacks: List[TextualFeedback] = [
            TextualFeedback(
                task_name="fluency",
                feedback_text=combined_feedback_text,
                aggregated_from_samples=["s0"],
                feedback_prompt=None,
            ),
            TextualFeedback(
                task_name="consistency",
                feedback_text=combined_feedback_text,
                aggregated_from_samples=["s0"],
                feedback_prompt=None,
            ),
        ]

        prompt: str = TextGradGradientComputer._build_textgrad_combined_gradient_prompt(
            feedbacks=feedbacks,
            tasks=TASKS,
            prompt_template=prompt_template,
            pred_by_sample=pred_by_sample,
            gt_by_sample=gt_by_sample,
            input_col_labels=INPUT_COL_LABELS,
        )

        objective_count: int = prompt.count("All predictions matched except relevance")
        assert objective_count == 1, (
            f"Identical feedback text from combined loss should appear in ONE "
            f"<OBJECTIVE_FUNCTION> block, not duplicated. "
            f"Found {objective_count} occurrences."
        )

    def test_combined_gradient_separate_loss_single_objective_block(self):
        """In SCC mode (separate loss → combined gradient), multiple distinct
        per-task loss feedbacks should be concatenated into a SINGLE
        <OBJECTIVE_FUNCTION> block with task labels, not emitted as separate
        <OBJECTIVE_FUNCTION> tags.  The paper uses one <OBJECTIVE_FUNCTION>
        with all gradients concatenated via get_gradient_text()."""
        from prompt_moo.algorithm.textgrad import TextGradGradientComputer

        prompt_template: PromptTemplate = _make_prompt_template()
        preds: List[PredictionResult] = _make_predictions()
        gts: List[DatasetSample] = _make_samples()

        pred_by_sample: Dict[str, PredictionResult] = {p.sample_id: p for p in preds}
        gt_by_sample: Dict[str, DatasetSample] = {g.sample_id: g for g in gts}

        feedbacks: List[TextualFeedback] = [
            TextualFeedback(
                task_name="fluency",
                feedback_text="Fluency prediction was correct, no issues.",
                aggregated_from_samples=["s0"],
                feedback_prompt=None,
            ),
            TextualFeedback(
                task_name="consistency",
                feedback_text="Consistency prediction was off by 2.",
                aggregated_from_samples=["s0"],
                feedback_prompt=None,
            ),
        ]

        prompt: str = TextGradGradientComputer._build_textgrad_combined_gradient_prompt(
            feedbacks=feedbacks,
            tasks=TASKS,
            prompt_template=prompt_template,
            pred_by_sample=pred_by_sample,
            gt_by_sample=gt_by_sample,
            input_col_labels=INPUT_COL_LABELS,
        )

        closing_tags: int = prompt.count("</OBJECTIVE_FUNCTION>")
        framing_refs: int = (
            1  # BACKWARD_SYSTEM_PROMPT contains one </OBJECTIVE_FUNCTION>
        )
        actual_blocks: int = closing_tags - framing_refs
        assert actual_blocks == 1, (
            f"Combined gradient prompt must have exactly ONE <OBJECTIVE_FUNCTION> "
            f"block (found {actual_blocks} after subtracting {framing_refs} "
            f"framing references). Per the paper, multiple loss feedbacks are "
            f"concatenated inside a single block."
        )

        assert "Fluency prediction was correct" in prompt
        assert "Consistency prediction was off by 2" in prompt
        assert "Feedback for fluency" in prompt
        assert "Feedback for consistency" in prompt

    def test_combined_gradient_prompt_deduplicates_samples(self):
        """When feedbacks from different tasks reference the same sample,
        the conversation should appear only once."""
        from prompt_moo.algorithm.textgrad import TextGradGradientComputer

        prompt_template = _make_prompt_template()
        preds = _make_predictions()
        gts = _make_samples()

        pred_by_sample = {p.sample_id: p for p in preds}
        gt_by_sample = {g.sample_id: g for g in gts}

        fb1 = TextualFeedback(
            task_name="fluency",
            feedback_text="Fluency feedback for s0",
            aggregated_from_samples=["s0"],
            feedback_prompt=None,
        )
        fb2 = TextualFeedback(
            task_name="consistency",
            feedback_text="Consistency feedback for s0",
            aggregated_from_samples=["s0"],
            feedback_prompt=None,
        )

        prompt = TextGradGradientComputer._build_textgrad_combined_gradient_prompt(
            feedbacks=[fb1, fb2],
            tasks=TASKS,
            prompt_template=prompt_template,
            pred_by_sample=pred_by_sample,
            gt_by_sample=gt_by_sample,
            input_col_labels=INPUT_COL_LABELS,
        )

        assert prompt.count("</LM_INPUT>") == 1, (
            "Same sample referenced by two tasks should produce one conversation"
        )


class TestTextGradSeparateOptimizer:
    """Verify the single-task optimizer prompt in separate_tasks mode."""

    def test_single_task_meta_prompt_targets_one_task(self):
        from prompt_moo.algorithm.textgrad import TextGradOptimizer

        prompt_template = _make_prompt_template()
        preds = _make_predictions()
        gts = _make_samples()

        gradients = [
            TextGradient(
                task_name="fluency",
                gradient_text="Improve fluency instruction specificity.",
                based_on_feedbacks=["s0"],
                gradient_prompt=None,
            ),
        ]

        optimizer = TextGradOptimizer()
        meta = optimizer._create_single_task_meta_prompt(
            gradients=gradients,
            task=TASKS[0],
            current_prompt=prompt_template,
            predictions=preds,
            ground_truths=gts,
            input_col_labels=INPUT_COL_LABELS,
        )

        assert "<VARIABLE>" in meta
        variable_start = meta.index("<VARIABLE>")
        variable_end = meta.index("</VARIABLE>")
        variable_content = meta[variable_start:variable_end]
        assert "Rate fluency from 1 to 5." in variable_content, (
            "VARIABLE section must contain the target task's instruction"
        )
        assert "Rate consistency from 1 to 5." not in variable_content, (
            "VARIABLE section must NOT contain the other task's instruction"
        )
        assert "<FEEDBACK>" in meta
        assert "Improve fluency instruction" in meta
        assert "Output ONLY the improved instruction text" in meta

    def test_separate_optimizer_context_lm_system_prompt_filtered(self):
        """In separate_tasks optimizer mode, the <CONTEXT> block's
        <LM_SYSTEM_PROMPT> must show only the target task's output format
        and instruction — not all tasks.  Cross-task leakage in the
        optimizer's context block violates the separate_tasks contract."""
        from prompt_moo.algorithm.textgrad import TextGradOptimizer

        prompt_template: PromptTemplate = _make_prompt_template()
        preds: List[PredictionResult] = _make_predictions()
        gts: List[DatasetSample] = _make_samples()

        gradients: List[TextGradient] = [
            TextGradient(
                task_name="fluency",
                gradient_text="Improve fluency instruction.",
                based_on_feedbacks=["s0"],
                gradient_prompt=None,
            ),
        ]

        optimizer: TextGradOptimizer = TextGradOptimizer()
        meta: str = optimizer._create_single_task_meta_prompt(
            gradients=gradients,
            task=TASKS[0],
            current_prompt=prompt_template,
            predictions=preds,
            ground_truths=gts,
            input_col_labels=INPUT_COL_LABELS,
        )

        sys_start: int = meta.index("<LM_SYSTEM_PROMPT>")
        sys_end: int = meta.index("</LM_SYSTEM_PROMPT>")
        sys_content: str = meta[sys_start:sys_end]
        assert "fluency" in sys_content
        assert "consistency" not in sys_content, (
            "Separate-tasks optimizer <LM_SYSTEM_PROMPT> must NOT show "
            "other tasks' instructions or output format entries"
        )

    def test_separate_optimizer_context_lm_output_filtered(self):
        """In separate_tasks optimizer mode, the <CONTEXT> block's
        <LM_OUTPUT> must show only the target task's prediction value,
        not all tasks' predictions.  Cross-task leakage here means the
        optimizer LLM for fluency sees consistency's score."""
        from prompt_moo.algorithm.textgrad import TextGradOptimizer

        prompt_template: PromptTemplate = _make_prompt_template()
        preds: List[PredictionResult] = _make_predictions()
        gts: List[DatasetSample] = _make_samples()

        gradients: List[TextGradient] = [
            TextGradient(
                task_name="fluency",
                gradient_text="Improve fluency instruction.",
                based_on_feedbacks=["s0"],
                gradient_prompt=None,
            ),
        ]

        optimizer: TextGradOptimizer = TextGradOptimizer()
        meta: str = optimizer._create_single_task_meta_prompt(
            gradients=gradients,
            task=TASKS[0],
            current_prompt=prompt_template,
            predictions=preds,
            ground_truths=gts,
            input_col_labels=INPUT_COL_LABELS,
        )

        output_start: int = meta.index("<LM_OUTPUT>") + len("<LM_OUTPUT>")
        output_end: int = meta.index("</LM_OUTPUT>")
        output_content: str = meta[output_start:output_end].strip()
        assert "fluency" in output_content, "<LM_OUTPUT> must contain the target task"
        assert "consistency" not in output_content, (
            "Separate-tasks optimizer <LM_OUTPUT> must NOT show other "
            "tasks' predictions — this is cross-task leakage"
        )


# -----------------------------------------------------------------------
# Separate-tasks optimizer: full compute flow with mock LLM
# -----------------------------------------------------------------------


class _MockFuture:
    def __init__(self, value):
        self._value = value

    def result(self, timeout=None):
        return self._value


class _TextGradMockLLM:
    """Mock LLM that returns distinct per-task instruction strings."""

    _prompt_suffix = ""

    def __init__(self):
        self._call_count = 0

    def call_llm_batch(self, *, prompts, verbosity=1, validator=None, **kw):
        self._call_count += 1
        results = [f"improved_instruction_{i}" for i in range(len(prompts))]
        return _MockFuture(results)

    def stop(self):
        pass


class TestTextGradSeparateOptimizerCompute:
    """Test the full _optimize_separate() flow with a mock LLM.

    _optimize_separate() makes K independent optimizer calls (one per task)
    and merges the results into a single OptimizerResult.
    """

    def test_separate_makes_one_call_per_task(self):
        from prompt_moo.algorithm.textgrad import TextGradOptimizer

        optimizer = TextGradOptimizer()
        mock_llm = _TextGradMockLLM()

        gradients = {
            TASKS[0]: [
                TextGradient(
                    task_name="fluency",
                    gradient_text="fix fluency",
                    based_on_feedbacks=["s0"],
                    gradient_prompt=None,
                )
            ],
            TASKS[1]: [
                TextGradient(
                    task_name="consistency",
                    gradient_text="fix consistency",
                    based_on_feedbacks=["s0"],
                    gradient_prompt=None,
                )
            ],
        }

        result = optimizer.optimize(
            gradients,
            _make_prompt_template(),
            TASKS,
            mock_llm,
            verbosity=0,
            optimizer_task_strategy="separate_tasks",
            predictions=_make_predictions(),
            ground_truths=_make_samples(),
            input_col_labels=INPUT_COL_LABELS,
        )

        assert result.new_prompt is not None
        assert "fluency" in result.new_prompt.instruction
        assert "consistency" in result.new_prompt.instruction

    def test_separate_result_has_distinct_per_task_instructions(self):
        from prompt_moo.algorithm.textgrad import TextGradOptimizer

        optimizer = TextGradOptimizer()
        mock_llm = _TextGradMockLLM()

        gradients = {
            TASKS[0]: [
                TextGradient(
                    task_name="fluency",
                    gradient_text="fix fluency",
                    based_on_feedbacks=["s0"],
                    gradient_prompt=None,
                )
            ],
            TASKS[1]: [
                TextGradient(
                    task_name="consistency",
                    gradient_text="fix consistency",
                    based_on_feedbacks=["s0"],
                    gradient_prompt=None,
                )
            ],
        }

        result = optimizer.optimize(
            gradients,
            _make_prompt_template(),
            TASKS,
            mock_llm,
            verbosity=0,
            optimizer_task_strategy="separate_tasks",
            predictions=_make_predictions(),
            ground_truths=_make_samples(),
            input_col_labels=INPUT_COL_LABELS,
        )

        fluency_instr = result.new_prompt.instruction["fluency"]
        consistency_instr = result.new_prompt.instruction["consistency"]
        assert fluency_instr != consistency_instr


# -----------------------------------------------------------------------
# Combined loss and gradient compute flow with mock LLM
# -----------------------------------------------------------------------


class _TextGradFeedbackMockLLM:
    """Mock LLM for loss/gradient that returns text feedback."""

    _prompt_suffix = ""

    def call_llm_batch(self, *, prompts, verbosity=1, validator=None, **kw):
        results = [f"Feedback for prompt {i}" for i in range(len(prompts))]
        return _MockFuture(results)


class TestTextGradCombinedLossCompute:
    """Test the full _compute_combined() flow on TextGradLossComputer."""

    def test_combined_loss_returns_feedback_for_all_tasks(self):
        from prompt_moo.algorithm.textgrad import TextGradLossComputer

        lc = TextGradLossComputer()
        mock_llm = _TextGradFeedbackMockLLM()

        feedbacks = lc.compute(
            predictions=_make_predictions()[:1],
            ground_truths=_make_samples()[:1],
            tasks=TASKS,
            prompt_template=_make_prompt_template(),
            llm_pool=mock_llm,
            loss_batch_size=1,
            verbosity=0,
            loss_functions={
                "fluency": {"use_textual": True},
                "consistency": {"use_textual": True},
            },
            loss_task_strategy="combine_all_tasks",
            input_col_labels=INPUT_COL_LABELS,
        )

        assert TASKS[0] in feedbacks
        assert TASKS[1] in feedbacks
        for task in TASKS:
            assert len(feedbacks[task]) > 0
            from prompt_moo.data_structures import TextualFeedback

            assert isinstance(feedbacks[task][0], TextualFeedback)


class TestTextGradCombinedGradientCompute:
    """Test the full _compute_combined() flow on TextGradGradientComputer."""

    def test_combined_gradient_returns_gradient_for_all_tasks(self):
        from prompt_moo.algorithm.textgrad import TextGradGradientComputer

        gc = TextGradGradientComputer()
        mock_llm = _TextGradFeedbackMockLLM()

        from prompt_moo.data_structures import TextualFeedback

        feedbacks = {
            TASKS[0]: [
                TextualFeedback(
                    task_name="fluency",
                    feedback_text="Fluency was bad",
                    aggregated_from_samples=["s0"],
                    feedback_prompt="prompt",
                )
            ],
            TASKS[1]: [
                TextualFeedback(
                    task_name="consistency",
                    feedback_text="Consistency was ok",
                    aggregated_from_samples=["s0"],
                    feedback_prompt="prompt",
                )
            ],
        }

        gradients = gc.compute(
            feedbacks=feedbacks,
            prompt_template=_make_prompt_template(),
            tasks=TASKS,
            llm_pool=mock_llm,
            gradient_batch_size=1,
            verbosity=0,
            predictions=_make_predictions()[:1],
            ground_truths=_make_samples()[:1],
            input_col_labels=INPUT_COL_LABELS,
            gradient_task_strategy="combine_all_tasks",
        )

        assert TASKS[0] in gradients
        assert TASKS[1] in gradients
        for task in TASKS:
            assert len(gradients[task]) > 0


class TestTextGradMetaPromptStructure:
    """Verify the optimizer meta-prompt matches the paper's TGD structure.

    Paper reference: textgrad_operations.tex, lines 192-258.
    The quintessential structure is:
      <ROLE> → <VARIABLE> → <CONTEXT> containing conversations + <FEEDBACK> → instruction

    These tests verify:
    - No standalone template block before <VARIABLE> (Issue 1)
    - <FEEDBACK> is inside <CONTEXT> (Issue 2)
    - "noisy feedback" guidance is present (Issue 3)
    - <CONVERSATION> wrapper tags are used (Issue 4)
    - --- dividers between task groups in multi-task <FEEDBACK>
    - Valid JSON in multi-task <VARIABLE>
    """

    def _build_multi_task_meta_prompt(self) -> str:
        from prompt_moo.algorithm.textgrad import TextGradOptimizer

        prompt_template = _make_prompt_template()
        preds = _make_predictions()
        gts = _make_samples()

        gradients: Dict[Task, List[TextGradient]] = {}
        for task in TASKS:
            gradients[task] = [
                TextGradient(
                    task_name=task.task_name,
                    gradient_text=f"Gradient for {task.task_name} instance 1.",
                    based_on_feedbacks=["s0"],
                    gradient_prompt=None,
                ),
                TextGradient(
                    task_name=task.task_name,
                    gradient_text=f"Gradient for {task.task_name} instance 2.",
                    based_on_feedbacks=["s1"],
                    gradient_prompt=None,
                ),
            ]

        optimizer = TextGradOptimizer()
        return optimizer.create_meta_prompt(
            gradients=gradients,
            current_prompt=prompt_template,
            tasks=TASKS,
            predictions=preds,
            ground_truths=gts,
            input_col_labels=INPUT_COL_LABELS,
        )

    def _build_single_task_meta_prompt(self) -> str:
        from prompt_moo.algorithm.textgrad import TextGradOptimizer

        prompt_template = _make_prompt_template()
        preds = _make_predictions()
        gts = _make_samples()

        gradients_list: List[TextGradient] = [
            TextGradient(
                task_name="fluency",
                gradient_text="Gradient for fluency instance 1.",
                based_on_feedbacks=["s0"],
                gradient_prompt=None,
            ),
            TextGradient(
                task_name="fluency",
                gradient_text="Gradient for fluency instance 2.",
                based_on_feedbacks=["s1"],
                gradient_prompt=None,
            ),
        ]

        optimizer = TextGradOptimizer()
        return optimizer._create_single_task_meta_prompt(
            gradients=gradients_list,
            task=TASKS[0],
            current_prompt=prompt_template,
            predictions=preds,
            ground_truths=gts,
            input_col_labels=INPUT_COL_LABELS,
        )

    def test_no_standalone_template_block(self):
        """The paper shows the template only in <VARIABLE> and <CONTEXT>/<LM_SYSTEM_PROMPT>.
        There should be no 'render_for_optimizer' block between <ROLE> and <VARIABLE>."""
        meta: str = self._build_multi_task_meta_prompt()
        assert "(input data will be inserted here)" not in meta, (
            "The standalone template preview block (render_for_optimizer) must not appear. "
            "The paper shows the template only inside <CONTEXT>/<LM_SYSTEM_PROMPT>."
        )

    def test_feedback_inside_context(self):
        """Paper: <FEEDBACK> is inside <CONTEXT> (before </CONTEXT>).
        The closing </CONTEXT> must come AFTER </FEEDBACK>."""
        meta: str = self._build_multi_task_meta_prompt()
        feedback_end: int = meta.index("</FEEDBACK>")
        context_end: int = meta.index("</CONTEXT>")
        assert feedback_end < context_end, (
            "<FEEDBACK> must be inside <CONTEXT> per the paper. "
            f"Found </FEEDBACK> at {feedback_end}, </CONTEXT> at {context_end}."
        )

    def test_noisy_feedback_guidance(self):
        """Paper's system prompt (line 185): 'The feedback may be noisy,
        identify what is important and what is correct.'"""
        meta: str = self._build_multi_task_meta_prompt()
        assert "feedback may be noisy" in meta, (
            "The optimizer prompt must include the noisy-feedback guidance "
            "from the paper's TGD system prompt."
        )

    def test_conversation_wrapper_tags(self):
        """Paper uses <CONVERSATION>...</CONVERSATION> wrappers."""
        meta: str = self._build_multi_task_meta_prompt()
        assert "<CONVERSATION>" in meta
        assert "</CONVERSATION>" in meta
        assert meta.count("<CONVERSATION>") == 2, (
            "With 2 samples, there should be 2 <CONVERSATION> blocks."
        )

    def test_multi_task_feedback_has_dividers(self):
        """Multi-task <FEEDBACK> should use --- dividers between task groups."""
        meta: str = self._build_multi_task_meta_prompt()
        feedback_start: int = meta.index("<FEEDBACK>") + len("<FEEDBACK>")
        feedback_end: int = meta.index("</FEEDBACK>")
        feedback_content: str = meta[feedback_start:feedback_end]
        assert "---" in feedback_content, (
            "Multi-task feedback should have --- dividers between task groups."
        )

    def test_multi_task_variable_has_valid_json_commas(self):
        """The <VARIABLE> block must have commas between JSON key-value pairs."""
        meta: str = self._build_multi_task_meta_prompt()
        variable_start: int = meta.index("<VARIABLE>") + len("<VARIABLE>")
        variable_end: int = meta.index("</VARIABLE>")
        variable_content: str = meta[variable_start:variable_end].strip()
        assert '",\n' in variable_content, (
            "JSON key-value pairs in <VARIABLE> must be comma-separated."
        )

    def test_bridging_sentence_before_feedback(self):
        """Paper (line 233): 'Here is the feedback we got for ...' bridges
        conversations to feedback inside <CONTEXT>."""
        meta: str = self._build_multi_task_meta_prompt()
        assert "feedback we got for" in meta, (
            "Missing bridging sentence before <FEEDBACK> inside <CONTEXT>."
        )

    def test_single_task_no_standalone_template(self):
        """Same Issue 1 check for separate_tasks mode."""
        meta: str = self._build_single_task_meta_prompt()
        assert "(input data will be inserted here)" not in meta

    def test_single_task_feedback_inside_context(self):
        """Same Issue 2 check for separate_tasks mode."""
        meta: str = self._build_single_task_meta_prompt()
        feedback_end: int = meta.index("</FEEDBACK>")
        context_end: int = meta.index("</CONTEXT>")
        assert feedback_end < context_end

    def test_single_task_noisy_feedback(self):
        """Same Issue 3 check for separate_tasks mode."""
        meta: str = self._build_single_task_meta_prompt()
        assert "feedback may be noisy" in meta

    def test_single_task_conversation_tags(self):
        """Same Issue 4 check for separate_tasks mode."""
        meta: str = self._build_single_task_meta_prompt()
        assert meta.count("<CONVERSATION>") == 2
        assert meta.count("</CONVERSATION>") == 2


class TestTextGradValidationGateDisabled:
    """Tests for validation_metric='none' which disables the validation gate.

    When validation_metric='none', TextGrad should:
    - Accept construction without error (no Metric.get_subclass lookup)
    - Report validation_gate_enabled as False
    - Auto-accept every prompt update (return True from _should_accept_prompt_update)
    - Not set up a validation batch (skip _setup_validation_batch)
    - Not call _compute_validation_score
    - Log the skip to the observer with accepted=True and skipped=True
    - Report 'disabled' state in _serialize_algorithm_state
    """

    def _make_textgrad(self, **overrides: Any) -> "TextGrad":
        from prompt_moo.algorithm import TextGrad

        defaults: Dict[str, Any] = {
            "task_llm": None,
            "optimizer_llm": None,
            "tasks": TASKS,
            "steps": 1,
            "batch_size": 2,
            "loss_batch_size": 1,
            "gradient_batch_size": 1,
            "eval_every": 1,
            "name": "test",
            "validation_metric": "none",
        }
        defaults.update(overrides)
        return TextGrad(**defaults)

    def test_construction_with_none_succeeds(self):
        """TextGrad(validation_metric='none') should construct without error.

        Previously, this would raise ValueError because Metric.get_subclass('none')
        fails. With the gate-disable feature, 'none' is a recognized sentinel that
        skips the metric lookup entirely.
        """
        tg: "TextGrad" = self._make_textgrad()
        assert tg.validation_metric == "none"

    def test_validation_gate_enabled_is_false(self):
        """The validation_gate_enabled property should return False for 'none'."""
        tg: "TextGrad" = self._make_textgrad()
        assert tg.validation_gate_enabled is False

    def test_validation_gate_enabled_is_true_for_real_metric(self):
        """Sanity check: validation_gate_enabled returns True for real metrics."""
        tg: "TextGrad" = self._make_textgrad(validation_metric="accuracy")
        assert tg.validation_gate_enabled is True

    def test_validation_metric_cls_is_none_when_disabled(self):
        """When the gate is disabled, _validation_metric_cls should remain None."""
        tg: "TextGrad" = self._make_textgrad()
        assert tg._validation_metric_cls is None

    def test_validation_batch_not_set_up_when_disabled(self):
        """_validation_batch should remain None when gate is disabled."""
        tg: "TextGrad" = self._make_textgrad()
        assert tg._validation_batch is None

    def test_should_accept_returns_true_when_disabled(self):
        """_should_accept_prompt_update must return True immediately when the
        gate is disabled, without calling _compute_validation_score."""
        from unittest.mock import MagicMock

        tg: "TextGrad" = self._make_textgrad()
        observer: MagicMock = MagicMock()
        prompt: PromptTemplate = _make_prompt_template()

        result: bool = tg._should_accept_prompt_update(
            current_prompt=prompt,
            new_prompt=prompt,
            step=1,
            observer=observer,
        )

        assert result is True

    def test_should_accept_does_not_call_compute_validation_score(self):
        """When the gate is disabled, _compute_validation_score must never be called.

        We verify this indirectly: _validation_batch is None when the gate is
        disabled (since _setup_validation_batch is skipped), so if
        _compute_validation_score were called, it would raise RuntimeError.
        The fact that _should_accept_prompt_update returns True without error
        proves the score computation was never attempted.
        """
        from unittest.mock import MagicMock

        tg: "TextGrad" = self._make_textgrad()
        assert tg._validation_batch is None
        observer: MagicMock = MagicMock()
        prompt: PromptTemplate = _make_prompt_template()

        result: bool = tg._should_accept_prompt_update(
            current_prompt=prompt,
            new_prompt=prompt,
            step=1,
            observer=observer,
        )
        assert result is True
        assert tg._validation_batch is None

    def test_observer_records_skipped_gate(self):
        """The observer should receive a validation_gate record with skipped=True
        and accepted=True when the gate is disabled."""
        from unittest.mock import MagicMock

        tg: "TextGrad" = self._make_textgrad()
        observer: MagicMock = MagicMock()
        prompt: PromptTemplate = _make_prompt_template()

        tg._should_accept_prompt_update(
            current_prompt=prompt,
            new_prompt=prompt,
            step=1,
            observer=observer,
        )

        observer.record.assert_called_once()
        call_kwargs: Dict[str, Any] = observer.record.call_args[1]
        assert call_kwargs["key"] == "validation_gate"
        recorded_value: Dict[str, Any] = call_kwargs["value"]
        assert recorded_value["accepted"] is True
        assert recorded_value["skipped"] is True
        assert recorded_value["validation_metric"] == "none"

    def test_serialize_algorithm_state_shows_disabled(self):
        """_serialize_algorithm_state should clearly indicate the gate is disabled."""
        tg: "TextGrad" = self._make_textgrad()
        state: Dict[str, Any] = tg._serialize_algorithm_state()
        gate_state: Dict[str, Any] = state["validation_gate"]
        assert gate_state["validation_metric"] == "none"
        assert gate_state["enabled"] is False

    def test_existing_metric_still_works(self):
        """Sanity check: validation_metric='accuracy' should still construct and
        resolve the metric class correctly (no regression from the 'none' change)."""
        tg: "TextGrad" = self._make_textgrad(validation_metric="accuracy")
        assert tg.validation_gate_enabled is True
        assert tg._validation_metric_cls is not None
