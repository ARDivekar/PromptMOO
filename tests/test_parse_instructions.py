"""
Tests for ``extract_instructions_json`` — the shared LLM response parser.

Every test case below is derived from an actual Llama-3.1-8B-Instruct
response sampled at temperature=1.0 with optimizer-style prompts.  The
patterns cover every failure mode observed in 40 samples across GPO, OPRO,
TextGrad, and single-task optimizer prompts.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prompt_moo.types import extract_instructions_json

MULTI_TASKS = ["fluency", "consistency"]
SINGLE_TASK = ["coherence"]


# -----------------------------------------------------------------------
# Pattern 1: CLEAN_JSON — valid nested JSON, no surrounding text
# -----------------------------------------------------------------------


class TestCleanJSON:
    """Bare JSON with no preamble or code fences."""

    def test_multiline_nested_json(self):
        raw = (
            '{\n  "instructions": {\n'
            '    "fluency": "Rate from 1 to 5, prioritizing grammar.",\n'
            '    "consistency": "Rate from 1 to 5, verifying facts."\n'
            "  }\n}"
        )
        result = extract_instructions_json(raw, task_names=MULTI_TASKS)
        assert result["fluency"] == "Rate from 1 to 5, prioritizing grammar."
        assert result["consistency"] == "Rate from 1 to 5, verifying facts."

    def test_single_line_nested_json(self):
        raw = '{"instructions": {"fluency": "Rate fluency.", "consistency": "Rate consistency."}}'
        result = extract_instructions_json(raw, task_names=MULTI_TASKS)
        assert result["fluency"] == "Rate fluency."

    def test_double_closing_braces_not_corrupted(self):
        """The }} at the end of nested JSON must NOT be corrupted."""
        raw = '{"instructions": {"fluency": "a", "consistency": "b"}}'
        result = extract_instructions_json(raw, task_names=MULTI_TASKS)
        assert result["fluency"] == "a"
        assert result["consistency"] == "b"


# -----------------------------------------------------------------------
# Pattern 2: PREAMBLE — text before the JSON
# -----------------------------------------------------------------------


class TestPreamble:
    def test_preamble_text(self):
        """Llama often starts with 'Here is the improved instruction:'"""
        raw = (
            "Here is the improved instruction:\n\n"
            '{"instructions": {"fluency": "Rate fluency carefully.", "consistency": "Check facts."}}'
        )
        result = extract_instructions_json(raw, task_names=MULTI_TASKS)
        assert result["fluency"] == "Rate fluency carefully."

    def test_preamble_with_improved_instructions(self):
        raw = (
            "Here are the improved instructions in JSON format:\n\n"
            '{"instructions": {"fluency": "Assess coherence.", "consistency": "Rate consistency."}}'
        )
        result = extract_instructions_json(raw, task_names=MULTI_TASKS)
        assert result["fluency"] == "Assess coherence."

    def test_preamble_and_postamble(self):
        """Llama sometimes adds explanation after the JSON."""
        raw = (
            "Here is an improved prompt:\n\n"
            '{"instructions": {"fluency": "Rate carefully.", "consistency": "Check all."}}\n\n'
            "I made the following changes:\n1. Added clarity.\n2. Improved detail."
        )
        result = extract_instructions_json(raw, task_names=MULTI_TASKS)
        assert result["fluency"] == "Rate carefully."
        assert result["consistency"] == "Check all."


# -----------------------------------------------------------------------
# Pattern 3: CODE_FENCE — JSON wrapped in markdown code blocks
# -----------------------------------------------------------------------


class TestCodeFence:
    def test_code_fence_json(self):
        raw = '```json\n{"instructions": {"fluency": "Rate fluency.", "consistency": "Rate consistency."}}\n```'
        result = extract_instructions_json(raw, task_names=MULTI_TASKS)
        assert result["fluency"] == "Rate fluency."

    def test_code_fence_no_language_tag(self):
        raw = '```\n{"instructions": {"fluency": "Rate fluency.", "consistency": "Rate consistency."}}\n```'
        result = extract_instructions_json(raw, task_names=MULTI_TASKS)
        assert result["fluency"] == "Rate fluency."

    def test_preamble_plus_code_fence(self):
        raw = (
            "Here is an improved prompt:\n\n"
            "```\n"
            '{\n  "instructions": {\n'
            '    "fluency": "Rate fluency carefully.",\n'
            '    "consistency": "Rate consistency accurately."\n'
            "  }\n"
            "}\n"
            "```\n\n"
            "I made the following changes to improve the prompt."
        )
        result = extract_instructions_json(raw, task_names=MULTI_TASKS)
        assert result["fluency"] == "Rate fluency carefully."

    def test_code_fence_with_json_tag_and_explanation(self):
        """Real sample from single_task prompt."""
        raw = (
            "Here is an optimized version:\n\n"
            "```json\n"
            '{\n  "instructions": {\n'
            '    "coherence": "Rate coherence on a 1-5 scale."\n'
            "  }\n"
            "}\n"
            "```\n\n"
            "I've made the following changes to enhance clarity."
        )
        result = extract_instructions_json(raw, task_names=SINGLE_TASK)
        assert "1-5" in result["coherence"]


# -----------------------------------------------------------------------
# Pattern 4: ESCAPED_BRACES — f-string template artifacts {{ }}
# -----------------------------------------------------------------------


class TestEscapedBraces:
    def test_double_opening_and_closing_braces(self):
        """LLM echoes back the f-string template format."""
        raw = '{{"instructions": {{"fluency": "Rate fluency.", "consistency": "Rate consistency."}}}}'
        result = extract_instructions_json(raw, task_names=MULTI_TASKS)
        assert result["fluency"] == "Rate fluency."


# -----------------------------------------------------------------------
# Pattern 5: NESTED_DICT_VALUE — value is a dict, not a string
# -----------------------------------------------------------------------


class TestNestedDictValue:
    def test_nested_dict_flattened_to_string(self):
        """Llama sometimes returns structured dicts instead of plain strings."""
        raw = json.dumps(
            {
                "instructions": {
                    "coherence": {
                        "goal": "Evaluate coherence.",
                        "interpretation": {
                            "score3": "Partially conveys the main idea.",
                            "score4": "Effectively conveys with strong coherence.",
                        },
                    }
                }
            }
        )
        result = extract_instructions_json(raw, task_names=SINGLE_TASK)
        assert isinstance(result["coherence"], str)
        assert "Evaluate coherence." in result["coherence"]
        assert "Partially conveys" in result["coherence"]

    def test_nested_dict_with_description_and_rating(self):
        raw = json.dumps(
            {
                "instructions": {
                    "coherence": {
                        "description": "Evaluate coherence of the response.",
                        "rating": "Assign a rating from 1 to 5.",
                        "guidelines": {
                            "3": "Somewhat coherent.",
                            "4": "Generally coherent.",
                        },
                    }
                }
            }
        )
        result = extract_instructions_json(raw, task_names=SINGLE_TASK)
        assert isinstance(result["coherence"], str)
        assert "Evaluate coherence" in result["coherence"]


# -----------------------------------------------------------------------
# Pattern 6: TOP-LEVEL keys (no "instructions" wrapper)
# -----------------------------------------------------------------------


class TestTopLevelKeys:
    def test_bare_task_name_keys(self):
        raw = '{"fluency": "Rate from 1 to 5.", "consistency": "Rate consistency."}'
        result = extract_instructions_json(raw, task_names=MULTI_TASKS)
        assert result["fluency"] == "Rate from 1 to 5."

    def test_instruction_key_singular(self):
        raw = '{"instruction": {"fluency": "Rate it.", "consistency": "Check it."}}'
        result = extract_instructions_json(raw, task_names=MULTI_TASKS)
        assert result["fluency"] == "Rate it."


# -----------------------------------------------------------------------
# Edge cases
# -----------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_response_raises(self):
        with pytest.raises(ValueError):
            extract_instructions_json("", task_names=MULTI_TASKS)

    def test_no_json_raises(self):
        with pytest.raises(ValueError):
            extract_instructions_json(
                "Here is some text without JSON.", task_names=MULTI_TASKS
            )

    def test_missing_task_returns_only_present_tasks(self):
        raw = '{"instructions": {"fluency": "Rate fluency."}}'
        result = extract_instructions_json(raw, task_names=MULTI_TASKS)
        assert "fluency" in result
        assert "consistency" not in result

    def test_smart_quotes_handled(self):
        raw = '\u201c{"instructions": {"fluency": "Rate fluency.", "consistency": "Rate it."}}\u201d'
        result = extract_instructions_json(raw, task_names=MULTI_TASKS)
        assert result["fluency"] == "Rate fluency."


# -----------------------------------------------------------------------
# Real Llama-8B samples (verbatim from llama_optimizer_samples.json)
# -----------------------------------------------------------------------


class TestRealLlamaSamples:
    """Verbatim responses from Llama-3.1-8B-Instruct at temperature=1.0."""

    def test_gpo_sample_clean(self):
        raw = '{\n  "instructions": {\n    "fluency": "Rate from 1 to 5, prioritizing grammar and clarity.",\n    "consistency": "Rate from 1 to 5, verifying facts and coherence."\n  }\n}'
        result = extract_instructions_json(raw, task_names=MULTI_TASKS)
        assert "grammar and clarity" in result["fluency"]

    def test_gpo_sample_preamble_codefence_postamble(self):
        raw = 'Here is an improved prompt:\n\n```\n{\n  "instructions": {\n    "fluency": "Rate fluency carefully from 1 to 5, considering grammar and context.",\n    "consistency": "Rate from 1 to 5, assessing accuracy and factual correctness."\n  }\n}\n```\n\nI made the following changes to improve the prompt:\n\n1. Added "carefully" to the fluency instruction to emphasize the importance of attention to detail.\n2. Specifically mentioned "grammar" to ensure the model accounts for grammatical correctness.'
        result = extract_instructions_json(raw, task_names=MULTI_TASKS)
        assert "grammar and context" in result["fluency"]
        assert "factual correctness" in result["consistency"]

    def test_opro_sample_preamble_codefence(self):
        raw = 'Here are the improved instructions in JSON format:\n```\n{\n  "instructions": {\n    "fluency": "Assess how coherent and natural the summary sounds, with 1 being poor and 5 being excellent.",\n    "consistency": "Rate the consistency of the summary in terms of its structure and organization, with 1 being lacking and 5 being highly consistent."\n  }\n}\n```'
        result = extract_instructions_json(raw, task_names=MULTI_TASKS)
        assert "coherent and natural" in result["fluency"]

    def test_opro_sample_preamble_no_fence(self):
        raw = 'Here are the improved instructions in JSON format:\n\n{\n  "instructions": {\n    "fluency": "Rate the naturalness of the language output, with 1 being awkward and 5 being highly natural and engaging. Consider factors such as sentence structure, grammar, and vocabulary.",\n    "consistency": "Rate the consistency of the output in terms of tone, style, and overall coherence with the rest of the content, with 1 being inconsistent and 5 being highly consistent."\n  }\n}'
        result = extract_instructions_json(raw, task_names=MULTI_TASKS)
        assert "naturalness" in result["fluency"]

    def test_textgrad_sample_code_fence(self):
        raw = '```\n{\n  "instructions": {\n    "fluency": "Rate the fluency of the summary, focusing on grammar and sentence structure, with 1 being very poor (e.g., multiple grammatical errors) and 5 being excellent (idiomatic and natural flow).",\n    "consistency": "Rate the consistency of the summary in relation to its source content, with 1 being the summary deviating significantly from the source or main finding and 5 being the summary accurately and completely representing the source content."\n  }\n}\n```'
        result = extract_instructions_json(raw, task_names=MULTI_TASKS)
        assert "grammar and sentence structure" in result["fluency"]

    def test_textgrad_sample_improved_instruction_set_preamble(self):
        raw = 'Here is the improved instruction set:\n\n{\n  "instructions": {\n    "fluency": "Rate from 1 to 5, considering both grammatical correctness and natural phrasing of the summary.",\n    "consistency": "Rate from 1 to 5, specifically looking for alignment between the summary\'s main points and the original source material, avoiding contradictions and unclear or ambiguous statements."\n  }\n}'
        result = extract_instructions_json(raw, task_names=MULTI_TASKS)
        assert "grammatical correctness" in result["fluency"]

    def test_single_task_nested_dict_value(self):
        """Real sample: single_task index 3 — nested dict value."""
        raw = json.dumps(
            {
                "instructions": {
                    "coherence": {
                        "goal": "Evaluate the effectiveness of the text.",
                        "interpretation": {
                            "score1-2-5": "Does not clearly convey the main idea.",
                            "score3": "Partially conveys the main idea but lacks clarity.",
                            "score4": "Effectively conveys with strong coherence.",
                            "score5": "Clearly and precisely conveys the central idea.",
                        },
                    }
                }
            }
        )
        result = extract_instructions_json(raw, task_names=SINGLE_TASK)
        assert isinstance(result["coherence"], str)
        assert "Evaluate the effectiveness" in result["coherence"]

    def test_single_task_preamble_with_explanation_then_codefence(self):
        """Real sample: single_task index 9 — long preamble, then ```json block at the end."""
        raw = (
            "To improve the instruction for rating coherence from 1 to 5, "
            "let's refine the instruction. Here's an optimized version:\n\n"
            "**Original Instruction:** Rate from 1 to 5.\n\n"
            "**Improved Instruction:** Provide a coherence score.\n\n"
            "This optimized instruction includes:\n"
            "1. Clear definition\n2. Clarified criteria\n\n"
            "Here is the optimized instruction in the requested format:\n\n"
            "```json\n"
            '{\n  "instructions": {\n'
            '    "coherence": "Provide a coherence score from 1 to 5."\n'
            "  }\n"
            "}\n"
            "```\n\n"
            "This improvement is designed to enhance clarity."
        )
        result = extract_instructions_json(raw, task_names=SINGLE_TASK)
        assert "coherence score" in result["coherence"]

    def test_single_task_preamble_json_postamble(self):
        """Real sample: single_task index 2 — preamble, JSON, then explanation."""
        raw = (
            "Here is an optimized version of the instruction:\n\n"
            '{\n  "instructions": {\n'
            '    "coherence": "Rate coherence from 1-2 (low) or 4-5 (high), '
            'with 3 indicating a neutral point."\n'
            "  }\n}\n\n"
            "Changes made:\n"
            "1. Added context to ambiguity.\n"
            "2. Rephrased for clarity."
        )
        result = extract_instructions_json(raw, task_names=SINGLE_TASK)
        assert "neutral point" in result["coherence"]


# -----------------------------------------------------------------------
# Real Qwen3-8B samples (verbatim from qwen3_optimizer_samples.json)
# -----------------------------------------------------------------------


class TestRealQwen3Samples:
    """Verbatim responses from Qwen3-8B at temperature=1.0 with /no_think."""

    def test_qwen3_gpo_clean(self):
        raw = '{\n  "instructions": {\n    "fluency": "Rate from 1 to 5, ensuring clear and correct grammar.",\n    "consistency": "Rate from 1 to 5, verifying facts and coherence."\n  }\n}'
        result = extract_instructions_json(raw, task_names=MULTI_TASKS)
        assert "grammar" in result["fluency"]
        assert "verifying facts" in result["consistency"]

    def test_qwen3_opro_long_instructions(self):
        raw = '{\n  "instructions": {\n    "fluency": "Rate the fluency of the summary on a scale from 1 to 5, where 1 indicates very poor fluency (e.g., fragmented or unclear sentences) and 5 indicates excellent fluency (e.g., well-structured, easy-to-read, and natural-sounding text).",\n    "consistency": "Rate the consistency of the summary on a scale from 1 to 5, where 1 indicates very poor consistency (e.g., contradicts the source or contains unrelated information) and 5 indicates excellent consistency (e.g., accurately reflects the source content without introducing new or conflicting information)."\n  }\n}'
        result = extract_instructions_json(raw, task_names=MULTI_TASKS)
        assert "fluency" in result
        assert "consistency" in result
        assert "fragmented" in result["fluency"]

    def test_qwen3_textgrad_single_line(self):
        raw = '{"instructions": {"fluency": "Rate from 1 to 5. Consider grammar, sentence structure, and natural flow.", "consistency": "Rate from 1 to 5. Check alignment with the source, especially key findings."}}'
        result = extract_instructions_json(raw, task_names=MULTI_TASKS)
        assert "grammar" in result["fluency"]
        assert "alignment" in result["consistency"]

    def test_qwen3_single_task_code_fence(self):
        raw = '```json\n{\n  "instructions": {\n    "coherence": "Rate the coherence of the text on a scale from 1 to 5, where 3 indicates a partial logical flow with some inconsistencies, and 4 indicates a mostly logical and well-organized text with minor issues."\n  }\n}\n```'
        result = extract_instructions_json(raw, task_names=SINGLE_TASK)
        assert "logical flow" in result["coherence"]


# -----------------------------------------------------------------------
# Known unparseable responses (should raise ValueError cleanly)
# -----------------------------------------------------------------------


class TestUnparseableResponses:
    """Responses that contain template placeholders or malformed JSON.
    These should raise ValueError (triggering a retry) with a clear message."""

    def test_placeholder_angle_brackets(self):
        """Llama sometimes outputs <your_score> or <rated_integer> as placeholders."""
        raw = '{\n  "coherence": {\n    "rating": <your_score>\n  }\n}'
        with pytest.raises(ValueError, match="Failed to parse JSON"):
            extract_instructions_json(raw, task_names=SINGLE_TASK)

    def test_placeholder_rated_integer(self):
        raw = '{\n  "coherence": <rated_integer>\n}'
        with pytest.raises(ValueError, match="Failed to parse JSON"):
            extract_instructions_json(raw, task_names=SINGLE_TASK)
