"""
Tests for task_predictor: JSON parsing, validation, and response handling.

Tests are derived from actual Llama-3.1-8B and Qwen3-8B responses sampled
at temperature 0.1 and 0.9 with both single-task and multi-task prompts.

Observed patterns:
- CLEAN_JSON: bare {"fluency": 5, "consistency": 4} (most common)
- CODE_FENCE: ```json {"consistency": 1} ``` (Qwen3 minimal prompt)
- PREAMBLE: "Here is the score:\n{...}" (rare at low temp)
- SAFETY_REFUSAL: "I can't carry out that request." (Llama + adversarial input)
- NESTED_JSON: {"scores": {"fluency": 5}} (possible with some prompt formats)
- DOUBLE_BRACES: {{"fluency": 5}} (template echo artifact)
"""

import pytest

from prompt_moo.task_predictor import parse_task_response, validate_task_response


# -----------------------------------------------------------------------
# parse_task_response
# -----------------------------------------------------------------------


@pytest.mark.unit
class TestParseTaskResponse:
    """Tests for parse_task_response — JSON extraction from LLM output."""

    # --- Clean JSON (the normal case) ---

    def test_clean_single_task(self):
        result = parse_task_response('{"consistency": 1}')
        assert result == {"consistency": 1}

    def test_clean_multi_task(self):
        result = parse_task_response(
            '{"fluency": 5, "relevance": 4, "coherence": 4, "consistency": 5}'
        )
        assert result["fluency"] == 5
        assert result["relevance"] == 4
        assert result["coherence"] == 4
        assert result["consistency"] == 5

    def test_clean_multi_task_with_whitespace(self):
        """Verbatim Llama-8B output at temp=0.1."""
        raw = (
            '{   "fluency": 5,   "relevance": 5,   "coherence": 5,   "consistency": 5 }'
        )
        result = parse_task_response(raw)
        assert result["fluency"] == 5
        assert len(result) == 4

    def test_clean_multi_task_multiline(self):
        raw = '{\n  "fluency": 4,\n  "relevance": 4,\n  "coherence": 5,\n  "consistency": 4\n}'
        result = parse_task_response(raw)
        assert result["fluency"] == 4

    def test_empty_json_object(self):
        result = parse_task_response("{}")
        assert result == {}

    # --- Code fences (Qwen3 pattern) ---

    def test_code_fence_json_tag(self):
        """Verbatim Qwen3-8B output: ```json { } ```"""
        raw = '```json {   "consistency": 1 } ```'
        result = parse_task_response(raw)
        assert result["consistency"] == 1

    def test_code_fence_multiline(self):
        raw = '```json\n{"fluency": 4}\n```'
        result = parse_task_response(raw)
        assert result["fluency"] == 4

    def test_code_fence_no_language_tag(self):
        raw = '```\n{"fluency": 3, "consistency": 2}\n```'
        result = parse_task_response(raw)
        assert result["fluency"] == 3

    # --- Preamble / postamble ---

    def test_preamble_text(self):
        raw = 'Here is the evaluation:\n{"fluency": 3, "coherence": 5}\nDone.'
        result = parse_task_response(raw)
        assert result["fluency"] == 3
        assert result["coherence"] == 5

    def test_whitespace_surrounding(self):
        raw = '  \n  { "fluency" : 5 }  \n  '
        result = parse_task_response(raw)
        assert result["fluency"] == 5

    # --- Double braces (template echo) ---

    def test_double_braces_single_task(self):
        raw = '{{"consistency": 5}}'
        result = parse_task_response(raw)
        assert result["consistency"] == 5

    def test_double_braces_multi_task(self):
        raw = '{{"fluency": 5, "consistency": 4}}'
        result = parse_task_response(raw)
        assert result["fluency"] == 5
        assert result["consistency"] == 4

    # --- Nested JSON (the }} bug fix) ---

    def test_nested_json_with_double_closing_braces(self):
        """This was the }} corruption bug. The nested JSON has }} at the end
        which must NOT be corrupted by the brace replacement."""
        raw = '{"scores": {"fluency": 5, "consistency": 4}}'
        result = parse_task_response(raw)
        assert result["scores"]["fluency"] == 5
        assert result["scores"]["consistency"] == 4

    def test_nested_json_with_space_before_close(self):
        raw = '{"scores": {"fluency": 4} }'
        result = parse_task_response(raw)
        assert result["scores"]["fluency"] == 4

    # --- String values (model outputs string instead of int) ---

    def test_string_integer_values(self):
        """Some models return "5" instead of 5."""
        raw = '{"fluency": "5", "consistency": "3"}'
        result = parse_task_response(raw)
        assert result["fluency"] == "5"
        assert result["consistency"] == "3"

    # --- Safety refusal (no JSON at all) ---

    def test_safety_refusal_raises_valueerror(self):
        """Llama-8B refuses adversarial inputs with no JSON."""
        raw = "I can't carry out that request. Is there anything else I can help you with?"
        with pytest.raises(ValueError, match="No JSON found"):
            parse_task_response(raw)

    def test_safety_refusal_cant_fulfill(self):
        raw = "I can't fulfill that request."
        with pytest.raises(ValueError, match="No JSON found"):
            parse_task_response(raw)

    # --- Error inputs ---

    def test_none_raises(self):
        with pytest.raises(ValueError, match="None"):
            parse_task_response(None)

    def test_no_json_raises(self):
        with pytest.raises(ValueError, match="No JSON found"):
            parse_task_response("No JSON here, just text.")

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="Failed to parse JSON"):
            parse_task_response("{fluency: not_a_number}")

    def test_placeholder_angle_brackets_raises(self):
        """Model outputs template placeholder instead of value."""
        raw = '{"consistency": <score>}'
        with pytest.raises(ValueError, match="Failed to parse JSON"):
            parse_task_response(raw)

    # --- Real Llama-8B verbatim samples ---

    def test_llama_multi_messy_prompt(self):
        """Verbatim from Llama-8B at temp=0.9 with optimized prompt."""
        raw = (
            '{   "fluency": 4,   "relevance": 4,   "coherence": 5,   "consistency": 4 }'
        )
        result = parse_task_response(raw)
        assert result["fluency"] == 4
        assert result["consistency"] == 4

    def test_llama_single_minimal(self):
        raw = '{   "consistency": 1 }'
        result = parse_task_response(raw)
        assert result["consistency"] == 1

    # --- Real Qwen3-8B verbatim samples ---

    def test_qwen3_multi_task(self):
        raw = (
            '{   "fluency": 5,   "relevance": 4,   "coherence": 4,   "consistency": 5 }'
        )
        result = parse_task_response(raw)
        assert result["fluency"] == 5
        assert result["consistency"] == 5

    def test_qwen3_single_minimal_code_fence(self):
        """Qwen3 always wraps minimal-prompt responses in code fences."""
        raw = '```json {   "consistency": 1 } ```'
        result = parse_task_response(raw)
        assert result["consistency"] == 1


# -----------------------------------------------------------------------
# validate_task_response
# -----------------------------------------------------------------------


@pytest.mark.unit
class TestValidateTaskResponse:
    """Tests for validate_task_response — returns bool, never raises."""

    def test_valid_clean_json(self):
        assert validate_task_response('{"fluency": 4}') is True

    def test_valid_multi_task(self):
        assert validate_task_response('{"fluency": 5, "consistency": 4}') is True

    def test_valid_with_preamble(self):
        assert validate_task_response('Here:\n{"fluency": 4}\nDone.') is True

    def test_valid_code_fence(self):
        assert validate_task_response('```json\n{"consistency": 1}\n```') is True

    def test_valid_double_braces(self):
        assert validate_task_response('{{"fluency": 5}}') is True

    def test_valid_nested_json(self):
        assert validate_task_response('{"scores": {"fluency": 5}}') is True

    def test_invalid_no_json(self):
        assert validate_task_response("No JSON here.") is False

    def test_invalid_bad_json(self):
        assert validate_task_response("{bad json}") is False

    def test_invalid_empty_string(self):
        assert validate_task_response("") is False

    def test_invalid_none(self):
        assert validate_task_response(None) is False

    def test_invalid_safety_refusal(self):
        assert validate_task_response("I can't carry out that request.") is False

    def test_invalid_placeholder(self):
        assert validate_task_response('{"consistency": <score>}') is False


# -----------------------------------------------------------------------
# _extract_numeric_outputs
# -----------------------------------------------------------------------


@pytest.mark.unit
class TestExtractNumericOutputs:
    """Tests for the int/float extraction filter used by StandardTaskPredictor."""

    def test_integer_values(self):
        from prompt_moo.task_predictor import _extract_numeric_outputs

        result = _extract_numeric_outputs({"fluency": 4, "consistency": 3})
        assert result == {"fluency": 4, "consistency": 3}

    def test_float_values(self):
        from prompt_moo.task_predictor import _extract_numeric_outputs

        result = _extract_numeric_outputs({"fluency": 4.5, "consistency": 3.2})
        assert result == {"fluency": 4.5, "consistency": 3.2}

    def test_string_integer_coerced(self):
        """Models sometimes return "5" instead of 5."""
        from prompt_moo.task_predictor import _extract_numeric_outputs

        result = _extract_numeric_outputs({"fluency": "5", "consistency": "3"})
        assert result == {"fluency": 5, "consistency": 3}
        assert isinstance(result["fluency"], int)

    def test_string_float_coerced(self):
        from prompt_moo.task_predictor import _extract_numeric_outputs

        result = _extract_numeric_outputs({"fluency": "4.5"})
        assert result == {"fluency": 4.5}
        assert isinstance(result["fluency"], float)

    def test_mixed_int_and_string_int(self):
        from prompt_moo.task_predictor import _extract_numeric_outputs

        result = _extract_numeric_outputs({"fluency": 4, "consistency": "3"})
        assert result == {"fluency": 4, "consistency": 3}

    def test_boolean_excluded(self):
        """bool is a subclass of int in Python; True would be 1, False would be 0."""
        from prompt_moo.task_predictor import _extract_numeric_outputs

        result = _extract_numeric_outputs({"fluency": 4, "consistency": True})
        assert result == {"fluency": 4}
        assert "consistency" not in result

    def test_none_excluded(self):
        from prompt_moo.task_predictor import _extract_numeric_outputs

        result = _extract_numeric_outputs({"fluency": 4, "consistency": None})
        assert result == {"fluency": 4}

    def test_string_text_excluded(self):
        """Non-numeric strings (e.g. reasoning text) should be dropped."""
        from prompt_moo.task_predictor import _extract_numeric_outputs

        result = _extract_numeric_outputs(
            {
                "fluency": 4,
                "consistency": 3,
                "reasoning": "The summary is good",
            }
        )
        assert result == {"fluency": 4, "consistency": 3}

    def test_dict_value_excluded(self):
        from prompt_moo.task_predictor import _extract_numeric_outputs

        result = _extract_numeric_outputs({"fluency": 4, "meta": {"score": 5}})
        assert result == {"fluency": 4}

    def test_empty_dict(self):
        from prompt_moo.task_predictor import _extract_numeric_outputs

        result = _extract_numeric_outputs({})
        assert result == {}

    def test_all_string_values(self):
        from prompt_moo.task_predictor import _extract_numeric_outputs

        result = _extract_numeric_outputs({"fluency": "5", "consistency": "4"})
        assert result == {"fluency": 5, "consistency": 4}

    def test_string_with_whitespace(self):
        from prompt_moo.task_predictor import _extract_numeric_outputs

        result = _extract_numeric_outputs({"fluency": " 4 "})
        assert result == {"fluency": 4}
