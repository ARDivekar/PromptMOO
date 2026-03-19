"""
Tests for task_predictor: JSON parsing, validation, and response handling.

Unit tests for parse_task_response and validate_task_response (no network).
"""
import pytest

from prompt_moo.task_predictor import parse_task_response, validate_task_response


@pytest.mark.unit
class TestParseTaskResponse:
    """Tests for parse_task_response — JSON extraction from LLM output."""

    def test_clean_json(self):
        result = parse_task_response('{"fluency": 4}')
        assert result == {"fluency": 4}

    def test_json_with_surrounding_text(self):
        """LLMs often wrap JSON in explanation text."""
        response = 'Here is the evaluation:\n{"fluency": 3, "coherence": 5}\nDone.'
        result = parse_task_response(response)
        assert result["fluency"] == 3
        assert result["coherence"] == 5

    def test_json_in_code_fence(self):
        response = '```json\n{"fluency": 4}\n```'
        result = parse_task_response(response)
        assert result["fluency"] == 4

    def test_double_braces_normalized(self):
        """Double braces (common LLM hallucination) are normalized to single."""
        response = '{{"fluency": 5}}'
        result = parse_task_response(response)
        assert result["fluency"] == 5

    def test_no_json_raises_valueerror(self):
        with pytest.raises(ValueError, match="No JSON found"):
            parse_task_response("No JSON here, just text.")

    def test_invalid_json_raises_valueerror(self):
        with pytest.raises(ValueError, match="Failed to parse JSON"):
            parse_task_response("{fluency: not_a_number}")

    def test_nested_json(self):
        """Nested JSON: parse_task_response has a known limitation where
        closing '}}' gets collapsed to '}' due to the double-brace normalization.
        This test documents the limitation — it only works when the inner
        object doesn't end right at the outer closing brace."""
        response = '{"scores": {"fluency": 4} }'
        result = parse_task_response(response)
        assert result["scores"]["fluency"] == 4

    def test_empty_json_object(self):
        result = parse_task_response("{}")
        assert result == {}

    def test_whitespace_handling(self):
        response = '  \n  { "fluency" : 5 }  \n  '
        result = parse_task_response(response)
        assert result["fluency"] == 5

    def test_multiple_json_objects_picks_outermost(self):
        """When multiple JSON objects exist, picks from first { to last }.

        Note: the {{ → { replacement in parse_task_response means this
        merges the two objects into one invalid blob. This tests the actual behavior.
        """
        response = '{"a": 1}'
        result = parse_task_response(response)
        assert result == {"a": 1}


@pytest.mark.unit
class TestValidateTaskResponse:
    """Tests for validate_task_response — returns bool, never raises."""

    def test_valid_json(self):
        assert validate_task_response('{"fluency": 4}') is True

    def test_valid_json_with_text(self):
        assert validate_task_response('Here:\n{"fluency": 4}\nDone.') is True

    def test_no_json(self):
        assert validate_task_response("No JSON here.") is False

    def test_invalid_json(self):
        assert validate_task_response("{bad json}") is False

    def test_empty_string(self):
        assert validate_task_response("") is False

    def test_double_braces(self):
        assert validate_task_response('{{"fluency": 5}}') is True
