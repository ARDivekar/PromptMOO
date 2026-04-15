"""Shared types and utilities for PromptMOO."""

import json
import re
from typing import Any, List, Literal, Optional, Protocol, runtime_checkable

from morphic import validate
from morphic.typed import format_exception_msg

from prompt_moo.exceptions import PromptParsingError

TaskStrategy = Literal["separate_tasks", "combine_all_tasks"]
SEPARATE_TASKS: TaskStrategy = "separate_tasks"
COMBINE_ALL_TASKS: TaskStrategy = "combine_all_tasks"


@runtime_checkable
class LLMPool(Protocol):
    """Duck-typed interface for LLM worker pools (e.g. SlowBurnLLM)."""

    def call_llm_batch(
        self,
        *,
        prompts: List[str],
        verbosity: int = 1,
        validator: Optional[Any] = None,
    ) -> Any:
        """Submit a batch of prompts and return a Future-like result."""
        ...

    def stop(self) -> None:
        """Shut down the pool."""
        ...


_SMART_QUOTES = '"\u201c\u201d\u201e\u201f\u00ab\u00bb'


def strip_smart_quotes(text: str) -> str:
    """Strip all leading/trailing straight and curly quotes from *text*.

    Replaces the repeated ``.removeprefix('\"').removesuffix('\"')`` chains
    that appeared throughout the optimizer parsers.
    """
    return text.strip(_SMART_QUOTES)


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_ESCAPED_BRACE_OPEN_RE = re.compile(r"^(\s*)\{\{", re.MULTILINE)
_ESCAPED_BRACE_CLOSE_RE = re.compile(r"\}\}(\s*)$", re.MULTILINE)


def clean_json_response(response: str) -> str:
    """Extract and clean a JSON object from a raw LLM response string.

    Strips surrounding whitespace, replaces escaped ``{{`` / ``}}`` that
    appear at line boundaries (template artefacts) with single braces,
    then returns the outermost ``{...}`` block.

    Raises:
        PromptParsingError: If the response contains no JSON object.
    """
    text = response.strip()
    text = _ESCAPED_BRACE_OPEN_RE.sub(r"\1{", text)
    text = _ESCAPED_BRACE_CLOSE_RE.sub(r"}\1", text)

    match = _JSON_BLOCK_RE.search(text)
    if match is None:
        raise PromptParsingError(f"No JSON object found in LLM response: {response}")
    return match.group(0)


# ---------------------------------------------------------------------------
# Shared instruction-JSON parser
# ---------------------------------------------------------------------------

_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)


@validate
def extract_instructions_json(
    response: str,
    *,
    task_names: List[str],
) -> dict:
    """Parse an LLM response into a ``{task_name: instruction_str}`` dict.

    Handles every output pattern observed from Llama-3.1-8B:

    1. **Clean JSON**: ``{"instructions": {"task": "..."}}``
    2. **Preamble / postamble**: text before or after the JSON object
    3. **Code fences**: JSON wrapped in markdown ``` blocks
    4. **Nested dict values**: ``{"instructions": {"task": {"goal": ".."}}}``
       — flattened by joining the leaf string values
    5. **Escaped braces**: ``{{`` / ``}}`` from f-string template echo

    Args:
        response: Raw LLM response text.
        task_names: Expected task name keys.

    Returns:
        Dict mapping each task name to its instruction string.

    Raises:
        ValueError: If no valid instructions can be extracted.
    """
    if response is None or len(response.strip()) == 0:
        raise ValueError("Empty LLM response")

    json_str = _extract_json_substring(response)
    parsed = _parse_json_robust(json_str, raw_response=response)
    instructions = _extract_instructions_dict(parsed, raw_response=response)
    return _normalize_instruction_values(instructions, task_names=task_names)


def _extract_json_substring(response: str) -> str:
    """Extract the outermost JSON object from an LLM response.

    Tries in order:
    1. Content inside markdown code fences (```json ... ```)
    2. The outermost { ... } in the raw text
    """
    text = response.strip()

    fence_match = _CODE_FENCE_RE.search(text)
    if fence_match is not None:
        inner = fence_match.group(1).strip()
        if len(inner) > 0 and "{" in inner:
            text = inner

    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object found in response:\n{response}")

    return text[start:end]


def _parse_json_robust(json_str: str, *, raw_response: str) -> dict:
    """Parse a JSON string with fallback for escaped braces."""
    cleaned = json_str.replace("\n", " ").strip()
    cleaned = strip_smart_quotes(cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    fallback = cleaned.replace("{{", "{").replace("}}", "}")
    try:
        return json.loads(fallback)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse JSON: {format_exception_msg(e)}\n"
            f"Extracted: {json_str}\n"
            f"Response: {raw_response}"
        )


def _extract_instructions_dict(parsed: dict, *, raw_response: str) -> dict:
    """Pull the instructions dict from various response formats."""
    if not isinstance(parsed, dict):
        raise ValueError(f"Parsed JSON is not a dict: {type(parsed).__name__}")

    if "instructions" in parsed and isinstance(parsed["instructions"], dict):
        return parsed["instructions"]

    if "instruction" in parsed and isinstance(parsed["instruction"], dict):
        return parsed["instruction"]

    if len(parsed) > 0 and all(isinstance(v, (str, dict)) for v in parsed.values()):
        return parsed

    raise ValueError(
        f"Cannot find 'instructions' dict in parsed JSON. "
        f"Keys: {list(parsed.keys())}\nResponse: {raw_response}"
    )


def _normalize_instruction_values(
    instructions: dict,
    *,
    task_names: List[str],
) -> dict:
    """Ensure every value is a string.  Nested dicts are flattened by joining
    their leaf string values (handles the NESTED_DICT_VALUE pattern).
    """

    def _flatten_to_str(value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            parts = []
            for v in value.values():
                parts.append(_flatten_to_str(v))
            return " ".join(parts)
        return str(value)

    result = {}
    for tn in task_names:
        if tn in instructions:
            result[tn] = _flatten_to_str(instructions[tn])
    return result
