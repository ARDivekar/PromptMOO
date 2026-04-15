"""
Tests for the LLM worker (SlowBurnLLM via runner.py factory functions).

Covers:
- Single async LLM call
- Batch concurrent calls (50 prompts)
- High-load stress test (100 concurrent calls)
- Sequential call stability
"""

import time

import pytest


EVAL_PROMPT = (
    'Evaluate the summary. Output ONLY JSON: {"fluency": <1-5>}\n\n'
    "## Sample Point\nsummary: The cat sat on the mat.\n"
)


@pytest.mark.integration
class TestLLMWorkerCalls:
    """Basic correctness: LLM calls return valid responses."""

    def test_single_call(self, task_llm):
        """Single async call returns a non-empty string."""
        result = task_llm.call_llm(prompt="Say hello in one sentence.").result(
            timeout=30
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_single_call_json(self, task_llm):
        """Single call with a JSON-format prompt returns parseable JSON."""
        result = task_llm.call_llm(prompt=EVAL_PROMPT).result(timeout=30)
        assert "fluency" in result.lower()

    def test_batch_5(self, task_llm):
        """Small batch of 5 concurrent calls all succeed."""
        prompts = [EVAL_PROMPT] * 5
        results = task_llm.call_llm_batch(prompts=prompts, verbosity=0).result(
            timeout=60
        )
        assert len(results) == 5
        valid = sum(1 for r in results if "fluency" in str(r).lower())
        assert valid == 5

    def test_batch_50(self, task_llm):
        """Realistic batch of 50 concurrent calls (matches runner.py batch_size=48)."""
        prompts = [EVAL_PROMPT] * 50
        results = task_llm.call_llm_batch(prompts=prompts, verbosity=0).result(
            timeout=120
        )
        assert len(results) == 50
        valid = sum(1 for r in results if "fluency" in str(r).lower())
        assert valid >= 45, f"Only {valid}/50 responses contained valid JSON"

    def test_empty_batch(self, task_llm):
        """Empty batch returns empty list without errors."""
        results = task_llm.call_llm_batch(prompts=[], verbosity=0).result(timeout=10)
        assert results == []


@pytest.mark.integration
class TestHighLoad:
    """Stress tests with high concurrency."""

    @pytest.mark.timeout(300)
    def test_100_concurrent_calls(self, task_llm):
        """100 concurrent calls complete successfully."""
        prompts = [EVAL_PROMPT] * 100
        t0 = time.time()
        results = task_llm.call_llm_batch(prompts=prompts, verbosity=0).result(
            timeout=300
        )
        elapsed = time.time() - t0

        assert len(results) == 100
        valid = sum(1 for r in results if "fluency" in str(r).lower())
        assert valid >= 90, f"Only {valid}/100 valid JSON responses"
        print(f"  100 calls in {elapsed:.1f}s ({elapsed / 100:.2f}s/call)")


@pytest.mark.integration
class TestSequentialCalls:
    """Sequential call stability tests."""

    def test_calls_work_consecutively(self, task_llm):
        """Multiple sequential single calls all succeed."""
        for i in range(5):
            result = task_llm.call_llm(
                prompt=f"What is {i} + 1? Answer with just the number."
            ).result(timeout=30)
            assert isinstance(result, str)
            assert len(result) > 0

    def test_batch_after_single_calls(self, task_llm):
        """A batch call works after several single calls."""
        for _ in range(3):
            task_llm.call_llm(prompt="Say hello.").result(timeout=30)

        results = task_llm.call_llm_batch(
            prompts=[EVAL_PROMPT] * 10, verbosity=0
        ).result(timeout=60)
        assert len(results) == 10
