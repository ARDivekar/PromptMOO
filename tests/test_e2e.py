"""
End-to-end integration tests with real LLM calls via SlowBurnLLM.

API keys are loaded from .env by conftest.py at session startup.
Tests are skipped if OPENROUTER_API_KEY is not available.

These tests verify the full PromptMOO pipeline works end-to-end:
1. Factory functions create working SlowBurnLLM workers
2. Workers can make single and batch LLM calls
3. The validator callback pattern works for structured parsing
4. All four LLM roles (task, optimizer, gradient, loss) work with shared limits
5. The config system correctly parameterizes workers
6. The task predictor pipeline produces parseable results
"""
import json
import re
import time

import pytest

from prompt_moo.config import promptmoo_config, temp_config
from prompt_moo.data_structures import Batch, DatasetSample, PredictionResult, Task
from prompt_moo.task_predictor import parse_task_response, StandardTaskPredictor
from prompt_moo.prompt_template_utils import PromptTemplate

from tests.conftest import skip_no_api_key

FLUENCY_TASK = Task(
    task_name="fluency",
    task_description="1-5 grammar score",
    task_instruction="Check grammar, clarity, and readability. Output an integer 1-5.",
    gt_col="fluency",
)

EVAL_PROMPT = (
    'Evaluate the summary. Output ONLY JSON: {"fluency": <1-5>}\n\n'
    "## Sample Point\nsummary: The cat sat on the mat.\n"
)


# ---------------------------------------------------------------------------
# 1. Basic single + batch calls via factory-created workers
# ---------------------------------------------------------------------------
@skip_no_api_key
class TestFactoryWorkerRealCalls:
    """Real LLM calls through runner.py factory-created SlowBurnLLM workers.

    Verifies:
    - create_task_llm produces a worker that can call LLMs
    - Single calls return non-empty strings
    - Batch calls return the correct number of results
    - The SlowBurnLLM reporter tracks call count
    """

    @pytest.fixture(autouse=True)
    def _setup(self, api_key):
        from runner import create_shared_limits, create_task_llm
        self._limits = create_shared_limits()
        self._llm = create_task_llm(llm="llama3.1", api_key=api_key, limits=self._limits)
        yield
        self._llm.stop()

    def test_single_call_returns_text(self):
        """A single call_llm returns a non-empty string response."""
        result = self._llm.call_llm(
            prompt="What is 2+2? Answer in one word.",
        ).result(timeout=30.0)

        assert isinstance(result, str)
        assert len(result) > 0
        print(f"Response: {result!r}")

    def test_single_call_tracks_cost(self):
        """SlowBurnLLM reporter logs the call."""
        self._llm.call_llm(prompt="Say hello.").result(timeout=30.0)

        reporter = self._llm.get_reporter().result(timeout=5.0)
        assert reporter.num_calls == 1
        assert reporter.total_cost() >= 0
        print(f"Calls: {reporter.num_calls}, Cost: ${reporter.total_cost():.6f}")

    def test_batch_3_calls(self):
        """Batch of 3 concurrent calls all succeed."""
        results = self._llm.call_llm_batch(
            prompts=["Capital of France?", "Capital of Japan?", "Capital of Brazil?"],
            verbosity=0,
        ).result(timeout=60.0)

        assert len(results) == 3
        for r in results:
            assert isinstance(r, str)
            assert len(r) > 0
        print(f"Results: {[r.strip()[:50] for r in results]}")

    def test_batch_10_json_responses(self):
        """Batch of 10 evaluation prompts produce JSON-parseable results."""
        prompts = [EVAL_PROMPT] * 10
        results = self._llm.call_llm_batch(prompts=prompts, verbosity=0).result(timeout=120.0)

        assert len(results) == 10
        valid = sum(1 for r in results if "fluency" in str(r).lower())
        assert valid >= 8, f"Only {valid}/10 responses contained 'fluency'"

    def test_empty_batch_returns_empty(self):
        """Empty prompt list returns empty results without error."""
        results = self._llm.call_llm_batch(prompts=[], verbosity=0).result(timeout=10.0)
        assert results == []

    def test_multiple_sequential_calls_accumulate(self):
        """Three sequential calls accumulate in the reporter."""
        for prompt in ["Say hi.", "Say bye.", "Say thanks."]:
            self._llm.call_llm(prompt=prompt).result(timeout=30.0)

        reporter = self._llm.get_reporter().result(timeout=5.0)
        assert reporter.num_calls == 3
        assert reporter.total_cost() > 0
        print(f"3 calls, cost: ${reporter.total_cost():.6f}")


# ---------------------------------------------------------------------------
# 2. Validator callback pattern (structured parsing with retry)
# ---------------------------------------------------------------------------
@skip_no_api_key
class TestValidatorRealCalls:
    """Validator-based parsing with real LLM calls.

    Verifies:
    - A validator function receives the raw response text
    - Successful validation returns the parsed result (not the raw string)
    - The validator pattern replaces the old dead-retry-loop in prompt_optimizer
    """

    def test_validator_parses_json_dict(self, api_key):
        """Validator extracts a JSON dict from the LLM response.

        Steps:
        1. Create a task LLM with retry on ValueError
        2. Send a prompt asking for JSON output
        3. Use parse_task_response as the validator
        4. Verify the result is a parsed dict, not a raw string
        """
        from runner import create_shared_limits, create_task_llm
        limits = create_shared_limits()
        llm = create_task_llm(llm="llama3.1", api_key=api_key, limits=limits)

        try:
            result = llm.call_llm(
                prompt=EVAL_PROMPT,
                validator=parse_task_response,
            ).result(timeout=30.0)

            assert isinstance(result, dict), f"Expected dict, got {type(result).__name__}: {result!r}"
            assert "fluency" in result
            print(f"Parsed result: {result}")
        finally:
            llm.stop()

    def test_validator_parses_integer(self, api_key):
        """Validator extracts a number from the LLM response.

        Steps:
        1. Ask the LLM a math question
        2. Use a regex-based validator to extract the integer
        3. Verify the result is an int, not a string
        """
        from runner import create_shared_limits, create_task_llm
        limits = create_shared_limits()
        llm = create_task_llm(llm="llama3.1", api_key=api_key, limits=limits)

        try:
            def extract_number(text: str) -> int:
                match = re.search(r"\d+", text)
                if match is None:
                    raise ValueError(f"No number found in: {text!r}")
                return int(match.group())

            result = llm.call_llm(
                prompt="What is 7 * 8? Reply with just the number.",
                validator=extract_number,
            ).result(timeout=30.0)

            assert isinstance(result, int)
            assert result == 56
            print(f"Parsed integer: {result}")
        finally:
            llm.stop()

    def test_batch_validator(self, api_key):
        """Validator works in batch mode, parsing each response independently.

        Steps:
        1. Send 3 evaluation prompts in a batch
        2. Use parse_task_response as the validator
        3. Verify all 3 results are parsed dicts
        """
        from runner import create_shared_limits, create_task_llm
        limits = create_shared_limits()
        llm = create_task_llm(llm="llama3.1", api_key=api_key, limits=limits)

        try:
            results = llm.call_llm_batch(
                prompts=[EVAL_PROMPT] * 3,
                validator=parse_task_response,
                verbosity=0,
            ).result(timeout=60.0)

            assert len(results) == 3
            for r in results:
                assert isinstance(r, dict), f"Expected dict, got {type(r).__name__}"
                assert "fluency" in r
            print(f"All 3 batch results parsed: {results}")
        finally:
            llm.stop()


# ---------------------------------------------------------------------------
# 3. All four LLM roles with shared limits
# ---------------------------------------------------------------------------
@skip_no_api_key
class TestAllFourRoles:
    """Create and use all four LLM roles (task, optimizer, gradient, loss).

    Verifies:
    - All four factory functions create working workers
    - All four workers share the same LimitSet (no deadlocks)
    - Each worker can independently make LLM calls
    """

    @pytest.mark.timeout(120)
    def test_all_roles_make_calls(self, api_key):
        """Create all 4 LLM workers and make one call each.

        Steps:
        1. Create shared limits
        2. Create task, optimizer, gradient, and loss LLMs
        3. Make one call on each
        4. Verify all 4 return non-empty strings
        5. Stop all workers cleanly
        """
        from runner import (
            create_shared_limits,
            create_task_llm,
            create_optimizer_llm,
            create_gradient_llm,
            create_loss_llm,
        )
        limits = create_shared_limits()
        workers = {
            "task": create_task_llm(llm="llama3.1", api_key=api_key, limits=limits),
            "optimizer": create_optimizer_llm(llm="llama3.1", api_key=api_key, limits=limits),
            "gradient": create_gradient_llm(llm="llama3.1", api_key=api_key, limits=limits),
            "loss": create_loss_llm(llm="llama3.1", api_key=api_key, limits=limits),
        }

        try:
            for role, worker in workers.items():
                result = worker.call_llm(prompt=f"Say '{role}' and nothing else.").result(timeout=60.0)
                assert isinstance(result, str)
                assert len(result) > 0
                print(f"  {role}: {result.strip()!r}")
        finally:
            for worker in workers.values():
                worker.stop()


# ---------------------------------------------------------------------------
# 4. Config system integration: temp_config affects worker behavior
# ---------------------------------------------------------------------------
@skip_no_api_key
class TestConfigIntegration:
    """Verify the config system parameterizes real worker creation.

    Verifies:
    - temp_config overrides are read by factory functions
    - Workers created inside temp_config use the overridden values
    """

    def test_temp_config_changes_max_tokens(self, api_key):
        """Workers created inside temp_config(task_llm_max_tokens=10) have low max_tokens.

        Steps:
        1. Create a task LLM inside temp_config with max_tokens=10
        2. Make a call asking for a long response
        3. Verify the response is short (truncated by low max_tokens)
        4. Compare with a call using normal max_tokens
        """
        from runner import create_shared_limits, create_task_llm

        with temp_config(task_llm_max_tokens=10):
            limits = create_shared_limits()
            short_llm = create_task_llm(llm="llama3.1", api_key=api_key, limits=limits)
            try:
                short_result = short_llm.call_llm(
                    prompt="Write a 500 word essay about cats.",
                ).result(timeout=30.0)
                print(f"Short response ({len(short_result)} chars): {short_result[:100]!r}")
            finally:
                short_llm.stop()

        limits2 = create_shared_limits()
        normal_llm = create_task_llm(llm="llama3.1", api_key=api_key, limits=limits2)
        try:
            normal_result = normal_llm.call_llm(
                prompt="Write a 500 word essay about cats.",
            ).result(timeout=30.0)
            print(f"Normal response ({len(normal_result)} chars): {normal_result[:100]!r}")
        finally:
            normal_llm.stop()

        assert len(short_result) < len(normal_result), (
            f"Short response ({len(short_result)}) should be shorter "
            f"than normal ({len(normal_result)})"
        )


# ---------------------------------------------------------------------------
# 5. Task predictor pipeline e2e
# ---------------------------------------------------------------------------
@skip_no_api_key
class TestTaskPredictorE2E:
    """End-to-end test of the task predictor pipeline.

    Verifies:
    - StandardTaskPredictor.predict() calls the LLM and parses JSON
    - PredictionResult objects are returned with task_outputs populated
    - The full pipeline (prompt formatting -> LLM call -> JSON parse) works
    """

    @pytest.mark.timeout(120)
    def test_predict_summeval_batch(self, api_key):
        """Run StandardTaskPredictor on a small SummEval-like batch.

        Steps:
        1. Create task LLM and a PromptTemplate for fluency evaluation
        2. Build a batch of 3 synthetic samples
        3. Call predictor.predict()
        4. Verify PredictionResult objects have fluency scores
        """
        from runner import (
            create_shared_limits,
            create_task_llm,
            get_initial_prompt,
        )

        limits = create_shared_limits()
        llm = create_task_llm(llm="llama3.1", api_key=api_key, limits=limits)

        try:
            prompt_template = get_initial_prompt(
                dataset_name="SummEval",
                tasks=[FLUENCY_TASK],
            )

            samples = [
                DatasetSample(
                    sample_id=f"e2e_s{i}",
                    inputs={"summary": text, "source": "Test source document."},
                    ground_truths={"fluency": gt},
                )
                for i, (text, gt) in enumerate([
                    ("The cat sat on the mat. It was a good cat.", 4),
                    ("cat mat sat good. very nice.", 2),
                    ("The feline creature positioned itself upon the woven floor covering.", 5),
                ])
            ]
            batch = Batch(step=0, samples=samples)

            predictor = StandardTaskPredictor()
            predictions = predictor.predict(
                batch=batch,
                prompt_template=prompt_template,
                llm_pool=llm,
                verbosity=1,
            )

            assert len(predictions) >= 1, "Should produce at least 1 prediction"
            for pred in predictions:
                assert isinstance(pred, PredictionResult)
                assert pred.raw_response is not None
                print(f"  Sample {pred.sample_id}: outputs={pred.task_outputs}, raw={pred.raw_response[:80]!r}")

            has_fluency = sum(1 for p in predictions if "fluency" in p.task_outputs)
            print(f"  {has_fluency}/{len(predictions)} predictions have fluency scores")
            assert has_fluency >= 1, "At least 1 prediction should have a fluency score"
        finally:
            llm.stop()


# ---------------------------------------------------------------------------
# 6. Shared limits enforcement (no deadlocks under concurrent load)
# ---------------------------------------------------------------------------
@skip_no_api_key
class TestSharedLimitsE2E:
    """Verify shared limits don't deadlock under concurrent load.

    Verifies:
    - Two workers sharing limits can make calls concurrently
    - No deadlock occurs when both workers are active
    """

    @pytest.mark.timeout(120)
    def test_two_workers_concurrent_no_deadlock(self, api_key):
        """Two workers sharing limits make concurrent batch calls without deadlocking.

        Steps:
        1. Create shared limits with low parallel_calls (5)
        2. Create two task LLMs sharing those limits
        3. Submit 5 calls on each worker (10 total, exceeding the parallel limit)
        4. Verify all 10 calls complete (proves no deadlock)
        """
        from concurry import CallLimit, LimitSet, RateLimit, ResourceLimit
        from runner import LLM_CONFIGS, _build_litellm_params, _build_retry_config
        from slowburn import SlowBurnLLM

        cfg = promptmoo_config.defaults
        limits = LimitSet(
            limits=[
                ResourceLimit(key="parallel_calls", capacity=5),
                CallLimit(window_seconds=60, capacity=cfg.max_rpm),
                RateLimit(key="input_tokens", window_seconds=60, capacity=cfg.max_input_tpm),
                RateLimit(key="output_tokens", window_seconds=60, capacity=cfg.max_output_tpm),
            ],
            mode="asyncio",
            shared=True,
        )

        model_name = LLM_CONFIGS["llama3.1"]["task_model"]
        providers = LLM_CONFIGS["llama3.1"]["provider_order"].get(model_name)

        worker_a = SlowBurnLLM.options(
            mode="asyncio",
            limits=limits,
            **_build_retry_config(cfg=cfg),
        ).init(
            name="worker_a",
            model_name=model_name,
            api_key=api_key,
            temperature=0.1,
            max_tokens=50,
            timeout=60.0,
            litellm_params=_build_litellm_params(
                providers=providers,
                model_name=model_name,
                reasoning=False,
            ),
        )

        worker_b = SlowBurnLLM.options(
            mode="asyncio",
            limits=limits,
            **_build_retry_config(cfg=cfg),
        ).init(
            name="worker_b",
            model_name=model_name,
            api_key=api_key,
            temperature=0.1,
            max_tokens=50,
            timeout=60.0,
            litellm_params=_build_litellm_params(
                providers=providers,
                model_name=model_name,
                reasoning=False,
            ),
        )

        try:
            future_a = worker_a.call_llm_batch(
                prompts=["Say A."] * 5, verbosity=0,
            )
            future_b = worker_b.call_llm_batch(
                prompts=["Say B."] * 5, verbosity=0,
            )

            results_a = future_a.result(timeout=90.0)
            results_b = future_b.result(timeout=90.0)

            assert len(results_a) == 5
            assert len(results_b) == 5
            print(f"Worker A: {len(results_a)} results, Worker B: {len(results_b)} results")
        finally:
            worker_a.stop()
            worker_b.stop()
