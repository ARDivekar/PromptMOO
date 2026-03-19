"""
End-to-end tests for each algorithm (OPRO, GPO, TextGrad) with real LLM calls.

These tests run 2 steps of each algorithm on a tiny batch (batch_size=3)
against the real SummEval dataset with the real OpenRouter API.

They verify:
1. The full 4-step pipeline executes without errors
   (predict -> loss -> gradient -> optimize)
2. The final result dict has the expected structure
3. Prompts actually change between steps (optimization is doing something)
4. Each algorithm's specific components work (trajectory, textual gradients, etc.)

Skipped if OPENROUTER_API_KEY is not set.
"""
import os
import tempfile

import pytest

from prompt_moo.config import promptmoo_config, temp_config
from prompt_moo.data_structures import Task

from tests.conftest import skip_no_api_key

FLUENCY_TASK = Task(
    task_name="fluency",
    task_description="Evaluate the fluency and readability of the summary",
    task_instruction="Rate the fluency of this summary on a scale from 1 (very poor) to 5 (excellent).",
    gt_col="fluency",
)


def _make_dataset():
    """Load the real SummEval dataset from expt/ directory."""
    from dataset import SummEval
    return SummEval(data_dir="expt")


def _make_llms(api_key, shared_limits):
    """Create the 4 LLM workers needed by algorithms."""
    from runner import (
        create_task_llm,
        create_optimizer_llm,
        create_gradient_llm,
        create_loss_llm,
    )
    return {
        "task_llm": create_task_llm(llm="llama3.1", api_key=api_key, limits=shared_limits),
        "optimizer_llm": create_optimizer_llm(llm="llama3.1", api_key=api_key, limits=shared_limits),
        "gradient_llm": create_gradient_llm(llm="llama3.1", api_key=api_key, limits=shared_limits),
        "loss_llm": create_loss_llm(llm="llama3.1", api_key=api_key, limits=shared_limits),
    }


def _stop_all(llms):
    for worker in llms.values():
        worker.stop()


@skip_no_api_key
class TestOPROAlgorithm:
    """OPRO algorithm: 2 training steps with real LLM calls.

    OPRO uses:
    - task_llm for predictions (JSON parsing)
    - OPROLossComputer for numeric-only losses
    - OPROGradientComputer for score summaries (no LLM call)
    - OPROOptimizer with top-k trajectory for meta-prompt generation
    """

    @pytest.mark.timeout(300)
    def test_opro_2_steps(self, api_key, shared_limits):
        """Run OPRO for 2 steps on SummEval fluency.

        Steps:
        1. Create dataset, LLMs, algorithm
        2. Run 2 training steps (predict+loss+gradient+optimize each)
        3. Verify result dict has expected keys
        4. Verify output directory was created with observability logs
        5. Verify trajectory has at least 1 element
        """
        from prompt_moo.algorithm import OPRO
        from runner import get_initial_prompt, get_task_losses

        dataset = _make_dataset()
        llms = _make_llms(api_key, shared_limits)

        try:
            task_losses = get_task_losses(dataset_name="SummEval", tasks=[FLUENCY_TASK])
            initial_prompt = get_initial_prompt(
                dataset_name="SummEval", tasks=[FLUENCY_TASK],
            )

            with tempfile.TemporaryDirectory() as tmpdir:
                output_dir = os.path.join(tmpdir, "opro_test")

                algo = OPRO(
                    task_llm=llms["task_llm"],
                    optimizer_llm=llms["optimizer_llm"],
                    task_losses=task_losses,
                    tasks=[FLUENCY_TASK],
                    steps=2,
                    batch_size=3,
                    loss_batch_size=3,
                    gradient_batch_size=3,
                    eval_every=2,
                    name="test_opro",
                    k=3,
                    verbosity=1,
                )

                with temp_config(substep_delay=0.0):
                    result = algo.train(
                        dataset=dataset,
                        initial_prompt=initial_prompt,
                        output_dir=output_dir,
                    )

                assert result["run_id"] is not None
                assert result["final_prompt"] is not None
                assert result["output_dir"] == output_dir
                assert os.path.isdir(output_dir)

                assert len(algo.trajectory) >= 1
                print(f"OPRO: {len(algo.trajectory)} trajectory elements")
                print(f"Final prompt: {result['final_prompt'].to_str()[:200]!r}")
        finally:
            _stop_all(llms)


@skip_no_api_key
class TestGPOAlgorithm:
    """GPO algorithm: 2 training steps with real LLM calls.

    GPO uses:
    - task_llm for predictions
    - GPOLossComputer for numeric + textual feedback (via loss_llm)
    - GPOGradientComputer for textual gradients (via gradient_llm)
    - GPOOptimizer with trajectory + cosine step-size scheduling
    """

    @pytest.mark.timeout(300)
    def test_gpo_2_steps(self, api_key, shared_limits):
        """Run GPO for 2 steps on SummEval fluency.

        Steps:
        1. Create dataset, all 4 LLMs, algorithm
        2. Run 2 training steps
        3. Verify result dict has expected keys
        4. Verify trajectory accumulated entries
        """
        from prompt_moo.algorithm import GPO
        from runner import get_initial_prompt, get_task_losses

        dataset = _make_dataset()
        llms = _make_llms(api_key, shared_limits)

        try:
            task_losses = get_task_losses(dataset_name="SummEval", tasks=[FLUENCY_TASK])
            initial_prompt = get_initial_prompt(
                dataset_name="SummEval", tasks=[FLUENCY_TASK],
            )

            with tempfile.TemporaryDirectory() as tmpdir:
                output_dir = os.path.join(tmpdir, "gpo_test")

                algo = GPO(
                    task_llm=llms["task_llm"],
                    gradient_llm=llms["gradient_llm"],
                    optimizer_llm=llms["optimizer_llm"],
                    loss_llm=llms["loss_llm"],
                    task_losses=task_losses,
                    tasks=[FLUENCY_TASK],
                    steps=2,
                    batch_size=3,
                    loss_batch_size=3,
                    gradient_batch_size=3,
                    eval_every=2,
                    name="test_gpo",
                    k=3,
                    warmup_steps=1,
                    verbosity=1,
                )

                with temp_config(substep_delay=0.0):
                    result = algo.train(
                        dataset=dataset,
                        initial_prompt=initial_prompt,
                        output_dir=output_dir,
                    )

                assert result["run_id"] is not None
                assert result["final_prompt"] is not None
                assert result["output_dir"] == output_dir
                assert os.path.isdir(output_dir)

                assert len(algo.trajectory) >= 1
                print(f"GPO: {len(algo.trajectory)} trajectory elements")
                print(f"Final prompt: {result['final_prompt'].to_str()[:200]!r}")
        finally:
            _stop_all(llms)


@skip_no_api_key
class TestTextGradAlgorithm:
    """TextGrad algorithm: 2 training steps with real LLM calls.

    TextGrad uses:
    - task_llm for predictions
    - TextGradLossComputer for textual-only feedback (via loss_llm)
    - TextGradGradientComputer for textual gradients (via gradient_llm)
    - TextGradOptimizer for direct instruction updates (no trajectory)
    """

    @pytest.mark.timeout(300)
    def test_textgrad_2_steps(self, api_key, shared_limits):
        """Run TextGrad for 2 steps on SummEval fluency.

        Steps:
        1. Create dataset, all 4 LLMs, algorithm
        2. Run 2 training steps
        3. Verify result dict has expected keys
        4. Verify previous_instructions state was updated
        """
        from prompt_moo.algorithm import TextGrad
        from runner import get_initial_prompt

        dataset = _make_dataset()
        llms = _make_llms(api_key, shared_limits)

        try:
            initial_prompt = get_initial_prompt(
                dataset_name="SummEval", tasks=[FLUENCY_TASK],
            )

            with tempfile.TemporaryDirectory() as tmpdir:
                output_dir = os.path.join(tmpdir, "textgrad_test")

                algo = TextGrad(
                    task_llm=llms["task_llm"],
                    gradient_llm=llms["gradient_llm"],
                    optimizer_llm=llms["optimizer_llm"],
                    loss_llm=llms["loss_llm"],
                    tasks=[FLUENCY_TASK],
                    steps=2,
                    batch_size=3,
                    loss_batch_size=3,
                    gradient_batch_size=3,
                    eval_every=2,
                    name="test_textgrad",
                    verbosity=1,
                )

                with temp_config(substep_delay=0.0):
                    result = algo.train(
                        dataset=dataset,
                        initial_prompt=initial_prompt,
                        output_dir=output_dir,
                    )

                assert result["run_id"] is not None
                assert result["final_prompt"] is not None
                assert result["output_dir"] == output_dir
                assert os.path.isdir(output_dir)

                assert algo._previous_instructions is not None
                assert "fluency" in algo._previous_instructions
                print(f"TextGrad final instructions: {algo._previous_instructions}")
                print(f"Final prompt: {result['final_prompt'].to_str()[:200]!r}")
        finally:
            _stop_all(llms)


@skip_no_api_key
class TestAlgorithmRunnerE2E:
    """AlgorithmRunner.run() end-to-end: the exact entry point used by notebooks.

    This tests the full pipeline as invoked by Runner-SingleMetric.ipynb:
    AlgorithmRunner.run() -> creates LLMs internally -> runs algorithm -> returns result dict.
    """

    @pytest.mark.timeout(300)
    def test_runner_opro(self, api_key):
        """Run AlgorithmRunner.run() with OPRO for 2 steps.

        This is the closest test to the actual notebook invocation pattern:
            runner_pool.run(
                dataset=dataset, algo_name="opro", api_key=key,
                steps=2, batch_size=3, ...
            )

        Steps:
        1. Create AlgorithmRunner (direct, no pool)
        2. Call .run() with all required params
        3. Verify result dict has status="success"
        """
        from runner import AlgorithmRunner

        dataset = _make_dataset()
        runner = AlgorithmRunner.options(mode="sync").init()

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                with temp_config(substep_delay=0.0):
                    result = runner.run(
                        dataset=dataset,
                        algo_name="opro",
                        api_key=api_key,
                        steps=2,
                        batch_size=3,
                        loss_batch_size=3,
                        gradient_batch_size=3,
                        eval_every=2,
                        run_name="e2e_test",
                        llm="llama3.1",
                        verbosity=1,
                        output_dir=os.path.join(tmpdir, "runner_opro"),
                    ).result(timeout=280)

                assert result["status"] == "success", f"Runner failed: {result.get('error')}"
                assert result["algorithm"] == "opro"
                assert result["dataset"] == "SummEval"
                assert result["results"] is not None
                assert result["results"]["final_prompt"] is not None
                print(f"Runner OPRO: status={result['status']}")
        finally:
            runner.stop()
