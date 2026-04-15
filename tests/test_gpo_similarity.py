"""
Tests for GPO semantic-similarity trajectory retrieval.

Verifies the core innovation of the GPO paper (Tang et al., AAAI 2025):
relevance-based retrieval via sentence-transformer embeddings.

Tests cover:
1. _instructions_to_text: string and dict flattening
2. PromptTrajectory.push with embeddings: storage, parallel list integrity
3. get_most_similar: ascending similarity order, limit, edge cases
4. get_most_similar vs get_topk: different elements selected
5. GPO._embed_instructions: lazy model loading, L2 normalization
6. GPOOptimizer.create_meta_prompt: dispatches on trajectory_strategy
7. GPO E2E with trajectory_strategy="importance" fallback
8. GPO E2E with trajectory_strategy="relevance" (real embeddings)
"""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prompt_moo.data_input import Dataset
from prompt_moo.data_structures import (
    Batch,
    DatasetSample,
    NumericFeedback,
    PredictionResult,
    Task,
)
from prompt_moo.metrics import Accuracy
from prompt_moo.prompt_template import PromptTemplate
from prompt_moo.algorithm.gpo import (
    GPOOptimizer,
    GPOTrajectoryElement,
)
from prompt_moo.prompt_trajectory import (
    PromptTrajectory,
    _instructions_to_text,
)

TASKS = [
    Task(
        task_name="fluency",
        task_description="Evaluate fluency",
        task_instruction="Rate from 1 to 5.",
        gt_col="fluency",
    ),
    Task(
        task_name="consistency",
        task_description="Evaluate consistency",
        task_instruction="Rate from 1 to 5.",
        gt_col="consistency",
    ),
]

SKELETON = (
    "Evaluate the summary. Output JSON with the requested task scores. "
    "Do NOT include reasoning or explanations."
)

TASK_LOSSES = {"fluency": "accuracy", "consistency": "accuracy"}

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
    )


def _make_feedback(*, task_name: str, value: float) -> NumericFeedback:
    return NumericFeedback(
        task_name=task_name,
        metric=Accuracy(value=value),
        aggregated_from_samples=[],
    )


def _make_element(
    *,
    instructions: Dict[str, str],
    fluency_score: float = 0.5,
    consistency_score: float = 0.5,
) -> GPOTrajectoryElement:
    return GPOTrajectoryElement(
        instructions=instructions,
        numeric_scores={
            "fluency": [_make_feedback(task_name="fluency", value=fluency_score)],
            "consistency": [
                _make_feedback(task_name="consistency", value=consistency_score)
            ],
        },
    )


def _random_normalized_embedding(*, dim: int = 8, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim).astype(np.float32)
    return vec / np.linalg.norm(vec)


# -----------------------------------------------------------------------
# 1. _instructions_to_text
# -----------------------------------------------------------------------


class TestInstructionsToText:
    """Verify flattening logic for embedding input."""

    def test_string_passthrough(self):
        assert _instructions_to_text("Rate fluency 1-5.") == "Rate fluency 1-5."

    def test_dict_concatenation(self):
        result = _instructions_to_text(
            {
                "fluency": "Rate fluency.",
                "consistency": "Rate consistency.",
            }
        )
        assert "fluency: Rate fluency." in result
        assert "consistency: Rate consistency." in result
        assert "\n" in result

    def test_empty_dict(self):
        assert _instructions_to_text({}) == ""

    def test_single_key_dict(self):
        result = _instructions_to_text({"fluency": "Rate it."})
        assert result == "fluency: Rate it."


# -----------------------------------------------------------------------
# 2. push() with embeddings
# -----------------------------------------------------------------------


class TestPushWithEmbeddings:
    """Verify that embeddings are stored parallel to heap entries."""

    def test_push_stores_embedding(self):
        traj = PromptTrajectory(
            k=None, order="worst_to_best", metric_priority="accuracy"
        )
        emb = _random_normalized_embedding(seed=42)
        elem = _make_element(instructions={"fluency": "v1", "consistency": "v1"})
        traj.push(elem, embedding=emb)

        assert len(traj) == 1
        assert len(traj._embedding_by_id) == 1
        stored_emb = list(traj._embedding_by_id.values())[0]
        assert np.allclose(stored_emb, emb)

    def test_push_without_embedding_stores_nothing(self):
        traj = PromptTrajectory(
            k=None, order="worst_to_best", metric_priority="accuracy"
        )
        elem = _make_element(instructions={"fluency": "v1", "consistency": "v1"})
        traj.push(elem)

        assert len(traj._embedding_by_id) == 0

    def test_mixed_with_and_without_embeddings(self):
        traj = PromptTrajectory(
            k=None, order="worst_to_best", metric_priority="accuracy"
        )
        traj.push(
            _make_element(instructions={"fluency": "v1", "consistency": "v1"}),
            embedding=_random_normalized_embedding(seed=1),
        )
        traj.push(
            _make_element(instructions={"fluency": "v2", "consistency": "v2"}),
        )
        traj.push(
            _make_element(instructions={"fluency": "v3", "consistency": "v3"}),
            embedding=_random_normalized_embedding(seed=3),
        )

        assert len(traj) == 3
        assert len(traj._embedding_by_id) == 2

    def test_unbounded_trajectory_grows(self):
        """k=None means no eviction, so all elements and embeddings are retained."""
        traj = PromptTrajectory(
            k=None, order="worst_to_best", metric_priority="accuracy"
        )
        for i in range(20):
            traj.push(
                _make_element(
                    instructions={"fluency": f"v{i}", "consistency": f"v{i}"},
                    fluency_score=i / 20.0,
                ),
                embedding=_random_normalized_embedding(seed=i),
            )
        assert len(traj) == 20
        assert len(traj._embedding_by_id) == 20


# -----------------------------------------------------------------------
# 3. get_most_similar: core retrieval logic
# -----------------------------------------------------------------------


class TestGetMostSimilar:
    """Verify cosine-similarity retrieval with controlled embeddings."""

    def _build_trajectory_with_known_embeddings(self) -> PromptTrajectory:
        """Create a trajectory with 5 elements whose embeddings have known
        pairwise similarities, so we can predict retrieval order exactly.

        Element embeddings (2D for simplicity):
          e0 = [1, 0]   (pure x-axis)
          e1 = [0.9, 0.44]  (close to x-axis, ~26°)
          e2 = [0.6, 0.8]   (45° region)
          e3 = [0, 1]   (pure y-axis, orthogonal to e0)
          e4 = [-0.7, 0.71] (opposite quadrant from e0)

        For query = [1, 0]:
          sim(e0) ≈ 1.0, sim(e1) ≈ 0.9, sim(e2) ≈ 0.6, sim(e3) = 0.0, sim(e4) ≈ -0.7
        """
        traj = PromptTrajectory(
            k=None, order="worst_to_best", metric_priority="accuracy"
        )

        raw_embeddings = [
            np.array([1.0, 0.0]),
            np.array([0.9, 0.44]),
            np.array([0.6, 0.8]),
            np.array([0.0, 1.0]),
            np.array([-0.7, 0.71]),
        ]
        instructions_list = [
            {"fluency": "pure_x", "consistency": "v0"},
            {"fluency": "near_x", "consistency": "v1"},
            {"fluency": "mid_angle", "consistency": "v2"},
            {"fluency": "pure_y", "consistency": "v3"},
            {"fluency": "opposite", "consistency": "v4"},
        ]
        scores = [0.5, 0.3, 0.9, 0.1, 0.7]

        for instr, emb_raw, score in zip(instructions_list, raw_embeddings, scores):
            emb_normalized = emb_raw / np.linalg.norm(emb_raw)
            traj.push(
                _make_element(instructions=instr, fluency_score=score),
                embedding=emb_normalized.astype(np.float32),
            )

        return traj

    def test_returns_ascending_similarity_order(self):
        """Most similar element should be LAST (closest to generation instruction)."""
        traj = self._build_trajectory_with_known_embeddings()
        query = np.array([1.0, 0.0], dtype=np.float32)

        results = traj.get_most_similar(query_embedding=query, limit=3)
        assert len(results) == 3

        result_instructions = [r.instructions["fluency"] for r in results]
        assert result_instructions[0] == "mid_angle"
        assert result_instructions[1] == "near_x"
        assert result_instructions[2] == "pure_x"

    def test_limit_respected(self):
        traj = self._build_trajectory_with_known_embeddings()
        query = np.array([1.0, 0.0], dtype=np.float32)

        results_2 = traj.get_most_similar(query_embedding=query, limit=2)
        assert len(results_2) == 2

        results_5 = traj.get_most_similar(query_embedding=query, limit=5)
        assert len(results_5) == 5

    def test_limit_larger_than_trajectory_returns_all(self):
        traj = self._build_trajectory_with_known_embeddings()
        query = np.array([1.0, 0.0], dtype=np.float32)

        results = traj.get_most_similar(query_embedding=query, limit=100)
        assert len(results) == 5

    def test_different_query_changes_order(self):
        """Querying with [0, 1] should rank pure_y highest (last)."""
        traj = self._build_trajectory_with_known_embeddings()
        query = np.array([0.0, 1.0], dtype=np.float32)

        results = traj.get_most_similar(query_embedding=query, limit=3)
        result_labels = [r.instructions["fluency"] for r in results]
        assert result_labels[-1] == "pure_y"

    def test_empty_trajectory_returns_empty(self):
        traj = PromptTrajectory(
            k=None, order="worst_to_best", metric_priority="accuracy"
        )
        query = np.array([1.0, 0.0], dtype=np.float32)
        assert len(traj.get_most_similar(query_embedding=query, limit=5)) == 0

    def test_no_embeddings_returns_empty(self):
        """Elements pushed without embeddings are skipped."""
        traj = PromptTrajectory(
            k=None, order="worst_to_best", metric_priority="accuracy"
        )
        traj.push(_make_element(instructions={"fluency": "v1", "consistency": "v1"}))
        traj.push(_make_element(instructions={"fluency": "v2", "consistency": "v2"}))

        query = np.array([1.0, 0.0], dtype=np.float32)
        assert len(traj.get_most_similar(query_embedding=query, limit=5)) == 0

    def test_mixed_embedded_and_non_embedded(self):
        """Only elements with embeddings participate in similarity retrieval."""
        traj = PromptTrajectory(
            k=None, order="worst_to_best", metric_priority="accuracy"
        )
        traj.push(
            _make_element(instructions={"fluency": "embedded", "consistency": "v1"}),
            embedding=np.array([1.0, 0.0], dtype=np.float32),
        )
        traj.push(
            _make_element(
                instructions={"fluency": "not_embedded", "consistency": "v2"}
            ),
        )

        query = np.array([1.0, 0.0], dtype=np.float32)
        results = traj.get_most_similar(query_embedding=query, limit=5)
        assert len(results) == 1
        assert results[0].instructions["fluency"] == "embedded"


# -----------------------------------------------------------------------
# 4. get_most_similar vs get_topk: different selection
# -----------------------------------------------------------------------


class TestSimilarityVsImportance:
    """Verify that relevance retrieval selects different elements than
    importance (score-based) retrieval — the core GPO vs OPRO difference."""

    def test_different_elements_selected(self):
        """Build a trajectory where the highest-scoring elements are NOT
        the most similar to the query, proving the two strategies diverge."""
        traj = PromptTrajectory(
            k=None, order="worst_to_best", metric_priority="accuracy"
        )

        traj.push(
            _make_element(
                instructions={
                    "fluency": "grammar and readability focus",
                    "consistency": "v_high",
                },
                fluency_score=0.95,
                consistency_score=0.95,
            ),
            embedding=np.array([0.0, 1.0], dtype=np.float32),
        )
        traj.push(
            _make_element(
                instructions={
                    "fluency": "writing quality assessment",
                    "consistency": "v_low",
                },
                fluency_score=0.10,
                consistency_score=0.10,
            ),
            embedding=np.array([1.0, 0.0], dtype=np.float32),
        )
        traj.push(
            _make_element(
                instructions={
                    "fluency": "rate text smoothness",
                    "consistency": "v_mid",
                },
                fluency_score=0.50,
                consistency_score=0.50,
            ),
            embedding=np.array([0.95, 0.31], dtype=np.float32)
            / np.linalg.norm(np.array([0.95, 0.31])),
        )

        query = np.array([1.0, 0.0], dtype=np.float32)
        by_similarity = traj.get_most_similar(query_embedding=query, limit=2)
        by_score = traj.get_topk(limit=2)

        sim_labels = {r.instructions["fluency"] for r in by_similarity}
        score_labels = {r.instructions["fluency"] for r in by_score}

        assert "writing quality assessment" in sim_labels, (
            "Similarity should select the element closest to query [1,0], "
            "which is 'writing quality assessment' with embedding [1,0]"
        )
        assert "grammar and readability focus" in score_labels, (
            "Importance should select the highest-scoring element (0.95)"
        )
        assert sim_labels != score_labels, (
            "The two strategies must select different elements for this test "
            "to be meaningful"
        )


# -----------------------------------------------------------------------
# 5. GPO._embed_instructions: real model, L2 normalization
# -----------------------------------------------------------------------


class TestGPOEmbedInstructions:
    """Test the embedding function with the real sentence-transformer model."""

    @pytest.fixture(scope="class")
    def gpo_instance(self):
        from prompt_moo.algorithm import GPO

        return GPO(
            task_llm=None,
            optimizer_llm=None,
            tasks=TASKS,
            steps=1,
            batch_size=2,
            loss_batch_size=2,
            gradient_batch_size=2,
            eval_every=1,
            name="embed_test",
            task_losses=TASK_LOSSES,
            trajectory_strategy="relevance",
        )

    def test_returns_1d_numpy_array(self, gpo_instance):
        emb = gpo_instance._embed_instructions(instructions="Rate from 1 to 5.")
        assert isinstance(emb, np.ndarray)
        assert emb.ndim == 1
        assert len(emb) > 0

    def test_is_l2_normalized(self, gpo_instance):
        emb = gpo_instance._embed_instructions(instructions="Rate from 1 to 5.")
        norm = float(np.linalg.norm(emb.astype(np.float32)))
        assert abs(norm - 1.0) < 1e-3, f"Expected unit norm, got {norm}"

    def test_dict_instructions_work(self, gpo_instance):
        emb = gpo_instance._embed_instructions(
            instructions={
                "fluency": "Rate fluency.",
                "consistency": "Rate consistency.",
            },
        )
        assert isinstance(emb, np.ndarray)
        assert emb.ndim == 1

    def test_similar_instructions_have_higher_similarity(self, gpo_instance):
        """Semantically similar instructions should have higher cosine similarity."""
        emb_a = gpo_instance._embed_instructions(
            instructions="Rate the fluency and readability of this text."
        )
        emb_b = gpo_instance._embed_instructions(
            instructions="Evaluate grammatical quality and reading flow."
        )
        emb_c = gpo_instance._embed_instructions(
            instructions="Count the number of blue objects in the room."
        )

        sim_ab = float(np.dot(emb_a, emb_b))
        sim_ac = float(np.dot(emb_a, emb_c))

        assert sim_ab > sim_ac, (
            f"Fluency-related instructions should be more similar to each other "
            f"(sim={sim_ab:.4f}) than to object counting (sim={sim_ac:.4f})"
        )

    def test_model_is_lazy_loaded(self):
        """The embedding model should NOT be loaded until first _embed call."""
        from prompt_moo.algorithm import GPO

        gpo = GPO(
            task_llm=None,
            optimizer_llm=None,
            tasks=TASKS,
            steps=1,
            batch_size=2,
            loss_batch_size=2,
            gradient_batch_size=2,
            eval_every=1,
            name="lazy_test",
            task_losses=TASK_LOSSES,
            trajectory_strategy="relevance",
        )
        assert gpo._embedding_model is None


# -----------------------------------------------------------------------
# 6. GPOOptimizer dispatches on trajectory_strategy
# -----------------------------------------------------------------------


class TestGPOOptimizerTrajectoryStrategy:
    """Verify that GPOOptimizer.create_meta_prompt uses the right retrieval."""

    def _build_trajectory_and_kwargs(
        self,
        *,
        strategy: str,
    ) -> Dict[str, Any]:
        from prompt_moo.algorithm.gpo import GPOOptimizer

        traj = PromptTrajectory(
            k=None, order="worst_to_best", metric_priority="accuracy"
        )

        emb_x = np.array([1.0, 0.0], dtype=np.float32)
        emb_y = np.array([0.0, 1.0], dtype=np.float32)

        traj.push(
            _make_element(
                instructions={"fluency": "x-axis instruction", "consistency": "v0"},
                fluency_score=0.2,
            ),
            embedding=emb_x,
        )
        traj.push(
            _make_element(
                instructions={"fluency": "y-axis instruction", "consistency": "v1"},
                fluency_score=0.9,
            ),
            embedding=emb_y,
        )

        def _mock_embed_fn(*, instructions):
            return np.array([1.0, 0.0], dtype=np.float32)

        kwargs: Dict[str, Any] = dict(
            gradients={},
            current_prompt=_make_prompt_template(),
            tasks=TASKS,
            batch=Batch(step=5, samples=[]),
            trajectory=traj,
            warmup_steps=0,
            total_steps=100,
            initial_step_size=25,
            final_step_size=5,
            use_textual_feedback=False,
            input_col_labels=INPUT_COL_LABELS,
            top_k_retrieve=2,
            trajectory_strategy=strategy,
            loss_functions={task.task_name: {"metric": "accuracy"} for task in TASKS},
        )
        if strategy == "relevance":
            kwargs["embed_fn"] = _mock_embed_fn

        return kwargs

    def test_relevance_meta_prompt_contains_similar_element(self):
        from prompt_moo.algorithm.gpo import GPOOptimizer

        optimizer = GPOOptimizer()
        kwargs = self._build_trajectory_and_kwargs(strategy="relevance")
        meta = optimizer.create_meta_prompt(**kwargs)
        assert "x-axis instruction" in meta, (
            "Relevance retrieval with query [1,0] should retrieve "
            "the x-axis element (embedding [1,0]) as most similar"
        )

    def test_importance_meta_prompt_contains_highest_scored(self):
        from prompt_moo.algorithm.gpo import GPOOptimizer

        optimizer = GPOOptimizer()
        kwargs = self._build_trajectory_and_kwargs(strategy="importance")
        meta = optimizer.create_meta_prompt(**kwargs)
        assert "y-axis instruction" in meta, (
            "Importance retrieval should retrieve the highest-scored "
            "element (0.9), which is y-axis instruction"
        )

    def test_relevance_requires_embed_fn(self):
        from prompt_moo.algorithm.gpo import GPOOptimizer

        optimizer = GPOOptimizer()
        kwargs = self._build_trajectory_and_kwargs(strategy="relevance")
        del kwargs["embed_fn"]
        with pytest.raises(ValueError, match="embed_fn"):
            optimizer.create_meta_prompt(**kwargs)


# -----------------------------------------------------------------------
# 7. GPO E2E with trajectory_strategy="importance" (no embedding model)
# -----------------------------------------------------------------------


class _MockDataset(Dataset):
    """Minimal Dataset subclass for testing without real data files."""

    _allow_subclass_override = True

    dataset_name: ClassVar[str] = "MockSummEval"
    train_size: ClassVar[int] = 4
    test_size: ClassVar[int] = 4
    input_cols: ClassVar[List[str]] = ["machine_summary", "text"]
    gt_cols: ClassVar[List[str]] = ["fluency", "consistency"]
    input_col_labels: ClassVar[Dict[str, str]] = INPUT_COL_LABELS
    task_output_formats: ClassVar[dict] = {
        "fluency": "1|2|3|4|5",
        "consistency": "1|2|3|4|5",
    }
    task_losses: ClassVar[Dict[str, str]] = TASK_LOSSES
    tasks: ClassVar[List[Any]] = TASKS

    @classmethod
    def setup(cls, base_dir: str):
        pass

    def train(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "machine_summary": "Cat sat.",
                    "text": "Cat story.",
                    "fluency": 4,
                    "consistency": 5,
                },
                {
                    "machine_summary": "Dog ran.",
                    "text": "Dog story.",
                    "fluency": 3,
                    "consistency": 2,
                },
                {
                    "machine_summary": "Bird flew.",
                    "text": "Bird story.",
                    "fluency": 5,
                    "consistency": 4,
                },
                {
                    "machine_summary": "Fish swam.",
                    "text": "Fish story.",
                    "fluency": 2,
                    "consistency": 3,
                },
            ]
        )

    def test(self) -> pd.DataFrame:
        return self.train()


class _MockLLMPool:
    def __init__(self, *, role: str):
        self._role = role

    def call_llm_batch(self, prompts, verbosity=0, validator=None, **kwargs):
        responses = []
        for prompt in prompts:
            resp = self._generate(prompt)
            if validator is not None:
                resp = validator(resp)
            responses.append(resp)
        return _MockFuture(responses)

    def _generate(self, prompt: str) -> str:
        if self._role == "task":
            return '{"fluency": 3, "consistency": 4}'
        if self._role == "optimizer":
            return json.dumps(
                {
                    "instructions": {
                        "fluency": "Improved fluency instruction.",
                        "consistency": "Improved consistency instruction.",
                    }
                }
            )
        return "Mock response"

    def stop(self):
        pass


class _MockFuture:
    def __init__(self, value):
        self._value = value

    def result(self, timeout=None):
        return self._value


class TestGPOE2EImportanceStrategy:
    """E2E test with trajectory_strategy='importance' — no embedding model needed."""

    @pytest.fixture()
    def output_dir(self):
        d = tempfile.mkdtemp(prefix="gpo_importance_")
        yield d
        shutil.rmtree(d, ignore_errors=True)

    def test_e2e_importance_runs_without_embedding_model(self, output_dir):
        from prompt_moo.algorithm import GPO
        from prompt_moo.config import temp_config

        dataset = _MockDataset(data_dir=".")
        task_llm = _MockLLMPool(role="task")
        optimizer_llm = _MockLLMPool(role="optimizer")

        with temp_config(substep_delay=0, verbosity=0):
            algo = GPO(
                task_llm=task_llm,
                optimizer_llm=optimizer_llm,
                tasks=TASKS,
                steps=3,
                batch_size=2,
                loss_batch_size=2,
                gradient_batch_size=2,
                eval_every=99,
                name="importance_e2e",
                task_losses=TASK_LOSSES,
                trajectory_strategy="importance",
                num_candidates=1,
                verbosity=0,
            )

            results = algo.train(
                dataset=dataset,
                initial_prompt=_make_prompt_template(),
                output_dir=output_dir,
            )

        assert results is not None
        assert algo._embedding_model is None, (
            "Embedding model should NOT be loaded when trajectory_strategy='importance'"
        )
        assert len(algo.trajectory) > 0


class TestGPOE2ERelevanceStrategy:
    """E2E test with trajectory_strategy='relevance' — uses real embedding model."""

    @pytest.fixture()
    def output_dir(self):
        d = tempfile.mkdtemp(prefix="gpo_relevance_")
        yield d
        shutil.rmtree(d, ignore_errors=True)

    def test_e2e_relevance_loads_embedding_model(self, output_dir):
        from prompt_moo.algorithm import GPO
        from prompt_moo.config import temp_config

        dataset = _MockDataset(data_dir=".")
        task_llm = _MockLLMPool(role="task")
        optimizer_llm = _MockLLMPool(role="optimizer")

        with temp_config(substep_delay=0, verbosity=0):
            algo = GPO(
                task_llm=task_llm,
                optimizer_llm=optimizer_llm,
                tasks=TASKS,
                steps=3,
                batch_size=2,
                loss_batch_size=2,
                gradient_batch_size=2,
                eval_every=99,
                name="relevance_e2e",
                task_losses=TASK_LOSSES,
                trajectory_strategy="relevance",
                num_candidates=1,
                verbosity=0,
            )

            results = algo.train(
                dataset=dataset,
                initial_prompt=_make_prompt_template(),
                output_dir=output_dir,
            )

        assert results is not None
        assert algo._embedding_model is not None, (
            "Embedding model SHOULD be loaded when trajectory_strategy='relevance'"
        )
        assert len(algo.trajectory) > 0

        all_have_embeddings = all(
            push_id in algo.trajectory._embedding_by_id
            for (_key, push_id, _elem) in algo.trajectory._heap
        )
        assert all_have_embeddings, (
            "Every trajectory element should have an embedding when strategy='relevance'"
        )

    def test_default_strategy_is_relevance(self):
        from prompt_moo.algorithm import GPO

        gpo = GPO(
            task_llm=None,
            optimizer_llm=None,
            tasks=TASKS,
            steps=1,
            batch_size=2,
            loss_batch_size=2,
            gradient_batch_size=2,
            eval_every=1,
            name="default_check",
            task_losses=TASK_LOSSES,
        )
        assert gpo.trajectory_strategy == "relevance"

    def test_default_embedding_model_is_mxbai(self):
        from prompt_moo.algorithm import GPO

        gpo = GPO(
            task_llm=None,
            optimizer_llm=None,
            tasks=TASKS,
            steps=1,
            batch_size=2,
            loss_batch_size=2,
            gradient_batch_size=2,
            eval_every=1,
            name="model_check",
            task_losses=TASK_LOSSES,
        )
        assert gpo.embedding_model_name == "mixedbread-ai/mxbai-embed-xsmall-v1"
