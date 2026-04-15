"""
Prompt Trajectory: Tracks optimization history for algorithms like OPRO and GPO.

Each trajectory element stores the prompt instructions and the
``NumericFeedback`` objects that scored them (carrying their own
``optimization_direction``).  The ``PromptTrajectory`` container
maintains a heap ranked by a direction-aware, optionally task-weighted
metric derived from the stored feedbacks.

GPO's paper-faithful mode stores the **full** unbounded history (``k=None``)
and retrieves the ``k`` most semantically similar entries at query time via
``get_most_similar()``, using a sentence-transformer embedding model.
"""

import heapq
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union

import numpy as np
from morphic import Registry, Typed, validate
from pydantic import PrivateAttr, conint

from .data_structures import NumericFeedback


def _instructions_to_text(instructions: Union[str, Dict[str, str]]) -> str:
    """Flatten instructions into a single string for embedding.

    For multi-task dicts, concatenates ``task_name: instruction`` pairs
    separated by newlines so the embedding captures all task content.
    """
    if isinstance(instructions, str):
        return instructions
    return "\n".join(
        f"{task_name}: {instruction_text}"
        for task_name, instruction_text in instructions.items()
    )


class TrajectoryElement(Typed, Registry, ABC):
    """A single point in the optimization trajectory.

    Abstract base class — algorithm-specific subclasses
    (``OPROTrajectoryElement``, ``GPOTrajectoryElement``) must implement
    ``__str__`` to render the element in the format expected by their
    optimizer's meta-prompt.

    Attributes:
        instructions: The prompt instructions used at this trajectory step.
            Either a single string or a dict mapping task_name -> instruction.
        numeric_scores: Per-task numeric feedback objects.  Each
            ``NumericFeedback`` carries its own ``optimization_direction``,
            so the trajectory never needs to guess which way is "better".
    """

    _allow_subclass_override = True

    instructions: Union[str, Dict[str, str]]
    numeric_scores: Dict[str, List[NumericFeedback]]

    @abstractmethod
    def __str__(self) -> str:
        """Render for inclusion in optimizer meta-prompts.

        Subclasses must implement this to match their paper's trajectory
        format (e.g. OPRO uses ``Instruction: .../Score: N``, GPO uses
        ``Prompt: .../Score: N``).
        """
        pass

    @validate
    def ranking_key(
        self,
        *,
        metric_priority: List[str],
        task_weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, ...]:
        """Compute a lexicographic ranking key for this element.

        For each metric name in ``metric_priority`` (in order), computes
        the weighted average of direction-normalized scores across tasks.
        The result is a tuple suitable for Python's built-in tuple
        comparison (first element is most significant).

        Args:
            metric_priority: Ordered list of metric names.  The first
                metric is the primary sort key, the second is the
                tiebreaker, etc.
            task_weights: Optional mapping of task_name -> weight.
                Weights must sum to 1.  When ``None``, all tasks
                receive equal weight.

        Returns:
            Tuple of floats (one per metric in priority order), where
            higher is always better regardless of the original metric
            direction.
        """
        if len(self.numeric_scores) == 0:
            return tuple(0.0 for _ in metric_priority)

        task_names = list(self.numeric_scores.keys())
        if task_weights is None:
            n = len(task_names)
            weights = {tn: 1.0 / n for tn in task_names}
        else:
            weights = task_weights

        key_parts: List[float] = []
        for metric_name in metric_priority:
            weighted_sum = 0.0
            for task_name, feedbacks in self.numeric_scores.items():
                if task_name not in weights:
                    raise ValueError(
                        f"PromptTrajectoryEntry.sort_key: task {task_name!r} is present in "
                        f"numeric_scores but has no weight in task_weights. "
                        f"Provided weights for: {list(weights.keys())}. "
                        f"Tasks in numeric_scores: {list(self.numeric_scores.keys())}."
                    )
                w = weights[task_name]
                matching = [fb for fb in feedbacks if fb.metric_name == metric_name]
                if len(matching) > 0:
                    avg = sum(fb.normalized_score for fb in matching) / len(matching)
                    weighted_sum += w * avg
            key_parts.append(weighted_sum)

        return tuple(key_parts)

    def __lt__(self, other: "TrajectoryElement") -> bool:
        raise NotImplementedError(
            "TrajectoryElement comparison requires metric_priority context. "
            "Use PromptTrajectory to manage ordering."
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TrajectoryElement):
            return NotImplemented
        return id(self) == id(other)

    def __hash__(self) -> int:
        return id(self)


class PromptTrajectory(Typed):
    """Maintains a heap of trajectory elements for optimization history.

    When ``k`` is an integer, the heap retains only the ``k`` elements with
    the *highest* ranking keys (direction-aware, optionally weighted),
    discarding lower-ranked elements as new ones are pushed.  This is the
    behaviour used by OPRO (top-20 by score).

    When ``k`` is ``None``, the heap grows unbounded — every pushed element
    is retained.  This is the behaviour needed by GPO, which stores the
    full optimization history and retrieves from it at query time via
    ``get_most_similar()``.

    ``get_topk()`` returns elements sorted according to ``order``.

    Args:
        k: Maximum number of elements to retain, or ``None`` for unbounded
            storage.  **Has no default** — callers must set it explicitly.
        order: Sort order for ``get_topk()``.
            ``"worst_to_best"``: ascending (OPRO recency-bias pattern).
            ``"best_to_worst"``: descending.
        metric_priority: Ordered list of metric names for lexicographic
            ranking.  The first metric is the primary sort key.  Accepts
            a single string (converted to a one-element list internally).
        task_weights: Optional mapping of task_name -> weight (must sum
            to 1.0).  ``None`` means equal weight across all tasks.
        element_separator: String placed between consecutive elements in
            ``get_top_k_str()`` output.  Default ``"\\n---\\n"`` matches
            the GPO/OPRO papers' delimiter convention.
    """

    k: Optional[conint(ge=1)]
    order: Literal["worst_to_best", "best_to_worst"]
    metric_priority: Union[str, List[str]]
    task_weights: Optional[Dict[str, float]] = None
    element_separator: str = "\n---\n"

    _heap: List[Tuple[Tuple[float, ...], int, TrajectoryElement]] = PrivateAttr(
        default_factory=list
    )
    _push_counter: int = PrivateAttr(default=0)

    # Embedding storage: maps push_counter -> L2-normalized embedding vector.
    # Populated only when the caller passes embeddings via push().
    # Using a dict (keyed by the unique push_counter) avoids the parallel-list
    # desync problem that heapq rearrangement causes.
    _embedding_by_id: Dict[int, np.ndarray] = PrivateAttr(default_factory=dict)

    @classmethod
    def pre_initialize(cls, data: dict) -> None:
        mp = data.get("metric_priority")
        if isinstance(mp, str):
            data["metric_priority"] = [mp]

    def model_post_init(self, __context: Any) -> None:
        if self.task_weights is not None:
            total = sum(self.task_weights.values())
            if abs(total - 1.0) > 1e-6:
                raise ValueError(
                    f"task_weights must sum to 1.0, got {total:.6f}: {self.task_weights}"
                )

    def _ranking_key(self, element: TrajectoryElement) -> Tuple[float, ...]:
        """Compute the ranking key used for heap ordering."""
        return element.ranking_key(
            metric_priority=self.metric_priority,
            task_weights=self.task_weights,
        )

    @validate
    def push(
        self,
        element: TrajectoryElement,
        *,
        embedding: Optional[np.ndarray] = None,
    ) -> None:
        """Push a new element, optionally maintaining top-k by ranking key.

        When ``k`` is not ``None``, evicts the lowest-ranked element if
        the heap exceeds ``k``.  When ``k`` is ``None``, the heap grows
        unbounded.

        Args:
            element: The trajectory element to store.
            embedding: Optional L2-normalized embedding vector for this
                element's instructions.  Required for ``get_most_similar()``
                to work.  When ``None``, similarity retrieval will skip
                this element.
        """
        found_metrics: Set = {
            fb.metric_name for fbs in element.numeric_scores.values() for fb in fbs
        }
        missing: Set = set(self.metric_priority) - found_metrics
        if len(missing) > 0 and len(found_metrics) > 0:
            raise ValueError(
                f"metric_priority references metrics {sorted(missing)} "
                f"not found in element. Available metrics: {sorted(found_metrics)}. "
                f"Set metric_priority to match the metrics your loss computer produces."
            )

        key: Tuple[float, ...] = self._ranking_key(element)
        self._push_counter += 1
        push_id = self._push_counter
        heapq.heappush(self._heap, (key, push_id, element))

        if embedding is not None:
            self._embedding_by_id[push_id] = embedding

        if self.k is not None and len(self._heap) > self.k:
            evicted = heapq.heappop(self._heap)
            evicted_id = evicted[1]
            self._embedding_by_id.pop(evicted_id, None)

    @validate
    def get_topk(self, *, limit: Optional[int] = None) -> List[TrajectoryElement]:
        """Return elements sorted according to ``self.order``.

        ``"worst_to_best"``: ascending ranking key (worst first, best last).
        ``"best_to_worst"``: descending ranking key (best first, worst last).

        Args:
            limit: Maximum number of elements to return.  ``None`` returns all.
                When the heap is larger than ``limit``, the top-ranked elements
                (highest ranking keys) are selected first, then sorted according
                to ``self.order``.
        """
        ascending = self.order == "worst_to_best"

        if limit is not None and len(self._heap) > limit:
            top_items = heapq.nlargest(limit, self._heap, key=lambda x: x[0])
            sorted_items = sorted(top_items, key=lambda x: x[0], reverse=not ascending)
        else:
            sorted_items = sorted(self._heap, key=lambda x: x[0], reverse=not ascending)

        return [elem for _key, _cnt, elem in sorted_items]

    @validate
    def get_most_similar(
        self,
        *,
        query_embedding: np.ndarray,
        limit: int,
    ) -> List[TrajectoryElement]:
        """Retrieve the ``limit`` most semantically similar elements.

        Computes cosine similarity between ``query_embedding`` and the stored
        embeddings (both must be L2-normalized, so similarity = dot product).
        Returns elements sorted in **ascending** similarity order: least
        similar first, most similar last.  This matches the GPO paper's
        meta-prompt structure where the most relevant prompts appear closest
        to the generation instruction.

        Elements pushed without an embedding are skipped.

        Args:
            query_embedding: L2-normalized embedding of the current prompt.
            limit: Maximum number of elements to return.

        Returns:
            List of TrajectoryElement sorted by ascending similarity.

        Raises:
            ValueError: If no elements have embeddings.
        """
        if len(self._heap) == 0:
            return []

        scored: List[Tuple[float, int, TrajectoryElement]] = []
        for rank_key, push_idx, element in self._heap:
            emb = self._embedding_by_id.get(push_idx)
            if emb is None:
                continue
            similarity = float(np.dot(query_embedding, emb))
            scored.append((similarity, push_idx, element))

        if len(scored) == 0:
            return []

        scored.sort(key=lambda x: x[0], reverse=True)
        top_scored = scored[:limit]

        top_scored.sort(key=lambda x: x[0])
        return [elem for _sim, _idx, elem in top_scored]

    def __len__(self) -> int:
        return len(self._heap)

    def to_serializable_list(
        self,
        *,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Serialize trajectory elements for observability logging.

        Each element is converted to a plain dict with ``instructions``,
        ``numeric_scores`` (model_dumped), and ``ranking_key``.  The
        algorithm calls this and passes the result to
        ``observer.record()`` — the observer never touches trajectory
        internals.

        Args:
            limit: Maximum number of elements.  ``None`` returns all.

        Returns:
            List of JSON-serializable dicts, one per trajectory element.
        """
        return [
            {
                "instructions": elem.instructions,
                "numeric_scores": {
                    task_name: [fb.model_dump() for fb in feedbacks]
                    for task_name, feedbacks in elem.numeric_scores.items()
                },
                "ranking_key": list(self._ranking_key(elem)),
            }
            for elem in self.get_topk(limit=limit)
        ]

    def get_top_k_str(
        self,
        *,
        limit: Optional[int] = None,
        separator: Optional[str] = None,
    ) -> str:
        """Joined ``str()`` of elements in ``get_topk()`` order.

        Args:
            limit: Maximum number of elements.  ``None`` returns all.
            separator: String placed between consecutive elements.
                When ``None``, uses ``self.element_separator``.
        """
        sep = separator if separator is not None else self.element_separator
        return sep.join([str(e) for e in self.get_topk(limit=limit)])
