"""
Prompt Trajectory: Tracks optimization history for algorithms like OPRO and GPO.
"""

import heapq
import json
from typing import Any, Dict, List, Optional, Tuple, Union

from morphic import Typed
from pydantic import PrivateAttr, conint


class TrajectoryElement(Typed):
    """Represents a single point in the optimization trajectory.

    For OPRO, this tracks:
    - The prompt instructions used
    - The scores achieved
    - The gradients computed
    - Loss function metadata
    """

    loss_fns: Dict[str, Any]                    # Map of task_name -> loss function config
    scores: Dict[str, Dict[str, float]]         # Map of task_name -> {metric: score}
    grads: Dict[str, str]                       # Map of task_name -> gradient text
    instructions: Union[str, Dict[str, str]]    # Prompt instructions

    def ranking_metric(self) -> float:
        """Compute the combined ranking metric across all tasks.

        For simplicity, we sum all scores. In OPRO, higher scores are better
        (we use negative loss values, so higher = better).

        Returns:
            Combined metric for ranking (higher is better)
        """
        if len(self.scores) == 0:
            return 0.0

        metric: float = 0.0
        for _task_name, task_scores in self.scores.items():
            # Sum all metrics for this task
            # In the new architecture, numeric feedback is typically negative loss
            # so higher values are better
            metric += sum(task_scores.values())

        return metric

    def __lt__(self, other: "TrajectoryElement") -> bool:
        return self.ranking_metric() < other.ranking_metric()

    def __le__(self, other: "TrajectoryElement") -> bool:
        return self.ranking_metric() <= other.ranking_metric()

    def __gt__(self, other: "TrajectoryElement") -> bool:
        return self.ranking_metric() > other.ranking_metric()

    def __ge__(self, other: "TrajectoryElement") -> bool:
        return self.ranking_metric() >= other.ranking_metric()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TrajectoryElement):
            return NotImplemented
        return self.ranking_metric() == other.ranking_metric()

    def __ne__(self, other: object) -> bool:
        if not isinstance(other, TrajectoryElement):
            return NotImplemented
        return self.ranking_metric() != other.ranking_metric()


class PromptTrajectory(Typed):
    """Maintains a top-k heap of trajectory elements for optimization history."""

    k: conint(ge=1)
    _heap: List[Tuple[float, TrajectoryElement]] = PrivateAttr(default_factory=list)

    def _compute_metric(self, element: TrajectoryElement) -> float:
        """Compute the value to push in the heap.

        Uses negative to simulate max-heap (higher rank_metric = better).

        Args:
            element: The trajectory element to compute metric for

        Returns:
            Negative ranking metric for min-heap ordering
        """
        return -element.ranking_metric()

    def push(self, element: TrajectoryElement) -> None:
        """Push a new element into the heap and maintain top-k.

        Args:
            element: The trajectory element to add
        """
        metric = self._compute_metric(element)
        heapq.heappush(self._heap, (metric, element))
        if len(self._heap) > self.k:
            heapq.heappop(self._heap)

    def get_topk(self) -> List[TrajectoryElement]:
        """Return elements sorted by descending ranking metric.

        Returns:
            List of top-k elements, best first
        """
        return [e for _, e in sorted(self._heap, key=lambda x: x[0])]

    def __len__(self) -> int:
        """Return the number of elements in the trajectory."""
        return len(self._heap)

    def get_top_k_str(self) -> str:
        """Return a string representation of the top-k elements.

        Returns:
            Newline-separated string of trajectory elements
        """
        return "\n".join([str(e) for e in self.get_topk()])

class OPROTrajectoryElement(TrajectoryElement):
    '''OPRO-specific trajectory element - uses score as the only gradient signal.'''

    def __str__(self) -> str:
        ## Only show scores and instructions
        lines = []
        lines.append("Instructions: " + (self.instructions if
            isinstance(self.instructions, str) else json.dumps(self.instructions)))
        lines.append("Scores:")
        for task_name, task_scores in self.scores.items():
            for metric_name, value in task_scores.items():
                lines.append(f"  {task_name} ({metric_name}): {value:.4f}")
        return "\n".join(lines)

class GPOTrajectoryElement(TrajectoryElement):
    """GPO-specific trajectory element - includes textual and numerical gradients."""

    text_grads: List[str] = []
    numerical_grads: Optional[str] = None 

    def __str__(self) -> str:
        lines = []
        lines.append("Instructions: " + (self.instructions if isinstance(self.instructions, str) 
                     else json.dumps(self.instructions)))
        lines.append("Scores:")
        for task_name, task_scores in self.scores.items():
            for metric, value in task_scores.items():
                lines.append(f"  {task_name} - {metric}: {value:.4f}")
        if self.text_grads:
            lines.append(f"Textual Gradients ({len(self.text_grads)}):")
            for i, grad in enumerate(self.text_grads, 1):
                lines.append(f"  {i}. {grad}...")  # Truncate for readability
        if self.numerical_grads:
            lines.append(f"Numerical Gradient: {self.numerical_grads}")
        return "\n".join(lines)