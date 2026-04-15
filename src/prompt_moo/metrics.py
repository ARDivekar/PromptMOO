"""
Metric classes for prompt optimization.

Each Metric subclass is the single source of truth for:
- ``name``: canonical string identifier (instance field with subclass default)
- ``optimization_direction``: ``"maximize"`` or ``"minimize"`` (ClassVar)
- ``compute()``: classmethod that computes the metric from y_true/y_pred arrays
- ``.display_score``: formats the instance's ``value`` for display
- ``normalized_score``: direction-aware score (higher is always better)

Usage::

    from prompt_moo.metrics import Metric, Accuracy

    # Resolve by name (Registry lookup):
    metric_cls = Metric.get_subclass("accuracy")
    value = metric_cls.compute(y_true=y_true, y_pred=y_pred)
    metric = metric_cls(value=value)
    print(metric.display_score)  # "70.8"

    # Or construct directly:
    metric = Accuracy(value=0.7083)
    print(metric.display_score)  # "70.8"
"""

from typing import Any, ClassVar, List, Literal, Tuple

import numpy as np
import pandas as pd
from morphic import Registry, Typed, classproperty, validate
from pydantic import ConfigDict
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import f1_score as sklearn_f1_score


class Metric(Typed, Registry):
    """Base metric class.

    Subclasses must declare ``optimization_direction`` and
    ``display_score_range`` as ClassVars, optionally declare ``aliases``,
    override ``name`` and ``display_decimals`` with defaults, and implement
    ``compute()``.

    Attributes:
        name: Canonical metric name (e.g. ``"accuracy"``).  Mandatory on
            the base class; each subclass overrides with its own default.
        value: The computed metric score.
        display_decimals: Number of decimal places used by ``.display_score``.
            This is a metric-level property, NOT settable by algorithms.
            Each subclass sets its own default.  For [0, 1] metrics displayed
            as percentages, this controls decimal places *after* the percentage
            conversion (e.g. ``display_decimals=1`` formats 0.7083 as ``"70.8"``).
    """

    _allow_subclass_override: ClassVar[bool] = True

    model_config = ConfigDict(
        extra="ignore",
    )

    optimization_direction: ClassVar[Literal["maximize", "minimize"]]

    """Score range shown to the optimizer LLM after ``.display_score`` formatting.

    Must reflect the actual output of ``.display_score``.  Percentage
    metrics (Accuracy, F1) that multiply by 100 declare ``(0, 100)``.
    Raw-value metrics declare their natural range.  Use ``float('inf')``
    for unbounded upper limits.

    Mandatory on the base class (no default); each subclass must override.
    """
    display_score_range: ClassVar[Tuple[float, float]]

    name: str
    value: float = 0.0
    display_decimals: int

    @classmethod
    @validate
    def compute(
        cls,
        *,
        y_true: List[Any],
        y_pred: List[Any],
        **kwargs: Any,
    ) -> float:
        """Compute the metric from ground-truth and prediction arrays.

        Args:
            y_true: Ground-truth values.
            y_pred: Predicted values.
            **kwargs: Metric-specific parameters (e.g. ``num_classes`` for LCE).
                Subclasses that do not need extra params absorb them via **kwargs.

        Returns:
            Scalar metric value.

        Raises:
            NotImplementedError: If the subclass has not implemented this method.
        """
        raise NotImplementedError(f"{cls.__name__}.compute() is not implemented")

    @property
    def display_score(self) -> str:
        """Format ``self.value`` for display.

        Default: round to ``self.display_decimals`` decimal places.  Subclasses
        that produce [0, 1] scores override to show percentage form.
        """
        rounded: float = round(self.value, self.display_decimals)
        if self.display_decimals == 0:
            return str(int(rounded))
        return str(rounded)

    @property
    def normalized_score(self) -> float:
        """Return value oriented so that higher is always better."""
        if self.optimization_direction == "minimize":
            return -self.value
        return self.value

    @classproperty
    def display_direction(cls) -> str:
        """Human-readable direction word: ``"higher"`` or ``"lower"``.

        Computed from ``optimization_direction``.  Access as an attribute
        (no parentheses): ``metric_cls.display_direction``.

        Returns:
            ``"higher"`` for maximize metrics, ``"lower"`` for minimize.
        """
        if cls.optimization_direction == "minimize":
            return "lower"
        return "higher"


class Accuracy(Metric):
    """Exact-match accuracy (fraction correct).

    Values are in [0, 1].  ``.display_score`` formats as a percentage
    (e.g. 0.7083 -> ``"70.8"`` at ``display_decimals=1``, ``"71"`` at ``display_decimals=0``).
    """

    aliases: ClassVar[List[str]] = ["acc"]

    optimization_direction: ClassVar[Literal["maximize", "minimize"]] = "maximize"
    display_score_range: ClassVar[Tuple[float, float]] = (0, 100)

    name: str = "accuracy"
    display_decimals: int = 1

    @classmethod
    @validate
    def compute(cls, *, y_true: List[Any], y_pred: List[Any], **kwargs: Any) -> float:
        """Compute exact-match accuracy.

        Args:
            y_true: Ground-truth values.
            y_pred: Predicted values.

        Returns:
            Fraction of predictions that exactly match ground truth.

        Raises:
            ValueError: If y_true and y_pred have different lengths.
        """
        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)
        if len(y_true_arr) != len(y_pred_arr):
            raise ValueError("y_true and y_pred must have the same length")
        if len(y_true_arr) == 0:
            return 0.0
        return float((y_true_arr == y_pred_arr).mean())

    @property
    def display_score(self) -> str:
        """Format as percentage with ``self.display_decimals`` decimal places."""
        rounded: float = round(self.value * 100, self.display_decimals)
        if self.display_decimals == 0:
            return str(int(rounded))
        return str(rounded)


class F1(Metric):
    """Macro-averaged F1 score.

    Values are in [0, 1].  ``.display_score`` formats as a percentage.
    NaN values in inputs are filtered out.
    """

    optimization_direction: ClassVar[Literal["maximize", "minimize"]] = "maximize"
    display_score_range: ClassVar[Tuple[float, float]] = (0, 100)

    name: str = "f1"
    display_decimals: int = 1

    @classmethod
    @validate
    def compute(cls, *, y_true: List[Any], y_pred: List[Any], **kwargs: Any) -> float:
        """Compute macro-averaged F1, filtering NaN values.

        Args:
            y_true: Ground-truth values (may contain NaN).
            y_pred: Predicted values (may contain NaN).

        Returns:
            Macro-averaged F1 score.
        """
        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)
        mask = ~pd.isna(y_true_arr) & ~pd.isna(y_pred_arr)
        y_true_arr = y_true_arr[mask]
        y_pred_arr = y_pred_arr[mask]
        if len(y_true_arr) == 0:
            return 0.0
        return float(
            sklearn_f1_score(y_true=y_true_arr, y_pred=y_pred_arr, average="macro")
        )

    @property
    def display_score(self) -> str:
        """Format as percentage with ``self.display_decimals`` decimal places."""
        rounded: float = round(self.value * 100, self.display_decimals)
        if self.display_decimals == 0:
            return str(int(rounded))
        return str(rounded)


class Precision(Metric):
    """Macro-averaged precision.

    Values are in [0, 1].  ``.display_score`` formats as a percentage.
    NaN values in inputs are filtered out.
    """

    optimization_direction: ClassVar[Literal["maximize", "minimize"]] = "maximize"
    display_score_range: ClassVar[Tuple[float, float]] = (0, 100)

    name: str = "precision"
    display_decimals: int = 1

    @classmethod
    @validate
    def compute(cls, *, y_true: List[Any], y_pred: List[Any], **kwargs: Any) -> float:
        """Compute macro-averaged precision, filtering NaN values.

        Args:
            y_true: Ground-truth values (may contain NaN).
            y_pred: Predicted values (may contain NaN).

        Returns:
            Macro-averaged precision.
        """
        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)
        mask = ~pd.isna(y_true_arr) & ~pd.isna(y_pred_arr)
        y_true_arr = y_true_arr[mask]
        y_pred_arr = y_pred_arr[mask]
        if len(y_true_arr) == 0:
            return 0.0
        classes = np.unique(np.concatenate([y_true_arr, y_pred_arr]))
        precisions = []
        for c in classes:
            tp = np.sum((y_true_arr == c) & (y_pred_arr == c))
            fp = np.sum((y_true_arr != c) & (y_pred_arr == c))
            precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
        return float(np.mean(precisions))

    @property
    def display_score(self) -> str:
        """Format as percentage with ``self.display_decimals`` decimal places."""
        rounded = round(self.value * 100, self.display_decimals)
        if self.display_decimals == 0:
            return str(int(rounded))
        return str(rounded)


class Recall(Metric):
    """Macro-averaged recall.

    Values are in [0, 1].  ``.display_score`` formats as a percentage.
    NaN values in inputs are filtered out.
    """

    optimization_direction: ClassVar[Literal["maximize", "minimize"]] = "maximize"
    display_score_range: ClassVar[Tuple[float, float]] = (0, 100)

    name: str = "recall"
    display_decimals: int = 1

    @classmethod
    @validate
    def compute(cls, *, y_true: List[Any], y_pred: List[Any], **kwargs: Any) -> float:
        """Compute macro-averaged recall, filtering NaN values.

        Args:
            y_true: Ground-truth values (may contain NaN).
            y_pred: Predicted values (may contain NaN).

        Returns:
            Macro-averaged recall.
        """
        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)
        mask = ~pd.isna(y_true_arr) & ~pd.isna(y_pred_arr)
        y_true_arr = y_true_arr[mask]
        y_pred_arr = y_pred_arr[mask]
        if len(y_true_arr) == 0:
            return 0.0
        classes = np.unique(np.concatenate([y_true_arr, y_pred_arr]))
        recalls = []
        for c in classes:
            tp = np.sum((y_true_arr == c) & (y_pred_arr == c))
            fn = np.sum((y_true_arr == c) & (y_pred_arr != c))
            recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        return float(np.mean(recalls))

    @property
    def display_score(self) -> str:
        """Format as percentage with ``self.display_decimals`` decimal places."""
        rounded = round(self.value * 100, self.display_decimals)
        if self.display_decimals == 0:
            return str(int(rounded))
        return str(rounded)


class LCE(Metric):
    """Log cross-entropy loss for ordinal predictions.

    Normalizes predictions to probabilities by dividing by ``num_classes``
    (the number of ordinal classes in the task's scale, e.g. 5 for a 1-5
    scale, 4 for a 0-3 scale).

    ``num_classes`` is a required instance field with no default — callers
    must pass the dataset's per-task class count explicitly.

    Values are non-negative floats.  ``.display_score`` formats with
    3 decimal places by default (e.g. ``"0.342"``).
    """

    aliases: ClassVar[List[str]] = ["ce"]

    optimization_direction: ClassVar[Literal["maximize", "minimize"]] = "minimize"
    display_score_range: ClassVar[Tuple[float, float]] = (0, float("inf"))

    name: str = "lce"
    display_decimals: int = 3
    num_classes: int

    @classmethod
    @validate
    def compute(
        cls,
        *,
        y_true: List[Any],
        y_pred: List[Any],
        num_classes: int,
        **kwargs: Any,
    ) -> float:
        """Compute mean log cross-entropy from ordinal predictions.

        Args:
            y_true: Ground-truth values (present for interface consistency;
                unused in the current implementation).
            y_pred: Predicted ordinal values.
            num_classes: Number of ordinal classes (e.g. 5 for a 1-5 scale).
                Used to normalize predictions to probabilities.

        Returns:
            Mean negative log probability.

        Raises:
            ValueError: If num_classes is less than 1.
        """
        if num_classes < 1:
            raise ValueError(
                f"LCE.compute requires num_classes >= 1, got {num_classes}"
            )
        losses = []
        for _gt, pred in zip(y_true, y_pred):
            prob = pred / num_classes
            losses.append(-np.log(prob + 1e-12))
        if len(losses) == 0:
            return 0.0
        return float(np.mean(losses))


# ---------------------------------------------------------------------------
# Error-sensitive ordinal metrics
# ---------------------------------------------------------------------------


def _to_numeric_arrays(
    y_true: List[Any], y_pred: List[Any]
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert to float arrays, filtering NaN/non-numeric pairs.

    Returns matched-length arrays with only rows where both values
    are finite numbers.
    """
    yt: np.ndarray = pd.to_numeric(pd.Series(y_true), errors="coerce").to_numpy()
    yp: np.ndarray = pd.to_numeric(pd.Series(y_pred), errors="coerce").to_numpy()
    mask: np.ndarray = np.isfinite(yt) & np.isfinite(yp)
    return yt[mask], yp[mask]


class MAE(Metric):
    """Mean Absolute Error for ordinal predictions.

    Measures the average magnitude of errors between predicted and true
    values.  Unlike exact-match accuracy, MAE gives credit for being
    close: predicting 4 when the truth is 5 costs 1.0, whereas predicting
    1 costs 4.0.  This makes it the natural error metric for ordinal
    scales (Likert 1-5, etc.) where the distance between classes matters.

    Values are non-negative floats.  ``.display_score`` shows the raw
    value rounded to ``display_decimals`` places (e.g. ``"0.83"``).

    Reference: Poliak et al. (2021), "Error-Sensitive Evaluation for
    Ordinal Target Variables", Eval4NLP @ EMNLP.
    """

    optimization_direction: ClassVar[Literal["maximize", "minimize"]] = "minimize"
    display_score_range: ClassVar[Tuple[float, float]] = (0, float("inf"))

    name: str = "mae"
    display_decimals: int = 2

    @classmethod
    @validate
    def compute(cls, *, y_true: List[Any], y_pred: List[Any], **kwargs: Any) -> float:
        """Compute mean absolute error.

        Args:
            y_true: Ground-truth values (numeric).
            y_pred: Predicted values (numeric).

        Returns:
            Mean absolute difference between predictions and ground truth.
        """
        yt, yp = _to_numeric_arrays(y_true, y_pred)
        if len(yt) == 0:
            return 0.0
        return float(np.mean(np.abs(yt - yp)))


class OffByOne(Metric):
    """Fraction of predictions within 1 of the true ordinal value.

    For a 1-5 Likert scale, predicting 4 when GT is 5 is "off by one"
    and counts as correct under this metric.  Predicting 3 does not.
    This is a relaxed accuracy that rewards near-misses.

    Values are in [0, 1].  ``.display_score`` formats as a percentage.
    """

    aliases: ClassVar[List[str]] = ["off_by_1", "offby1"]

    optimization_direction: ClassVar[Literal["maximize", "minimize"]] = "maximize"
    display_score_range: ClassVar[Tuple[float, float]] = (0, 100)

    name: str = "off_by_one"
    display_decimals: int = 1

    @classmethod
    @validate
    def compute(cls, *, y_true: List[Any], y_pred: List[Any], **kwargs: Any) -> float:
        yt, yp = _to_numeric_arrays(y_true, y_pred)
        if len(yt) == 0:
            return 0.0
        return float(np.mean(np.abs(yt - yp) <= 1))

    @property
    def display_score(self) -> str:
        rounded: float = round(self.value * 100, self.display_decimals)
        if self.display_decimals == 0:
            return str(int(rounded))
        return str(rounded)


class OffByTwo(Metric):
    """Fraction of predictions within 2 of the true ordinal value.

    Values are in [0, 1].  ``.display_score`` formats as a percentage.
    """

    aliases: ClassVar[List[str]] = ["off_by_2", "offby2"]

    optimization_direction: ClassVar[Literal["maximize", "minimize"]] = "maximize"
    display_score_range: ClassVar[Tuple[float, float]] = (0, 100)

    name: str = "off_by_two"
    display_decimals: int = 1

    @classmethod
    @validate
    def compute(cls, *, y_true: List[Any], y_pred: List[Any], **kwargs: Any) -> float:
        yt, yp = _to_numeric_arrays(y_true, y_pred)
        if len(yt) == 0:
            return 0.0
        return float(np.mean(np.abs(yt - yp) <= 2))

    @property
    def display_score(self) -> str:
        rounded: float = round(self.value * 100, self.display_decimals)
        if self.display_decimals == 0:
            return str(int(rounded))
        return str(rounded)


class OffByThree(Metric):
    """Fraction of predictions within 3 of the true ordinal value.

    Values are in [0, 1].  ``.display_score`` formats as a percentage.
    """

    aliases: ClassVar[List[str]] = ["off_by_3", "offby3"]

    optimization_direction: ClassVar[Literal["maximize", "minimize"]] = "maximize"
    display_score_range: ClassVar[Tuple[float, float]] = (0, 100)

    name: str = "off_by_three"
    display_decimals: int = 1

    @classmethod
    @validate
    def compute(cls, *, y_true: List[Any], y_pred: List[Any], **kwargs: Any) -> float:
        yt, yp = _to_numeric_arrays(y_true, y_pred)
        if len(yt) == 0:
            return 0.0
        return float(np.mean(np.abs(yt - yp) <= 3))

    @property
    def display_score(self) -> str:
        rounded: float = round(self.value * 100, self.display_decimals)
        if self.display_decimals == 0:
            return str(int(rounded))
        return str(rounded)


class SpearmanCorrelation(Metric):
    """Spearman rank correlation coefficient.

    Measures the monotonic association between predicted and true values
    using rank ordering.  Suitable for ordinal data because it only
    requires that the values have a meaningful order, not that they are
    evenly spaced.  Sensitive to the magnitude of rank differences.

    Values are in [-1, 1].  Returns 0.0 when fewer than 3 valid pairs
    exist or when either array has zero variance (constant predictions).
    ``.display_score`` shows the raw value (e.g. ``"0.742"``).

    Reference: SummEval (Fabbri et al., TACL 2021) uses Spearman
    correlation for sample-level metric evaluation.
    """

    aliases: ClassVar[List[str]] = ["spearman"]

    optimization_direction: ClassVar[Literal["maximize", "minimize"]] = "maximize"
    display_score_range: ClassVar[Tuple[float, float]] = (-1, 1)

    name: str = "spearman_correlation"
    display_decimals: int = 3

    @classmethod
    @validate
    def compute(cls, *, y_true: List[Any], y_pred: List[Any], **kwargs: Any) -> float:
        """Compute Spearman rank correlation.

        Args:
            y_true: Ground-truth values (numeric/ordinal).
            y_pred: Predicted values (numeric/ordinal).

        Returns:
            Spearman's rho in [-1, 1], or 0.0 if insufficient data.
        """
        yt, yp = _to_numeric_arrays(y_true, y_pred)
        if len(yt) < 3:
            return 0.0
        if np.std(yt) == 0 or np.std(yp) == 0:
            return 0.0
        rho, _pvalue = spearmanr(yt, yp)
        if np.isnan(rho):
            return 0.0
        return float(rho)


class KendallTau(Metric):
    """Kendall's tau-b rank correlation coefficient.

    Counts concordant and discordant pairs to measure ordinal association.
    More robust than Spearman to tied data and outliers, making it well
    suited for Likert-scale predictions where many samples share the
    same value.  Uses the tau-b variant which adjusts for ties.

    Values are in [-1, 1].  Returns 0.0 when fewer than 3 valid pairs
    exist or when either array has zero variance.
    ``.display_score`` shows the raw value (e.g. ``"0.681"``).

    Reference: Lapata (2006), "Automatic Evaluation of Information
    Ordering: Kendall's Tau", Computational Linguistics.  Also
    recommended by NLG meta-evaluation analysis (ACL 2023) as the
    correlation least sensitive to score granularity.
    """

    aliases: ClassVar[List[str]] = ["kendall"]

    optimization_direction: ClassVar[Literal["maximize", "minimize"]] = "maximize"
    display_score_range: ClassVar[Tuple[float, float]] = (-1, 1)

    name: str = "kendall_tau"
    display_decimals: int = 3

    @classmethod
    @validate
    def compute(cls, *, y_true: List[Any], y_pred: List[Any], **kwargs: Any) -> float:
        """Compute Kendall's tau-b rank correlation.

        Args:
            y_true: Ground-truth values (numeric/ordinal).
            y_pred: Predicted values (numeric/ordinal).

        Returns:
            Kendall's tau-b in [-1, 1], or 0.0 if insufficient data.
        """
        yt, yp = _to_numeric_arrays(y_true, y_pred)
        if len(yt) < 3:
            return 0.0
        if np.std(yt) == 0 or np.std(yp) == 0:
            return 0.0
        tau, _pvalue = kendalltau(yt, yp)
        if np.isnan(tau):
            return 0.0
        return float(tau)


class PearsonCorrelation(Metric):
    """Pearson linear correlation coefficient.

    Measures the linear relationship between predicted and true values.
    Assumes approximately interval-scale data.  Widely used in NLG
    meta-evaluation despite being less theoretically appropriate for
    ordinal data than Spearman or Kendall.  Included because it is
    the standard metric in many evaluation benchmarks and has the best
    discriminative power for system-level comparisons (ACL 2023
    meta-evaluation analysis).

    Values are in [-1, 1].  Returns 0.0 when fewer than 3 valid pairs
    exist or when either array has zero variance.
    ``.display_score`` shows the raw value (e.g. ``"0.815"``).
    """

    aliases: ClassVar[List[str]] = ["pearson"]

    optimization_direction: ClassVar[Literal["maximize", "minimize"]] = "maximize"
    display_score_range: ClassVar[Tuple[float, float]] = (-1, 1)

    name: str = "pearson_correlation"
    display_decimals: int = 3

    @classmethod
    @validate
    def compute(cls, *, y_true: List[Any], y_pred: List[Any], **kwargs: Any) -> float:
        """Compute Pearson correlation coefficient.

        Args:
            y_true: Ground-truth values (numeric).
            y_pred: Predicted values (numeric).

        Returns:
            Pearson's r in [-1, 1], or 0.0 if insufficient data.
        """
        yt, yp = _to_numeric_arrays(y_true, y_pred)
        if len(yt) < 3:
            return 0.0
        if np.std(yt) == 0 or np.std(yp) == 0:
            return 0.0
        r, _pvalue = pearsonr(yt, yp)
        if np.isnan(r):
            return 0.0
        return float(r)
