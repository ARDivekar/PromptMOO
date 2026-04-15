"""
Unit tests for typed metric output classes, Metrics, and ExptEvaluator.

Migrated from test/test_analysis.py (the old test directory).
"""

import os

import numpy as np
import pandas as pd
import pytest

from prompt_moo.data_structures import (
    AlgoMetricSeries,
    ExptMetricReport,
    StepMetricResult,
)
from prompt_moo.context_manager import SingleRunContext
from prompt_moo.analysis import ExptEvaluator, Accuracy, F1


DATASET_CONFIG = {
    "prompt_prefix": "Test prefix",
    "task_output_formats": {
        "fluency": "An integer between 1 to 5",
        "coherence": "An integer between 1 to 5",
    },
    "task_losses": {
        "fluency": "accuracy",
        "coherence": "accuracy",
    },
}

TASKS = ("fluency", "coherence")


def _make_eval_parquet(directory: str, step: int, accuracy: float) -> str:
    """Create synthetic eval_step_N.parquet file with controlled accuracy."""
    n_samples = 20
    correct_count = int(n_samples * accuracy)

    gt_fluency = [1] * n_samples
    pred_fluency = [1] * correct_count + [2] * (n_samples - correct_count)

    gt_coherence = [3] * n_samples
    pred_coherence = [3] * correct_count + [4] * (n_samples - correct_count)

    df = pd.DataFrame(
        {
            "gt_fluency": gt_fluency,
            "pred_fluency": pred_fluency,
            "gt_coherence": gt_coherence,
            "pred_coherence": pred_coherence,
        }
    )

    path = os.path.join(directory, f"eval_step_{step}.parquet")
    df.to_parquet(path, index=False)
    return path


class TestMetrics:
    def test_accuracy_basic(self):
        metric = Accuracy.of()
        y_true = [1, 1, 0, 0]
        y_pred = [1, 0, 0, 0]
        assert metric.compute(y_true=y_true, y_pred=y_pred) == pytest.approx(0.75)

    def test_accuracy_length_mismatch(self):
        metric = Accuracy.of()
        with pytest.raises(ValueError):
            metric.compute(y_true=[1, 2], y_pred=[1])

    def test_f1_macro_basic(self):
        metric = F1.of()
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 1, 0, 2, 2]
        f1 = metric.compute(y_true=y_true, y_pred=y_pred)
        assert isinstance(f1, float)
        assert 0.0 <= f1 <= 1.0

    def test_f1_handles_nan(self):
        metric = F1.of()
        y_true = [1, 1, np.nan, 0]
        y_pred = [1, 0, 1, np.nan]
        f1 = metric.compute(y_true=y_true, y_pred=y_pred)
        assert isinstance(f1, float)
        assert 0.0 <= f1 <= 1.0


class TestStepMetricResult:
    def test_creation(self):
        result = StepMetricResult(
            step=5,
            split="eval",
            metric_name="accuracy",
            task_values={"fluency": 0.9},
        )
        assert result.step == 5
        assert result.task_values["fluency"] == 0.9

    def test_immutability(self):
        result = StepMetricResult(
            step=0,
            split="eval",
            metric_name="accuracy",
            task_values={"fluency": 0.5},
        )
        with pytest.raises(Exception):
            result.step = 10


class TestAlgoMetricSeries:
    def test_creation_with_steps(self):
        steps = [
            StepMetricResult(
                step=i * 5,
                split="eval",
                metric_name="accuracy",
                task_values={"fluency": 0.5 + i * 0.1},
            )
            for i in range(3)
        ]

        series = AlgoMetricSeries(
            algo_name="GPO",
            run_ctx=None,
            split="eval",
            metric_name="accuracy",
            steps=steps,
        )

        assert series.algo_name == "GPO"
        assert len(series.steps) == 3


class TestExptMetricReport:
    def test_creation(self):
        algo_series = AlgoMetricSeries(
            algo_name="GPO",
            run_ctx=None,
            split="eval",
            metric_name="accuracy",
            steps=[],
        )

        report = ExptMetricReport(
            expt_ctx=None,
            split="eval",
            metric_name="accuracy",
            algo_reports={"GPO_unique": algo_series},
        )

        assert "GPO_unique" in report.algo_reports


class TestComputeAllMetrics:
    def test_compute_all_metrics_structure(self, tmp_path):
        algo_dir = tmp_path / "AlgoRun"
        algo_dir.mkdir()

        _make_eval_parquet(str(algo_dir), step=0, accuracy=0.8)

        run_ctx = SingleRunContext.of(
            algo_name="TestAlgo",
            run_id="run001",
            unique_id="TestAlgo_abc123",
            run_dir=str(algo_dir),
            dataset_config=DATASET_CONFIG,
            prompts_dir=str(algo_dir / "prompts"),
            logs_path=str(algo_dir / "run_logs"),
            summary_path=str(algo_dir / "run_summary.json"),
        )

        parquet_path = os.path.join(str(algo_dir), "eval_step_0.parquet")

        result = ExptEvaluator.compute_all_metrics(
            parquet_path=parquet_path,
            tasks=TASKS,
            run_ctx=run_ctx,
            step=0,
            split="eval",
            metrics=["accuracy", "f1", "precision", "recall"],
        )

        assert result.unique_id == run_ctx.unique_id
        assert result.algo_name == run_ctx.algo_name
        assert len(result.task_metrics) == 2

        for task_metric in result.task_metrics:
            assert task_metric.accuracy is not None
            assert 0.0 <= task_metric.f1 <= 1.0


class TestGenerateSingleRunDF:
    def test_dataframe_generation(self, tmp_path):
        algo_dir = tmp_path / "AlgoRun"
        algo_dir.mkdir()

        _make_eval_parquet(str(algo_dir), step=0, accuracy=0.5)
        _make_eval_parquet(str(algo_dir), step=5, accuracy=0.7)

        run_ctx = SingleRunContext.of(
            algo_name="Algo",
            run_id="r1",
            unique_id="Algo_hash123",
            run_dir=str(algo_dir),
            dataset_config=DATASET_CONFIG,
            prompts_dir=str(algo_dir / "prompts"),
            logs_path=str(algo_dir / "run_logs"),
            summary_path=str(algo_dir / "run_summary.json"),
        )

        df = ExptEvaluator.generate_single_run_df(
            run_ctx=run_ctx,
            split="eval",
            k=5,
            metrics=["accuracy", "f1", "precision", "recall"],
        )

        assert len(df) == 4

        assert set(df.columns) == {
            "unique_id",
            "algo_name",
            "step",
            "split",
            "task_name",
            "accuracy",
            "f1",
            "precision",
            "recall",
        }

        assert df["unique_id"].iloc[0] == run_ctx.unique_id
        assert df["algo_name"].iloc[0] == run_ctx.algo_name

        assert sorted(df["step"].unique().tolist()) == [0, 5]
