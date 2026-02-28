import os 
import pandas as pd 
import numpy as np

import hvplot.pandas  # required for hvplot
import holoviews as hv
hv.extension("bokeh")

from typing import Dict, List, Optional, Tuple, ClassVar 
from morphic import Typed, Registry 
from morphic.typed import format_exception_msg

from .data_structures import TaskMetricResult, StepMultiMetricResult
from .context_manager import ExptRunContext, SingleRunContext


class Analysis(Typed, Registry):
    _allow_subclass_override: ClassVar[bool] = True


####################### METRIC SUBCLASS #######################

class Metric(Analysis):
    """
    Base metric class.
    """
    name: str 
    def compute(self, y_true, y_pred) -> float:
        raise NotImplementedError

class Accuracy(Metric):
    """
    Accuracy metric.
    """
    name: str = "accuracy"
    def compute(self, y_true, y_pred) -> float:
        y_true =  np.array(y_true)
        y_pred = np.array(y_pred)

        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")

        if len(y_true) == 0:
            return 0.0

        return float((y_true == y_pred).mean())

class Precision(Metric):
    name: str = "precision"

    def compute(self, y_true, y_pred) -> float:
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        mask = ~pd.isna(y_true) & ~pd.isna(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        if len(y_true) == 0:
            return 0.0

        classes = np.unique(np.concatenate([y_true, y_pred]))
        precisions = []

        for cls in classes:
            tp = np.sum((y_true == cls) & (y_pred == cls))
            fp = np.sum((y_true != cls) & (y_pred == cls))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            precisions.append(precision)

        return float(np.mean(precisions))

class Recall(Metric):
    name: str = "recall"

    def compute(self, y_true, y_pred) -> float:
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        mask = ~pd.isna(y_true) & ~pd.isna(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        if len(y_true) == 0:
            return 0.0

        classes = np.unique(np.concatenate([y_true, y_pred]))
        recalls = []

        for cls in classes:
            tp = np.sum((y_true == cls) & (y_pred == cls))
            fn = np.sum((y_true == cls) & (y_pred != cls))

            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            recalls.append(recall)

        return float(np.mean(recalls))

class F1(Metric):
    name: str = "f1"
    average: str = "macro"

    def compute(self, y_true, y_pred) -> float:

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        mask = ~pd.isna(y_true) & ~pd.isna(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        if len(y_true) == 0:
            return 0.0

        classes = np.unique(np.concatenate([y_true, y_pred]))
        f1_scores = []

        for cls in classes:
            tp = np.sum((y_true == cls) & (y_pred == cls))
            fp = np.sum((y_true != cls) & (y_pred == cls))
            fn = np.sum((y_true == cls) & (y_pred != cls))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)

            f1_scores.append(f1)

        return float(np.mean(f1_scores))

###################### VISUALIZATION SUBCLASS ##############################

class Visualizer(Analysis):
    _allow_subclass_override: ClassVar[bool] = True

    def render(self) -> None:
        raise NotImplementedError


class LinePlot(Visualizer):
    """
    Line plot for a SINGLE RUN.

    Expects DataFrame from:
        Evaluator.generate_single_run_df(...)
    """

    df: pd.DataFrame
    metric: str = "accuracy"
    metric_colors: Optional[Dict[str, str]] = None
    title: Optional[str] = None

    def render(self):

        if self.df.empty:
            raise ValueError("Provided DataFrame is empty")

        if self.metric not in {"accuracy", "f1", "precision", "recall", "hypervolume"}:
            raise ValueError("metric must be 'accuracy' or 'f1'")

        plot = None
        tasks = self.df["task_name"].unique()

        for task in tasks:

            task_df = (
                self.df[self.df["task_name"] == task]
                .sort_values("step")
            )

            color = None
            if self.metric_colors:
                color = self.metric_colors.get(task)

            # Line (Curve)
            line = task_df.hvplot.line(
                x="step",
                y=self.metric,
                color=color,
                line_width=2,
                label=task.capitalize(),
            )

            # Markers (Scatter)
            markers = task_df.hvplot.scatter(
                x="step",
                y=self.metric,
                color=color,
                size=60,
            )

            combined = line * markers

            plot = combined if plot is None else (plot * combined)

        return plot.opts(
            title=self.title,
            width=600,
            height=420,
            xlabel="Step",
            ylabel=self.metric.capitalize(),
            ylim=(0, 1),
            legend_position="right",
        )

class ExptEvaluator(Typed, Registry):
    _allow_subclass_override: ClassVar[bool] = True

    @staticmethod
    def compute_all_metrics(
        *,
        parquet_path: str,
        tasks: Tuple[str, ...],
        run_ctx: SingleRunContext,
        step: int,
        split: str,
        metrics: List[str],
    ) -> StepMultiMetricResult:
        """
        Compute selected metrics for all tasks in a step.
        Returns structured StepMultiMetricResult.
        """

        try:
            eval_df = pd.read_parquet(parquet_path, engine="pyarrow")
        except Exception as e:
            raise IOError(
                f"Failed to read evaluation parquet at {parquet_path!r}:\n"
                f"{format_exception_msg(e)}"
            ) from e
        task_results: List[TaskMetricResult] = []

        # Metric registry
        metric_map = {
            "accuracy": Accuracy.of(),
            "f1": F1.of(),
            "precision": Precision.of(),
            "recall": Recall.of(),
        }

        for m in metrics:
            if m not in metric_map:
                raise ValueError(f"Unsupported metric: {m}")

        for t in tasks:
            gt = f"gt_{t}"
            pred = f"pred_{t}"

            if gt not in eval_df.columns or pred not in eval_df.columns:
                print(
                    f"Either {gt} or {pred} not in dataset.\nAvailable columns:\n{eval_df.columns}"
                )

                # Fill None for all requested metrics
                metric_values = {m: None for m in metrics}

                task_results.append(
                    TaskMetricResult.of(
                        task_name=t,
                        **metric_values,
                    )
                )
                continue

            y_true = eval_df[gt]
            y_pred = eval_df[pred]

            metric_values = {}

            for m in metrics:
                metric_values[m] = metric_map[m].compute(y_true, y_pred)

            task_results.append(
                TaskMetricResult.of(
                    task_name=t,
                    **metric_values,
                )
            )

        return StepMultiMetricResult.of(
            unique_id=run_ctx.unique_id,
            algo_name=run_ctx.algo_name,
            step=step,
            split=split,
            task_metrics=task_results,
        )

    @staticmethod
    def generate_single_run_df(
        *,
        run_ctx: SingleRunContext,
        split: str,
        k: int = 5,
        metrics: List[str] = ["accuracy", "f1"],
    ) -> pd.DataFrame:
        """
        Generate dataframe for a single run containing only selected metrics.
        """

        base_dir = run_ctx.run_dir
        tasks = tuple(run_ctx.dataset_config["task_output_formats"].keys())

        rows = []
        step_files: List[Tuple[int, str]] = []

        for fname in os.listdir(base_dir):
            if fname.startswith("eval_step_") and fname.endswith(".parquet"):
                step_num = int(
                    fname.replace("eval_step_", "").replace(".parquet", "")
                )
                step_files.append((step_num, fname))

        step_files.sort()

        for step_num, fname in step_files:
            if step_num % k != 0:
                continue

            path = os.path.join(base_dir, fname)

            step_result = ExptEvaluator.compute_all_metrics(
                parquet_path=path,
                tasks=tasks,
                run_ctx=run_ctx,
                step=step_num,
                split=split,
                metrics=metrics,
            )

            for task_metric in step_result.task_metrics:
                row = {
                    "unique_id": step_result.unique_id,
                    "algo_name": step_result.algo_name,
                    "step": step_result.step,
                    "split": step_result.split,
                    "task_name": task_metric.task_name,
                }

                for m in metrics:
                    row[m] = getattr(task_metric, m, None)

                rows.append(row)

        return pd.DataFrame(rows)
if __name__ == "__main__":
    mega_dir_path = "./outputs/All_Together_B48_L3_G4/"
    dirs: List[str] = ["2", "3", "4", "5_5", "6"]
    dataset_config = {
        "prompt_prefix": "Evaluate the summary. Output JSON with the requested metric scores. Do NOT include reasoning or explanations. Each metric should contain a single integer. Formats like '4/5' or '4|5' are invalid.",
        "task_output_formats": {
            "fluency": "An integer between 1 to 5",                         # "1|2|3|4|5",
            "coherence": "An integer between 1 to 5",                       # "1|2|3|4|5",
            "relevance": "An integer between 1 to 5",                       # "1|2|3|4|5",
            "consistency": "An integer between 1 to 5",                     # "1|2|3|4|5",
        },
        "task_losses": {
            "fluency": "accuracy",
            "coherence": "accuracy",
            "relevance": "accuracy",
            "consistency": "accuracy",
        },
    }

    for dir in dirs:
        dir_path = mega_dir_path + dir
        ctx = ExptRunContext.build(
            expt_dir=dir_path,
            dataset_configuration=dataset_config,
        )

        print(ctx.keys())

        plot_dir : str = dir_path + "/plots"
        os.makedirs(plot_dir, exist_ok=True)

        for ctx_key, ctx_val in ctx.items():
            if ctx_key.split("_")[0] == "unknown":
                continue
            results = ExptEvaluator.generate_single_run_df(
                run_ctx=ctx_val,
                metrics=["accuracy", "f1", "precision", "recall"],
                split="test",
                k=5,
            )

            print(results.head())
            print('\n' * 5)

            # plot = LinePlot.of(
            #     df=results,
            #     metric="accuracy",
            #     metric_colors={
            #         "fluency": "#2ca02c",
            #         "coherence": "#1f77b4",
            #         "relevance": "#9467bd",
            #         "consistency": "#ff7f0e",
            #     },
            #     title=f"{ctx_key.split('_')[0]} - SummEval"
            # )

            # ## Saving a jpg file
            # hv.save(plot.render(), f"{plot_dir}/plot_{ctx_key}_accuracy.html")
            # print(f"Saved plot to {plot_dir}/plot_{ctx_key}_accuracy.html")