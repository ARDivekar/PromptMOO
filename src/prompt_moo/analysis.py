import os

import holoviews as hv
import pandas as pd

hv.extension("bokeh")

from typing import ClassVar, Dict, List, Optional, Tuple

from morphic import Registry, Typed, validate
from morphic.typed import format_exception_msg

from .context_manager import ExptRunContext, SingleRunContext
from .data_structures import StepMultiMetricResult, TaskMetricResult
from .metrics import F1, Accuracy, Metric, Precision, Recall  # noqa: F401 (re-exported)


class Analysis(Typed, Registry):
    _allow_subclass_override: ClassVar[bool] = True


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
            task_df = self.df[self.df["task_name"] == task].sort_values("step")

            color = None
            if self.metric_colors is not None:
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


class HeatmapPlot(Visualizer):
    """
    Confusion-matrix heatmap for evaluation steps.

    Behavior:
    - Detects which tasks actually exist in the parquet file
    - Warns if dataset_config tasks are missing
    - Plots only tasks present in the run
    """

    run_ctx: SingleRunContext
    annot: bool = True
    cmap: str = "Blues"
    title: Optional[str] = None
    width: int = 400
    height: int = 400
    verbose: bool = False

    def render(self, step: int):
        parquet_path = os.path.join(self.run_ctx.run_dir, f"eval_step_{step}.parquet")

        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"Evaluation file not found: {parquet_path}")

        try:
            eval_df = pd.read_parquet(parquet_path, engine="pyarrow")
        except (IOError, OSError) as e:
            raise IOError(f"Failed to read parquet: {format_exception_msg(e)}")

        dataset_tasks = tuple(self.run_ctx.dataset_config["task_output_formats"].keys())

        present_tasks = []

        # --------------------------------------------------
        # Detect which tasks are actually present
        # --------------------------------------------------
        for task in dataset_tasks:
            gt_col = f"gt_{task}"
            pred_col = f"pred_{task}"

            if gt_col in eval_df.columns and pred_col in eval_df.columns:
                present_tasks.append(task)
            else:
                if self.verbose:
                    print(
                        f"[WARNING] - Task {task} is not present | "
                        f"present cols: {list(eval_df.columns)}"
                    )
                else:
                    print(f"[WARNING] - Task {task} is not present")

        if len(present_tasks) == 0:
            raise ValueError("No valid task columns found in parquet")

        plots = []

        ## Build heatmap for each present task
        for task in present_tasks:
            gt_col = f"gt_{task}"
            pred_col = f"pred_{task}"

            y_true = eval_df[gt_col].dropna()
            y_pred = eval_df[pred_col].dropna()

            common_idx = y_true.index.intersection(y_pred.index)

            y_true = y_true.loc[common_idx]
            y_pred = y_pred.loc[common_idx]

            labels = sorted(set(y_true.unique()) | set(y_pred.unique()), key=str)

            cm = pd.crosstab(
                y_true,
                y_pred,
                rownames=["Actual"],
                colnames=["Predicted"],
                dropna=False,
            )

            cm = cm.reindex(index=labels, columns=labels, fill_value=0)

            records = [
                (str(pred), str(act), int(cm.loc[act, pred]))
                for act in labels
                for pred in labels
            ]

            heatmap = hv.HeatMap(
                records,
                kdims=["Predicted", "Actual"],
                vdims=["Count"],
            ).opts(
                cmap=self.cmap,
                width=self.width,
                height=self.height,
                title=task.capitalize(),
                colorbar=True,
                xrotation=45,
                tools=["hover"],
            )

            plot = heatmap

            if self.annot:
                labels_overlay = hv.Labels(
                    [(p, a, str(c)) for p, a, c in records],
                    kdims=["Predicted", "Actual"],
                    vdims=["Count"],
                ).opts(
                    text_font_size="10pt",
                    text_color="black",
                )

                plot = heatmap * labels_overlay

            plots.append(plot)

        ## Layout logic
        if len(plots) == 1:
            return plots[0].opts(title=self.title or f"Confusion Matrix — Step {step}")

        layout = plots[0]

        for p in plots[1:]:
            layout = layout + p

        return layout.opts(title=self.title or f"Confusion Matrix — Step {step}").cols(
            len(plots)
        )


class ExptEvaluator(Typed, Registry):
    _allow_subclass_override: ClassVar[bool] = True

    @staticmethod
    @validate
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
        except (IOError, OSError) as e:
            raise IOError(
                f"Failed to read evaluation parquet at {parquet_path!r}:\n"
                f"{format_exception_msg(e)}"
            ) from e
        task_results: List[TaskMetricResult] = []

        for m in metrics:
            try:
                Metric.get_subclass(m)
            except (KeyError, ValueError):
                raise ValueError(
                    f"Unsupported metric: {m!r}. "
                    f"Register it as a Metric subclass in metrics.py."
                )

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
                metric_cls = Metric.get_subclass(m)
                metric_values[m] = metric_cls.compute(
                    y_true=list(y_true), y_pred=list(y_pred)
                )

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
    @validate
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
                step_num = int(fname.replace("eval_step_", "").replace(".parquet", ""))
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
    import sys

    sys.path.insert(0, "expt")
    from dataset import SummEval

    mega_dir_path = "./outputs/All_Together_B48_L3_G4/"
    dirs: List[str] = ["2", "3", "4", "5_5", "6"]
    dataset_config = {
        "task_output_formats": SummEval.task_output_formats,
        "task_losses": SummEval.task_losses,
    }

    for dir in dirs:
        dir_path = mega_dir_path + dir
        ctx = ExptRunContext.build(
            expt_dir=dir_path,
            dataset_configuration=dataset_config,
        )

        print(ctx.keys())

        plot_dir: str = dir_path + "/plots"
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

            # Create subplots for each metric
            plots = []

            for metric_name in ["accuracy", "f1", "precision", "recall"]:
                single_plot = LinePlot.of(
                    df=results,
                    metric=metric_name,
                    metric_colors={
                        "fluency": "#2ca02c",
                        "coherence": "#1f77b4",
                        "relevance": "#9467bd",
                        "consistency": "#ff7f0e",
                    },
                    title=f"{ctx_key.split('_')[0]} - {metric_name.capitalize()}",
                )

                plots.append(single_plot.render())

            # Combine into a vertical layout
            combined_plot = plots[0]
            for p in plots[1:]:
                combined_plot = combined_plot + p  # '+' creates subplots (Layout)

            # Arrange as 2 columns (optional)
            combined_plot = combined_plot.cols(2)

            # Save combined subplot
            hv.save(combined_plot, f"{plot_dir}/plot_{ctx_key}_all_metrics.html")

            print(
                f"Saved combined subplot to {plot_dir}/plot_{ctx_key}_all_metrics.html"
            )
