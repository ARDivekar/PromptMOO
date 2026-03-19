"""
Observability Manager: Comprehensive logging for prompt optimization runs.

This module handles all logging, file storage, and output generation for research analysis.

File layout per run:
    output_dir/
        run_summary.json          — config + finalized step summary
        steps_summary.jsonl       — append-only step index (crash-safe)
        prompts/
            step_0_old.txt
            step_0_new.txt
            step_0_meta_prompt.txt
            step_0_optimizer_response.txt
            ...
        run_logs/                  — one parquet per step (O(1) per step, no re-reads)
            step_0000.parquet
            step_0001.parquet
            ...
        eval_step_0.parquet        — evaluation results (unchanged)
        eval_step_5.parquet
        ...
"""

import glob
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from morphic.typed import format_exception_msg

from .data_structures import (
    Batch,
    NumericFeedback,
    PredictionResult,
    Task,
    TextGradient,
    TextualFeedback,
)
from .prompt_template_utils import PromptTemplate
from .prompt_trajectory import PromptTrajectory


class ObservabilityManager:
    """Manages all logging and output for optimization runs.

    Each training step is written to its own parquet file under ``run_logs/``.
    This avoids the O(N^2) read-concat-rewrite pattern entirely: writing step N
    is always O(1) regardless of how many steps came before.

    To read all steps at once (e.g. after training), call
    ``ObservabilityManager.read_run_logs(output_dir)`` which concatenates the
    per-step parquets into a single DataFrame.
    """

    output_dir: str
    run_logs_dir: str
    summary_path: str
    steps_jsonl_path: str
    prompts_dir: str
    current_step_data: Dict[str, Any]

    def __init__(self, output_dir: str) -> None:
        """Initialize observability manager.

        Args:
            output_dir: Directory to store all outputs.
        """
        self.output_dir = output_dir
        self.run_logs_dir = os.path.join(output_dir, "run_logs")
        self.summary_path = os.path.join(output_dir, "run_summary.json")
        self.steps_jsonl_path = os.path.join(output_dir, "steps_summary.jsonl")
        self.prompts_dir = os.path.join(output_dir, "prompts")

        os.makedirs(self.prompts_dir, exist_ok=True)
        os.makedirs(self.run_logs_dir, exist_ok=True)

        self.current_step_data = {}
        self._total_steps_logged = 0

    # ------------------------------------------------------------------
    # Static helper: read all per-step parquets into one DataFrame
    # ------------------------------------------------------------------
    @staticmethod
    def read_run_logs(output_dir: str) -> pd.DataFrame:
        """Read all per-step parquet files and concatenate into one DataFrame.

        Uses ``glob.glob`` to find all ``step_*.parquet`` files under
        ``run_logs/``, sorts them by step number, and concatenates.
        Each row is guaranteed to have a ``step`` column.

        Falls back to reading a legacy single-file ``run_logs.parquet``
        if the ``run_logs/`` directory does not exist (for old runs).

        Args:
            output_dir: The run output directory containing ``run_logs/``.

        Returns:
            Combined DataFrame with one row per step, sorted by step number.

        Raises:
            FileNotFoundError: If neither ``run_logs/`` nor ``run_logs.parquet`` exists.
            IOError: If any parquet file cannot be read.
        """
        run_logs_dir = os.path.join(output_dir, "run_logs")
        if not os.path.isdir(run_logs_dir):
            legacy_path = os.path.join(output_dir, "run_logs.parquet")
            if os.path.exists(legacy_path):
                return pd.read_parquet(legacy_path, engine="pyarrow")
            raise FileNotFoundError(f"No run_logs/ directory found in {output_dir!r}")

        pattern = os.path.join(run_logs_dir, "step_*.parquet")
        matched_files = glob.glob(pattern)

        if len(matched_files) == 0:
            return pd.DataFrame()

        def _extract_step_number(fpath: str) -> int:
            basename = os.path.basename(fpath)
            num_str = basename.replace("step_", "").replace(".parquet", "")
            return int(num_str)

        sorted_files: List[Tuple[int, str]] = sorted(
            [(_extract_step_number(f), f) for f in matched_files],
            key=lambda t: t[0],
        )

        parts: List[pd.DataFrame] = []
        for step_num, fpath in sorted_files:
            try:
                part = pd.read_parquet(fpath, engine="pyarrow")
            except Exception as e:
                raise IOError(
                    f"Failed to read step parquet at {fpath!r}:\n"
                    f"{format_exception_msg(e)}"
                ) from e
            if "step" not in part.columns:
                part["step"] = step_num
            parts.append(part)

        combined = pd.concat(parts, ignore_index=True)
        if "step" in combined.columns:
            combined = combined.sort_values("step", ignore_index=True)
        return combined

    # ------------------------------------------------------------------
    # Logging methods
    # ------------------------------------------------------------------
    def log_config(self, config: Dict[str, Any]) -> None:
        """Log run configuration.

        Args:
            config: Configuration dictionary with all hyperparameters.
        """
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        if os.path.exists(self.steps_jsonl_path):
            try:
                with open(self.steps_jsonl_path, "r") as f:
                    self._total_steps_logged = sum(1 for line in f if line.strip())
            except Exception:
                pass

        with open(self.summary_path, "w") as f:
            json.dump(
                {
                    "run_id": run_id,
                    "started_at": datetime.now().isoformat(),
                    "config": config,
                },
                f,
                indent=2,
            )

    def log_step_start(self, step: int) -> None:
        """Log the start of a step.

        Args:
            step: Step number.
        """
        self.current_step_data = {"step": step, "timestamp": datetime.now().isoformat()}

    def log_batch(self, batch: Batch) -> None:
        """Log the batch samples.

        Args:
            batch: Batch of dataset samples.
        """
        self.current_step_data["batch"] = {
            "step": batch.step,
            "num_samples": len(batch.samples),
            "samples": [s.model_dump() for s in batch.samples],
        }

    def log_predictions(self, predictions: List[PredictionResult]) -> None:
        """Log prediction results.

        Args:
            predictions: List of prediction results.
        """
        self.current_step_data["predictions"] = {
            "num_predictions": len(predictions),
            "predictions": [p.model_dump() for p in predictions],
        }

    def log_feedbacks(
        self,
        feedbacks: Dict[Task, List[Union[NumericFeedback, TextualFeedback]]],
    ) -> None:
        """Log feedback/loss computations.

        Args:
            feedbacks: Dict of feedbacks from loss computer (keys are Task objects).
        """
        serialized_feedbacks: Dict[str, List[Dict[str, Any]]] = {}
        for task, feedback_list in feedbacks.items():
            serialized_feedbacks[task.task_name] = [
                fb.model_dump() for fb in feedback_list
            ]

        self.current_step_data["feedbacks"] = {
            "num_tasks": len(feedbacks),
            "feedbacks": serialized_feedbacks,
        }

    def log_gradients(
        self,
        gradients: Dict[Task, List[TextGradient]],
    ) -> None:
        """Log gradient computations including LLM prompts and responses.

        Args:
            gradients: Dict of gradients from gradient computer (keys are Task objects).
        """
        serialized_gradients: Dict[str, List[Dict[str, Any]]] = {}
        for task, gradient_list in gradients.items():
            serialized_gradients[task.task_name] = [
                g.model_dump() for g in gradient_list
            ]

        self.current_step_data["gradients"] = {
            "num_tasks": len(gradients),
            "gradients": serialized_gradients,
        }

    def log_algorithm_context(self, context: Dict[str, Any]) -> None:
        """Log algorithm-specific context passed to components.

        Args:
            context: Algorithm context dict (e.g., loss_functions, trajectory, etc.).
        """
        serialized_context: Dict[str, Any] = {}
        for key, value in context.items():
            if key == "trajectory" and value is not None:
                trajectory: PromptTrajectory = value
                try:
                    serialized_context["trajectory"] = [
                        {
                            "loss_fns": elem.loss_fns,
                            "scores": elem.scores,
                            "grads": elem.grads,
                            "instructions": elem.instructions,
                            "ranking_metric": elem.ranking_metric(),
                        }
                        for elem in trajectory.get_topk()
                    ]
                    serialized_context["trajectory_k"] = trajectory.k
                except Exception:
                    serialized_context["trajectory"] = str(value)
            elif key == "batch":
                continue
            else:
                serialized_context[key] = value

        self.current_step_data["algorithm_context"] = serialized_context

    def log_algorithm_state(self, state: Dict[str, Any]) -> None:
        """Log algorithm-specific state after updates.

        Args:
            state: Algorithm state dict (e.g., trajectory, previous_instructions, etc.).
        """
        serialized_state: Dict[str, Any] = {}
        for key, value in state.items():
            if key == "trajectory" and value is not None:
                trajectory: PromptTrajectory = value
                try:
                    serialized_state["trajectory"] = [
                        {
                            "loss_fns": elem.loss_fns,
                            "scores": elem.scores,
                            "grads": elem.grads,
                            "instructions": elem.instructions,
                            "ranking_metric": elem.ranking_metric(),
                        }
                        for elem in trajectory.get_topk()
                    ]
                    serialized_state["trajectory_k"] = trajectory.k
                    serialized_state["trajectory_size"] = len(trajectory)
                except Exception:
                    serialized_state["trajectory"] = str(value)
            else:
                serialized_state[key] = value

        self.current_step_data["algorithm_state"] = serialized_state

    def log_prompt_update(
        self,
        old_prompt: PromptTemplate,
        new_prompt: PromptTemplate,
        meta_prompt: Optional[str] = None,
        optimizer_response: Optional[str] = None,
    ) -> None:
        """Log prompt update with full text storage including optimizer LLM calls.

        Args:
            old_prompt: Previous prompt template.
            new_prompt: New prompt template.
            meta_prompt: The meta-prompt sent to optimizer LLM.
            optimizer_response: The raw response from optimizer LLM.
        """
        step = self.current_step_data["step"]

        old_prompt_path = os.path.join(self.prompts_dir, f"step_{step}_old.txt")
        new_prompt_path = os.path.join(self.prompts_dir, f"step_{step}_new.txt")

        with open(old_prompt_path, "w") as f:
            f.write(old_prompt.to_str())
        with open(new_prompt_path, "w") as f:
            f.write(new_prompt.to_str())

        if meta_prompt is not None:
            meta_prompt_path = os.path.join(
                self.prompts_dir, f"step_{step}_meta_prompt.txt"
            )
            with open(meta_prompt_path, "w") as f:
                f.write(meta_prompt)

        if optimizer_response is not None:
            optimizer_response_path = os.path.join(
                self.prompts_dir, f"step_{step}_optimizer_response.txt"
            )
            with open(optimizer_response_path, "w") as f:
                f.write(optimizer_response)

        self.current_step_data["prompt_update"] = {
            "old_prompt_file": f"prompts/step_{step}_old.txt",
            "new_prompt_file": f"prompts/step_{step}_new.txt",
            "old_instruction": old_prompt.instruction,
            "new_instruction": new_prompt.instruction,
            "instructions_changed": new_prompt.instruction,
            "meta_prompt": meta_prompt,
            "optimizer_response": optimizer_response,
            "meta_prompt_file": f"prompts/step_{step}_meta_prompt.txt"
            if meta_prompt is not None
            else None,
            "optimizer_response_file": f"prompts/step_{step}_optimizer_response.txt"
            if optimizer_response is not None
            else None,
        }

    def log_evaluation(self, step: int, results: Dict[str, Any]) -> None:
        """Log evaluation results.

        Args:
            step: Step number.
            results: Evaluation results dictionary.
        """
        prompt: str = results.get("task_prompt", "")
        preds: List[PredictionResult] = results.get("prompt_predictions", [])
        inputs: List[Any] = results.get("dataset_inputs", [])

        pred_map = {p.sample_id: p for p in preds}
        input_map = {s.sample_id: s for s in inputs}

        flattened = []
        for sid, sample in input_map.items():
            pred_obj = pred_map.get(sid)
            pred_outputs = pred_obj.task_outputs if pred_obj else {}

            ground_truths = sample.ground_truths
            ground_truth_flat = {f"gt_{k}": v for k, v in ground_truths.items()}
            pred_flat = {f"pred_{k}": v for k, v in pred_outputs.items()}

            raw_response = pred_obj.raw_response if pred_obj else None

            row = {
                "step": step,
                "sample_id": sid,
                "task_prompt": prompt,
                "inputs": sample.inputs,
                "prediction_score": raw_response,
            }
            row.update(ground_truth_flat)
            row.update(pred_flat)
            flattened.append(row)

        df = pd.DataFrame(flattened)
        output_path = os.path.join(self.output_dir, f"eval_step_{step}.parquet")
        df.to_parquet(output_path, engine="pyarrow")
        print(f"[Observer] Saved evaluation results → {output_path}")
        self.current_step_data["evaluation"] = {
            "step": step,
            "results_file": output_path,
        }

    # ------------------------------------------------------------------
    # Step finalization
    # ------------------------------------------------------------------
    def _serialize_step(self, step_data: Dict) -> Dict:
        serialized = {}
        for k, v in step_data.items():
            if isinstance(v, (dict, list)):
                serialized[k] = json.dumps(v, ensure_ascii=False)
            else:
                serialized[k] = v
        return serialized

    def log_step_end(self, step: int) -> None:
        """Finalize and write step data to its own parquet file.

        Each step is written as ``run_logs/step_NNNN.parquet``.  No previous
        files are read — this is O(1) per step.

        Args:
            step: Step number.
        """
        self._total_steps_logged += 1

        serialized_row = self._serialize_step(self.current_step_data)
        step_df = pd.DataFrame([serialized_row])
        step_parquet_path = os.path.join(self.run_logs_dir, f"step_{step:04d}.parquet")
        step_df.to_parquet(step_parquet_path, engine="pyarrow")
        print(f"[Observer] Wrote step {step} → {step_parquet_path}")

        step_entry = {
            "step": self.current_step_data.get("step"),
            "timestamp": self.current_step_data.get("timestamp"),
            "has_evaluation": "evaluation" in self.current_step_data,
        }
        try:
            with open(self.steps_jsonl_path, "a") as f:
                f.write(json.dumps(step_entry) + "\n")
        except Exception as e:
            print(f"[Observer] Warning: could not append to steps_summary.jsonl: {e}")

        self.current_step_data = {}

    # ------------------------------------------------------------------
    # Error logging
    # ------------------------------------------------------------------
    def log_error(self, step: int, error: str) -> None:
        """Log an error for the current run.

        Args:
            step: Step number where error occurred.
            error: Error message.
        """
        error_at = datetime.now().isoformat()
        try:
            error_entry = {
                "type": "error",
                "step": step,
                "error": error,
                "error_at": error_at,
            }
            with open(self.steps_jsonl_path, "a") as f:
                f.write(json.dumps(error_entry) + "\n")
        except Exception as e:
            print(f"[Observer] Error JSONL append failed: {e}")

        try:
            with open(self.summary_path, "r") as f:
                summary = json.load(f)

            summary["error"] = error
            summary["error_step"] = step
            summary["error_at"] = error_at

            with open(self.summary_path, "w") as f:
                json.dump(summary, f, indent=2)

            print(f"[Observer] Error logged → {self.summary_path}")
        except Exception as e:
            print(f"[Observer] Error summary update failed: {e}")

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------
    def finalize(self) -> None:
        """Finalize the run: merge JSONL into run_summary.json."""
        with open(self.summary_path, "r") as f:
            summary = json.load(f)

        steps_summary = []
        if os.path.exists(self.steps_jsonl_path):
            with open(self.steps_jsonl_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entry = json.loads(line)
                        if "type" not in entry or entry["type"] != "error":
                            steps_summary.append(entry)

        summary["completed_at"] = datetime.now().isoformat()
        summary["total_steps"] = self._total_steps_logged
        summary["steps_summary"] = steps_summary

        with open(self.summary_path, "w") as f:
            json.dump(summary, f, indent=2)
