"""
Observability Manager: Comprehensive logging for prompt optimization runs.

This module handles all logging, file storage, and output generation for research analysis.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

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

    Attributes:
        output_dir: Directory for storing all output files
        parquet_path: Path to Parquet file with step-by-step logs
        summary_path: Path to summary JSON file
        prompts_dir: Directory for storing prompt text files
        run_data: List of step data dictionaries
        current_step_data: Current step's data being accumulated
        max_steps_in_memory: Maximum number of steps to keep in memory (default: 5)
    """

    output_dir: str
    parquet_path: str
    summary_path: str
    steps_jsonl_path: str
    prompts_dir: str
    run_data: List[Dict[str, Any]]
    current_step_data: Dict[str, Any]
    max_steps_in_memory: int = 5

    def __init__(self, output_dir: str, max_steps_in_memory: int = 5) -> None:
        """Initialize observability manager.

        Args:
            output_dir: Directory to store all outputs
            max_steps_in_memory: Maximum number of steps to keep in memory (default: 5)
        """
        self.output_dir = output_dir
        self.parquet_path = os.path.join(output_dir, "run_logs.parquet")
        self.summary_path = os.path.join(output_dir, "run_summary.json")
        self.steps_jsonl_path = os.path.join(output_dir, "steps_summary.jsonl")
        self.prompts_dir = os.path.join(output_dir, "prompts")

        # Create directories
        os.makedirs(self.prompts_dir, exist_ok=True)

        self.run_data = []
        self.current_step_data = {}
        self.max_steps_in_memory = max_steps_in_memory
        self._total_steps_logged = 0  # Track total steps for summary

    def log_config(self, config: Dict[str, Any]) -> None:
        """Log run configuration.

        Args:
            config: Configuration dictionary with all hyperparameters
        """
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # On resume: count existing JSONL lines from prior run
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
            step: Step number
        """
        self.current_step_data = {"step": step, "timestamp": datetime.now().isoformat()}

    def log_batch(self, batch: Batch) -> None:
        """Log the batch samples.

        Args:
            batch: Batch of dataset samples
        """
        self.current_step_data["batch"] = {
            "step": batch.step,
            "num_samples": len(batch.samples),
            "samples": [s.model_dump() for s in batch.samples],
        }

    def log_predictions(self, predictions: List[PredictionResult]) -> None:
        """Log prediction results.

        Args:
            predictions: List of prediction results
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
            feedbacks: Dict of feedbacks from loss computer (keys are Task objects)
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
            gradients: Dict of gradients from gradient computer (keys are Task objects)
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
            context: Algorithm context dict (e.g., loss_functions, trajectory, etc.)
        """
        # Serialize trajectory if present
        serialized_context: Dict[str, Any] = {}
        for key, value in context.items():
            if key == "trajectory" and value is not None:
                # Serialize trajectory as list of elements
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
                # Skip batch - already logged separately
                continue
            else:
                serialized_context[key] = value

        self.current_step_data["algorithm_context"] = serialized_context

    def log_algorithm_state(self, state: Dict[str, Any]) -> None:
        """Log algorithm-specific state after updates.

        Args:
            state: Algorithm state dict (e.g., trajectory, previous_instructions, etc.)
        """
        # Similar serialization as context
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
            old_prompt: Previous prompt template
            new_prompt: New prompt template
            meta_prompt: The meta-prompt sent to optimizer LLM
            optimizer_response: The raw response from optimizer LLM
        """
        step = self.current_step_data["step"]

        # Save full prompts to files
        old_prompt_path = os.path.join(self.prompts_dir, f"step_{step}_old.txt")
        new_prompt_path = os.path.join(self.prompts_dir, f"step_{step}_new.txt")

        with open(old_prompt_path, "w") as f:
            f.write(old_prompt.to_str())
        with open(new_prompt_path, "w") as f:
            f.write(new_prompt.to_str())

        # Also save meta-prompt and optimizer response for full observability
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
            step: Step number
            results: Evaluation results dictionary
        """
        prompt: str = results.get("task_prompt", "")
        preds: List[PredictionResult] = results.get("prompt_predictions", [])
        inputs: List[Any] = results.get("dataset_inputs", [])

        pred_map = {p.sample_id: p for p in preds}
        input_map = {s.sample_id: s for s in inputs}

        ## Flatten rows
        flattened = []
        for sid, sample in input_map.items():
            pred_obj = pred_map.get(sid)
            pred_outputs = pred_obj.task_outputs if pred_obj else {}

            # Flatten ground_truths into separate columns
            ground_truths = sample.ground_truths
            ground_truth_flat = {f"gt_{k}": v for k, v in ground_truths.items()}

            # Flatten predicted outputs if they are dict-like (e.g., {"taskA": "...", "taskB": "..."})
            pred_flat = {f"pred_{k}": v for k, v in pred_outputs.items()}

            raw_response = pred_obj.raw_response if pred_obj else None

            row = {
                "step": step,
                "sample_id": sid,
                "task_prompt": prompt,
                "inputs": sample.inputs,
                "prediction_score": raw_response,
            }

            # Merge flattened components
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
            # "results_df": df,
        }

    def _serialize_step(self, step_data: Dict) -> Dict:
        serialized = {}
        for k, v in step_data.items():
            if isinstance(v, (dict, list)):
                serialized[k] = json.dumps(v, ensure_ascii=False)
            else:
                serialized[k] = v

        return serialized

    def log_step_end(self, step: int) -> None:
        """Finalize and write step data.

        Args:
            step: Step number
        """
        self.run_data.append(self.current_step_data)
        self._total_steps_logged += 1

        serialized_row = self._serialize_step(self.current_step_data)

        ### LOADING existing parquet if it exists
        if os.path.exists(self.parquet_path):
            try:
                existing = pd.read_parquet(self.parquet_path, engine="pyarrow")
            except Exception as e:
                raise IOError(
                    f"Failed to read existing parquet at {self.parquet_path!r}:\n"
                    f"{format_exception_msg(e)}"
                ) from e
            combined = pd.concat(
                [existing, pd.DataFrame([serialized_row])],
                ignore_index=True
            )
        else:
            combined = pd.DataFrame([serialized_row])

        combined.to_parquet(self.parquet_path, engine="pyarrow")
        print(f"[Observer] Appended step {step} -> {self.parquet_path}")

        # Append step summary entry to JSONL (streaming-friendly, crash-safe)
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

        # Prune old steps from memory to prevent Ray memory overflow
        if len(self.run_data) > self.max_steps_in_memory:
            # Keep only the last N steps in memory
            steps_to_remove = len(self.run_data) - self.max_steps_in_memory
            self.run_data = self.run_data[steps_to_remove:]
            print(f"[Observer] Pruned {steps_to_remove} old step(s) from memory, keeping last {self.max_steps_in_memory}")
        
        self.current_step_data = {}

    def log_error(self, step: int, error: str) -> None:
        """Log an error for the current run.
    
        Args:
            step: Step number where error occurred
            error: Error message
        """
        error_at = datetime.now().isoformat()
        # Append error entry to JSONL
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

        # Also write error fields to run_summary.json for check_run_status compatibility
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

    def finalize(self) -> None:
        """Finalize the run: merge JSONL into run_summary.json."""
        # Read existing summary
        with open(self.summary_path, "r") as f:
            summary = json.load(f)

        # Merge steps from JSONL into summary
        steps_summary = []
        if os.path.exists(self.steps_jsonl_path):
            with open(self.steps_jsonl_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entry = json.loads(line)
                        # Only include step entries (skip error entries)
                        if "type" not in entry or entry["type"] != "error":
                            steps_summary.append(entry)

        summary["completed_at"] = datetime.now().isoformat()
        summary["total_steps"] = self._total_steps_logged
        summary["steps_summary"] = steps_summary

        # Write final merged summary
        with open(self.summary_path, "w") as f:
            json.dump(summary, f, indent=2)
