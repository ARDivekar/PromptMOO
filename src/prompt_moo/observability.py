"""
Observability Manager: A domain-agnostic ledger for prompt optimization runs.

The ObservabilityManager is a **dumb storage layer**. It knows nothing about
algorithms, trajectories, prompts, or tasks. It provides three primitives:

1. ``record(key, value)`` — Add a key-value pair to the current step page.
   If the value is a Pydantic BaseModel (or Morphic Typed), ``model_dump()``
   is called automatically.  Lists of BaseModels are model_dumped elementwise.
   Everything else is stored as-is.

2. ``log_step_end(step)`` — Flush the current page to a Parquet file and
   start a new blank page.

3. ``write_file(relative_path, content)`` — Write arbitrary text to a file
   under the output directory.

The algorithm classes are responsible for serializing their own domain objects
before calling ``record()``.

Step numbering convention:
    Step 0: Baseline evaluation only (no optimization, no run_logs entry).
    Steps 1..N: Optimization steps (each gets a run_logs entry).

File layout per run::

    output_dir/
        run_summary.json          — config + finalized step summary
        steps_summary.jsonl       — append-only step index (crash-safe)
        run_logs/                  — one parquet per optimization step
            step_0001.parquet     — Step 1 (first optimization)
            step_0002.parquet
            ...
        eval_step_0.parquet       — baseline (initial prompt, before optimization)
        eval_step_1.parquet       — after Step 1
        eval_step_5.parquet       — after Step 5 (if eval_every=5)
        ...
"""

import glob
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Tuple

import pandas as pd
from morphic import Typed, validate
from morphic.typed import format_exception_msg
from pydantic import BaseModel, PrivateAttr


def _serialize_value(value: Any) -> Any:
    """Serialize a single value for JSON/Parquet storage.

    - Pydantic BaseModel / Morphic Typed → ``.model_dump()``
    - List of BaseModels → ``[item.model_dump() for item in value]``
    - Everything else → as-is
    """
    if isinstance(value, (Typed, BaseModel)):
        return value.model_dump()
    elif isinstance(value, (list, tuple, set)):
        return [_serialize_value(item) for item in value]
    elif isinstance(value, dict):
        return {_serialize_value(k): _serialize_value(v) for k, v in value.items()}
    return value


class ObservabilityManager(Typed):
    """Domain-agnostic ledger for optimization run logging.

    Each training step is written to its own parquet file under ``run_logs/``.
    This avoids the O(N^2) read-concat-rewrite pattern: writing step N is
    always O(1) regardless of how many steps came before.

    To read all steps at once (e.g. after training), call
    ``ObservabilityManager.read_run_logs(output_dir)`` which concatenates
    the per-step parquets into a single DataFrame.

    Attributes:
        output_dir: Root directory for all run outputs.
        run_logs_dir: Directory containing per-step parquet files.
        summary_path: Path to the run_summary.json file.
        steps_jsonl_path: Path to the append-only steps index.
    """

    output_dir: str
    run_logs_dir: str
    summary_path: str
    steps_jsonl_path: str
    verbosity: int = 1

    _current_step_data: Dict[str, Any] = PrivateAttr(default_factory=dict)
    _total_steps_logged: int = PrivateAttr(default=0)

    @classmethod
    def pre_initialize(cls, data: dict) -> None:
        output_dir = data["output_dir"]
        data.setdefault("run_logs_dir", os.path.join(output_dir, "run_logs"))
        data.setdefault("summary_path", os.path.join(output_dir, "run_summary.json"))
        data.setdefault(
            "steps_jsonl_path", os.path.join(output_dir, "steps_summary.jsonl")
        )

    def post_initialize(self) -> None:
        os.makedirs(os.path.join(self.output_dir, "prompts"), exist_ok=True)
        os.makedirs(self.run_logs_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Backward-compatible property so tests using .current_step_data work
    # ------------------------------------------------------------------
    @property
    def current_step_data(self) -> Dict[str, Any]:
        return self._current_step_data

    @current_step_data.setter
    def current_step_data(self, value: Dict[str, Any]) -> None:
        self._current_step_data = value

    # ------------------------------------------------------------------
    # Static helper: read all per-step parquets into one DataFrame
    # ------------------------------------------------------------------
    @staticmethod
    def read_run_logs(output_dir: str) -> pd.DataFrame:
        """Read all per-step parquet files and concatenate into one DataFrame.

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
            except (IOError, OSError) as e:
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
    # Primitive: config logging
    # ------------------------------------------------------------------
    @validate
    def log_config(self, config: Dict[str, Any]) -> None:
        """Log run configuration to run_summary.json.

        Args:
            config: Configuration dictionary with all hyperparameters.
        """
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        if os.path.exists(self.steps_jsonl_path):
            try:
                with open(self.steps_jsonl_path, "r") as f:
                    self._total_steps_logged = sum(1 for line in f if line.strip())
            except (IOError, OSError, json.JSONDecodeError) as e:
                raise IOError(
                    f"Could not read steps_summary.jsonl:\n{format_exception_msg(e)}"
                ) from e

        with open(self.summary_path, "w") as f:
            json.dump(
                {
                    "run_id": run_id,
                    "started_at": datetime.now().isoformat(),
                    "config": config,
                },
                f,
                indent=2,
                default=str,
            )

    # ------------------------------------------------------------------
    # Primitive: step lifecycle
    # ------------------------------------------------------------------
    @validate
    def log_step_start(self, step: int) -> None:
        """Start a new ledger page for the given step.

        Args:
            step: Step number.
        """
        self._current_step_data = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
        }

    @validate
    def record(self, *, key: str, value: Any) -> None:
        """Add a key-value pair to the current step page.

        If ``value`` is a Pydantic BaseModel (or Morphic Typed),
        ``model_dump()`` is called.  If it is a list of BaseModels,
        each element is model_dumped.  Otherwise stored as-is.

        Args:
            key: Column name in the step's parquet row.
            value: The data to store. Must be JSON-serializable
                   (after automatic model_dump conversion).
        """
        self._current_step_data[key] = _serialize_value(value)

    @validate
    def write_file(self, *, relative_path: str, content: str) -> None:
        """Write arbitrary text to a file under the output directory.

        Creates parent directories as needed.

        Args:
            relative_path: Path relative to ``output_dir``.
            content: Text content to write.
        """
        full_path = os.path.join(self.output_dir, relative_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w") as f:
            f.write(content)

    @validate
    def write_parquet(self, *, relative_path: str, dataframe: pd.DataFrame) -> str:
        """Write a DataFrame to a parquet file under the output directory.

        Args:
            relative_path: Path relative to ``output_dir``.
            dataframe: The DataFrame to write.

        Returns:
            Absolute path to the written file.
        """
        full_path = os.path.join(self.output_dir, relative_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        dataframe.to_parquet(full_path, engine="pyarrow")
        if self.verbosity >= 1:
            print(f"[Observer] Saved → {full_path}")
        return full_path

    @validate
    def log_step_end(self, step: int) -> None:
        """Flush the current step page to a Parquet file.

        Each step is written as ``run_logs/step_NNNN.parquet``.
        No previous files are read — O(1) per step.

        Args:
            step: Step number.
        """
        self._total_steps_logged += 1

        serialized_row = self._serialize_step(self._current_step_data)
        step_df = pd.DataFrame([serialized_row])
        step_parquet_path = os.path.join(self.run_logs_dir, f"step_{step:04d}.parquet")
        step_df.to_parquet(step_parquet_path, engine="pyarrow")
        if self.verbosity >= 2:
            print(f"[Observer] Wrote step {step} → {step_parquet_path}")

        step_entry = {
            "step": self._current_step_data["step"],
            "timestamp": self._current_step_data["timestamp"],
            "has_evaluation": "evaluation" in self._current_step_data,
        }
        try:
            with open(self.steps_jsonl_path, "a") as f:
                f.write(json.dumps(step_entry) + "\n")
        except (IOError, OSError) as e:
            raise IOError(
                f"Could not append to steps_summary.jsonl:\n{format_exception_msg(e)}"
            ) from e

        self._current_step_data = {}

    # ------------------------------------------------------------------
    # Error logging
    # ------------------------------------------------------------------
    @validate
    def log_error(self, step: int, error: str) -> None:
        """Log an error. Must NOT raise — called from exception handlers.

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
            print(
                f"[Observer] CRITICAL: Error JSONL append failed "
                f"(original error at step {step} may be lost):\n"
                f"{format_exception_msg(e)}",
                file=sys.stderr,
            )

        try:
            with open(self.summary_path, "r") as f:
                summary = json.load(f)

            summary["error"] = error
            summary["error_step"] = step
            summary["error_at"] = error_at

            with open(self.summary_path, "w") as f:
                json.dump(summary, f, indent=2)
        except Exception as e:
            print(
                f"[Observer] CRITICAL: Error summary update failed "
                f"(original error at step {step} may be lost):\n"
                f"{format_exception_msg(e)}",
                file=sys.stderr,
            )

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
                    if len(line) > 0:
                        entry = json.loads(line)
                        if "type" not in entry or entry["type"] != "error":
                            steps_summary.append(entry)

        summary["completed_at"] = datetime.now().isoformat()
        summary["total_steps"] = self._total_steps_logged
        summary["steps_summary"] = steps_summary

        with open(self.summary_path, "w") as f:
            json.dump(summary, f, indent=2)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    @staticmethod
    def _serialize_step(step_data: Dict[str, Any]) -> Dict[str, Any]:
        serialized: Dict[str, Any] = {}
        for k, v in step_data.items():
            if isinstance(v, (dict, list)):
                serialized[k] = json.dumps(v, ensure_ascii=False)
            else:
                serialized[k] = v
        return serialized
