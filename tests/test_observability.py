"""
Unit tests for ObservabilityManager step tracking.

Migrated from test/test_observability.py (the old test directory).

Verifies that:
- Each step produces its own parquet file under run_logs/
- steps_summary.jsonl is append-only
- finalize() merges JSONL lines into run_summary.json
- read_run_logs() concatenates per-step parquets correctly
"""

import json
import os

import pytest

from prompt_moo.observability import ObservabilityManager


def _read_jsonl(path: str):
    """Read all JSON lines from a JSONL file."""
    entries = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if len(line) > 0:
                entries.append(json.loads(line))
    return entries


class TestRunSummaryStepTracking:
    """Verify that steps_summary.jsonl records every step and finalize merges them."""

    def test_all_steps_logged_to_individual_parquets(self, tmp_path):
        """Each step produces its own parquet under run_logs/."""
        output_dir = str(tmp_path / "run_output")
        os.makedirs(output_dir, exist_ok=True)

        mgr = ObservabilityManager(output_dir=output_dir)
        mgr.log_config({"algorithm": "TEST", "steps": 10})

        num_steps = 10
        for step in range(num_steps):
            mgr.log_step_start(step)
            mgr.log_step_end(step)

        mgr.finalize()

        run_logs_dir = os.path.join(output_dir, "run_logs")
        parquet_files = sorted(
            f for f in os.listdir(run_logs_dir) if f.endswith(".parquet")
        )
        assert len(parquet_files) == num_steps

        expected_names = [f"step_{i:04d}.parquet" for i in range(num_steps)]
        assert parquet_files == expected_names

    def test_all_steps_in_jsonl(self, tmp_path):
        """All steps appear in steps_summary.jsonl."""
        output_dir = str(tmp_path / "run_output")
        os.makedirs(output_dir, exist_ok=True)

        mgr = ObservabilityManager(output_dir=output_dir)
        mgr.log_config({"algorithm": "TEST", "steps": 10})

        num_steps = 10
        for step in range(num_steps):
            mgr.log_step_start(step)
            mgr.log_step_end(step)

        mgr.finalize()

        entries = _read_jsonl(mgr.steps_jsonl_path)
        assert len(entries) == num_steps

        recorded_steps = [entry["step"] for entry in entries]
        assert recorded_steps == list(range(num_steps))

        with open(mgr.summary_path, "r") as f:
            summary = json.load(f)

        assert len(summary["steps_summary"]) == num_steps
        assert summary["total_steps"] == num_steps

    def test_read_run_logs_concatenates_all_steps(self, tmp_path):
        """read_run_logs() should return a DataFrame with all steps."""
        output_dir = str(tmp_path / "run_output")
        os.makedirs(output_dir, exist_ok=True)

        mgr = ObservabilityManager(output_dir=output_dir)
        mgr.log_config({"algorithm": "TEST", "steps": 5})

        for step in range(5):
            mgr.log_step_start(step)
            mgr.log_step_end(step)

        mgr.finalize()

        df = ObservabilityManager.read_run_logs(output_dir)
        assert len(df) == 5
        assert list(df["step"]) == list(range(5))

    def test_no_in_memory_accumulation(self, tmp_path):
        """No in-memory run_data list exists (removed entirely)."""
        output_dir = str(tmp_path / "run_output")
        os.makedirs(output_dir, exist_ok=True)

        mgr = ObservabilityManager(output_dir=output_dir)
        mgr.log_config({"algorithm": "TEST", "steps": 10})

        for step in range(10):
            mgr.log_step_start(step)
            mgr.log_step_end(step)

        assert not hasattr(mgr, "run_data")

    def test_steps_summary_available_before_finalize(self, tmp_path):
        """Steps should appear in steps_summary.jsonl even if finalize() hasn't been called yet."""
        output_dir = str(tmp_path / "run_output")
        os.makedirs(output_dir, exist_ok=True)

        mgr = ObservabilityManager(output_dir=output_dir)
        mgr.log_config({"algorithm": "TEST", "steps": 5})

        for step in range(5):
            mgr.log_step_start(step)
            mgr.log_step_end(step)

        entries = _read_jsonl(mgr.steps_jsonl_path)
        assert len(entries) == 5

        with open(mgr.summary_path, "r") as f:
            summary = json.load(f)
        assert "steps_summary" not in summary
        assert "completed_at" not in summary

    def test_has_evaluation_flag(self, tmp_path):
        """Steps with evaluation data should have has_evaluation=True."""
        output_dir = str(tmp_path / "run_output")
        os.makedirs(output_dir, exist_ok=True)

        mgr = ObservabilityManager(output_dir=output_dir)
        mgr.log_config({"algorithm": "TEST", "steps": 3})

        mgr.log_step_start(0)
        mgr.log_step_end(0)

        mgr.log_step_start(1)
        mgr.current_step_data["evaluation"] = {"step": 1, "results_file": "dummy"}
        mgr.log_step_end(1)

        mgr.log_step_start(2)
        mgr.log_step_end(2)

        mgr.finalize()

        entries = _read_jsonl(mgr.steps_jsonl_path)
        assert entries[0]["has_evaluation"] is False
        assert entries[1]["has_evaluation"] is True
        assert entries[2]["has_evaluation"] is False

    def test_resume_preserves_prior_steps(self, tmp_path):
        """When a run resumes from step 6, steps 0-5 from the prior run must be preserved."""
        output_dir = str(tmp_path / "run_output")
        os.makedirs(output_dir, exist_ok=True)

        mgr1 = ObservabilityManager(output_dir=output_dir)
        mgr1.log_config({"algorithm": "TEST", "steps": 21})
        for step in range(6):
            mgr1.log_step_start(step)
            mgr1.log_step_end(step)

        mgr2 = ObservabilityManager(output_dir=output_dir)
        mgr2.log_config({"algorithm": "TEST", "steps": 21, "start_step": 6})
        for step in range(6, 21):
            mgr2.log_step_start(step)
            mgr2.log_step_end(step)
        mgr2.finalize()

        entries = _read_jsonl(mgr2.steps_jsonl_path)
        assert len(entries) == 21

        recorded_steps = [entry["step"] for entry in entries]
        assert recorded_steps == list(range(21))

        with open(mgr2.summary_path, "r") as f:
            summary = json.load(f)
        assert len(summary["steps_summary"]) == 21
        assert summary["total_steps"] == 21

        df = ObservabilityManager.read_run_logs(output_dir)
        assert len(df) == 21

    def test_finalize_merges_jsonl_into_summary(self, tmp_path):
        """After finalize(), run_summary.json should contain the full steps_summary from JSONL."""
        output_dir = str(tmp_path / "run_output")
        os.makedirs(output_dir, exist_ok=True)

        mgr = ObservabilityManager(output_dir=output_dir)
        mgr.log_config({"algorithm": "TEST", "steps": 4})

        for step in range(4):
            mgr.log_step_start(step)
            if step == 2:
                mgr.current_step_data["evaluation"] = {
                    "step": step,
                    "results_file": "dummy",
                }
            mgr.log_step_end(step)

        with open(mgr.summary_path, "r") as f:
            pre_summary = json.load(f)
        assert "steps_summary" not in pre_summary

        mgr.finalize()

        with open(mgr.summary_path, "r") as f:
            post_summary = json.load(f)

        assert "steps_summary" in post_summary
        assert len(post_summary["steps_summary"]) == 4
        assert post_summary["steps_summary"][2]["has_evaluation"] is True
        assert post_summary["completed_at"] is not None

    def test_error_logged_to_jsonl(self, tmp_path):
        """log_error should append an error entry to JSONL and update run_summary.json."""
        output_dir = str(tmp_path / "run_output")
        os.makedirs(output_dir, exist_ok=True)

        mgr = ObservabilityManager(output_dir=output_dir)
        mgr.log_config({"algorithm": "TEST", "steps": 5})

        for step in range(3):
            mgr.log_step_start(step)
            mgr.log_step_end(step)

        mgr.log_error(3, "TimeoutError: LLM call timed out")

        entries = _read_jsonl(mgr.steps_jsonl_path)
        assert len(entries) == 4
        assert entries[3]["type"] == "error"
        assert entries[3]["step"] == 3
        assert "TimeoutError" in entries[3]["error"]

        with open(mgr.summary_path, "r") as f:
            summary = json.load(f)
        assert summary["error_step"] == 3
        assert "TimeoutError" in summary["error"]


class TestObservabilityErrorRecovery:
    """Verify that log_error() does not raise even when the filesystem fails.

    The log_error() contract is "must not raise" because it is called from
    the training loop's exception handler. If log_error() raised, it would
    mask the original training error.
    """

    def test_log_error_does_not_raise_on_missing_summary(self, tmp_path):
        """If run_summary.json was never created, log_error still succeeds."""
        output_dir = str(tmp_path / "broken_run")
        os.makedirs(output_dir, exist_ok=True)

        mgr = ObservabilityManager(output_dir=output_dir)

        os.remove(mgr.summary_path) if os.path.exists(mgr.summary_path) else None
        mgr.log_error(0, "Something went wrong")

    def test_log_error_does_not_raise_on_readonly_directory(self, tmp_path):
        """If the output directory becomes read-only, log_error still succeeds."""
        output_dir = str(tmp_path / "readonly_run")
        os.makedirs(output_dir, exist_ok=True)

        mgr = ObservabilityManager(output_dir=output_dir)
        mgr.log_config({"algorithm": "TEST", "steps": 1})

        os.chmod(output_dir, 0o444)
        try:
            mgr.log_error(0, "Cannot write")
        finally:
            os.chmod(output_dir, 0o755)
