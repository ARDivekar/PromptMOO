import os
from abc import ABC, abstractmethod
from typing import ClassVar, Dict, List, Optional

import pandas as pd
from morphic import Registry, Typed, validate
from morphic.typed import format_exception_msg

from .task_output_spec import TaskOutputSpec


class Dataset(Typed, Registry, ABC):
    _allow_subclass_override = True

    dataset_name: ClassVar[str]
    train_size: ClassVar[int]
    test_size: ClassVar[int]
    input_cols: ClassVar[List[str]]
    gt_cols: ClassVar[List[str]]
    seed: ClassVar[int] = 42

    # Per-task output specifications.  Maps task_name -> TaskOutputSpec.
    # Drives the JSON format example, natural-language description, and
    # num_classes for metrics.  Task iteration order follows this dict's
    # insertion order (which must match Dataset.tasks ordering).
    task_output_specs: ClassVar[Dict[str, TaskOutputSpec]] = {}

    # Human-readable labels for input columns used in the task LLM prompt.
    # Maps raw column name -> display label shown to the task LLM.
    input_col_labels: ClassVar[Dict[str, str]] = {}

    # One-sentence description of what the LLM should evaluate and what
    # input data to focus on.  Referenced in the prompt as:
    #   "{evaluation_directive}. Use the instructions below ..."
    # Example: "Evaluate the Summary of the Source Text"
    evaluation_directive: ClassVar[str] = ""

    # ---- Legacy fields (derived from task_output_specs) ------------------
    # These exist for backward compatibility with code that reads them.
    # New datasets should set task_output_specs only; these are computed
    # automatically from it.

    # Maps task_name -> output format string (e.g., "1|2|3|4|5").
    task_output_formats: ClassVar[dict] = {}

    # Maps task_name -> loss metric name (e.g., "accuracy", "f1").
    task_losses: ClassVar[Dict[str, str]] = {}

    # Legacy alias for evaluation_directive.
    prompt_prefix: ClassVar[str] = ""

    data_dir: str

    @classmethod
    @abstractmethod
    def setup(cls, base_dir: str):
        pass

    @validate
    def train_path(self, base_dir: Optional[str] = None) -> str:
        base_dir = base_dir or self.data_dir
        return os.path.join(
            base_dir, self.dataset_name, f"{self.dataset_name}-train.parquet"
        )

    @validate
    def test_path(self, base_dir: Optional[str] = None) -> str:
        base_dir = base_dir or self.data_dir
        return os.path.join(
            base_dir, self.dataset_name, f"{self.dataset_name}-test.parquet"
        )

    def train(self) -> pd.DataFrame:
        path = self.train_path()
        try:
            df = pd.read_parquet(path, engine="pyarrow")
        except (IOError, OSError, FileNotFoundError) as e:
            raise IOError(
                f"Failed to read train parquet at {path!r}:\n{format_exception_msg(e)}"
            ) from e
        return (
            df.sample(frac=1, random_state=self.seed)
            .reset_index(drop=True)
            .head(self.train_size)
        )

    def test(self) -> pd.DataFrame:
        path = self.test_path()
        try:
            df = pd.read_parquet(path, engine="pyarrow")
        except (IOError, OSError, FileNotFoundError) as e:
            raise IOError(
                f"Failed to read test parquet at {path!r}:\n{format_exception_msg(e)}"
            ) from e
        return (
            df.sample(frac=1, random_state=self.seed)
            .reset_index(drop=True)
            .head(self.test_size)
        )
