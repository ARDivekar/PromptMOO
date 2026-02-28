import os
from abc import ABC, abstractmethod
from typing import ClassVar, List, Optional

import pandas as pd
from morphic import Registry, Typed
from morphic.typed import format_exception_msg


class Dataset(Typed, Registry, ABC):
    _allow_subclass_override = True

    dataset_name: ClassVar[str]
    train_size: ClassVar[int]
    test_size: ClassVar[int]
    input_cols: ClassVar[List[str]]
    gt_cols: ClassVar[List[str]]
    seed: ClassVar[int] = 42

    # Task output format specifications for dynamic prompt generation
    # Maps task_name -> output format string (e.g., "1|2|3|4|5" or '"option1" | "option2"')
    task_output_formats: ClassVar[dict] = {}

    data_dir: str

    @classmethod
    @abstractmethod
    def setup(cls, base_dir: str):
        pass

    def train_path(self, base_dir: Optional[str] = None) -> str:
        base_dir = base_dir or self.data_dir
        return os.path.join(
            base_dir, self.dataset_name, f"{self.dataset_name}-train.parquet"
        )

    def test_path(self, base_dir: Optional[str] = None) -> str:
        base_dir = base_dir or self.data_dir
        return os.path.join(
            base_dir, self.dataset_name, f"{self.dataset_name}-test.parquet"
        )

    def train(self) -> pd.DataFrame:
        path = self.train_path()
        try:
            df = pd.read_parquet(path, engine="pyarrow")
        except Exception as e:
            raise IOError(
                f"Failed to read train parquet at {path!r}:\n"
                f"{format_exception_msg(e)}"
            ) from e
        return (
            df
            .sample(frac=1, random_state=self.seed)
            .reset_index(drop=True)
            .head(self.train_size)
        )

    def test(self) -> pd.DataFrame:
        path = self.test_path()
        try:
            df = pd.read_parquet(path, engine="pyarrow")
        except Exception as e:
            raise IOError(
                f"Failed to read test parquet at {path!r}:\n"
                f"{format_exception_msg(e)}"
            ) from e
        return (
            df
            .sample(frac=1, random_state=self.seed)
            .reset_index(drop=True)
            .head(self.test_size)
        )
