import hashlib
import json
import os
from typing import Any, ClassVar, Dict, Optional

from morphic import Registry, Typed

# from morphic.string import hash as hash_str


class SingleRunContext(Typed, Registry):
    """
    Full execution context for a single algorithm run
    """

    _allow_subclass_override: ClassVar[bool] = True

    algo_name: str
    run_id: Optional[str]
    unique_id: str

    run_dir: str

    dataset_config: Dict[str, Any]

    prompts_dir: str
    logs_path: str
    summary_path: str

    @classmethod
    # @validate omitted: return type "SingleRunContext" is a forward reference
    # that Pydantic's validate_call cannot resolve at class-definition time.
    def produce(
        cls,
        *,
        run_dir: str,
        dataset_config: Dict[str, Any],
    ) -> "SingleRunContext":
        run_directory = os.path.abspath(run_dir)

        summary_path = os.path.join(run_directory, "run_summary.json")
        algo_name = "unknown"
        run_id = None

        if os.path.exists(summary_path):
            with open(summary_path, "r") as f:
                summary = json.load(f)

                if "config" not in summary:
                    raise ValueError(
                        f"Run summary at {summary_path} is missing 'config' key. "
                        f"Available keys: {list(summary.keys())}."
                    )
                summary_config = summary["config"]

                if (
                    "algo_name" not in summary_config
                    and "algorithm" not in summary_config
                ):
                    raise ValueError(
                        f"Run summary config at {summary_path} is missing both 'algo_name' "
                        f"and 'algorithm' keys. Available keys: {list(summary_config.keys())}."
                    )
                algo_name = (
                    summary_config.get("algo_name")
                    or summary_config.get("algorithm")
                    or "unknown"
                )

                run_id = summary.get("run_id", None)

        hash_input = f"{run_directory}_{run_id}"
        # short_hash = hash_str(hash_input)[:6]
        short_hash = hashlib.md5(hash_input.encode()).hexdigest()[:6]

        unique_id = f"{algo_name}_{short_hash}"

        return cls.of(
            algo_name=algo_name,
            run_dir=run_directory,
            unique_id=unique_id,
            run_id=run_id,
            dataset_config=dataset_config,
            prompts_dir=os.path.join(run_directory, "prompts"),
            logs_path=os.path.join(run_directory, "run_logs"),
            summary_path=summary_path,
        )

    # ---------------------------------------------------------
    # Filesystem checks
    # ---------------------------------------------------------
    def has_prompts(self) -> bool:
        return os.path.isdir(self.prompts_dir)

    def has_logs(self) -> bool:
        if os.path.isdir(self.logs_path):
            return len(os.listdir(self.logs_path)) > 0
        legacy_path = self.logs_path + ".parquet"
        return os.path.exists(legacy_path)

    def has_summary(self) -> bool:
        return os.path.exists(self.summary_path)

    # ---------------------------------------------------------
    # Run Status
    # ---------------------------------------------------------
    def load_summary(self) -> Optional[Dict[str, Any]]:
        if self.has_summary():
            with open(self.summary_path, "r") as f:
                return json.load(f)

        print(f"No run_summary.json found in {self.run_dir}.")
        return None

    def status(self) -> str:
        summary = self.load_summary()

        if summary and summary.get("completed_at"):
            return "completed"

        if self.has_logs():
            return "running"

        if self.has_prompts():
            return "initialized"

        return "not_started"


class ExptRunContext(Typed, Registry):
    """
    Container Mapping:
        algo_name -> SingleRunContext
    """

    _allow_subclass_override: ClassVar[bool] = True
    expt_dir: str
    dataset_config: Dict[str, Any]
    runs: Dict[str, SingleRunContext]  ## Key = unique_id for each SingleRunContext

    @classmethod
    # @validate omitted: return type "ExptRunContext" is a forward reference
    # that Pydantic's validate_call cannot resolve at class-definition time.
    def build(
        cls, *, expt_dir: str, dataset_configuration: Dict[str, Any]
    ) -> "ExptRunContext":
        expt_dir = os.path.abspath(expt_dir)
        dataset_config = dataset_configuration

        runs: Dict[str, SingleRunContext] = {}
        if os.path.isdir(expt_dir):
            # print(f"Found expt_dir: {expt_dir}")
            for item in os.listdir(expt_dir):
                run_dir = os.path.join(expt_dir, item)

                if os.path.isdir(run_dir):
                    # print(f"Found run_dir: {run_dir}")
                    ctx = SingleRunContext.produce(
                        run_dir=run_dir,
                        dataset_config=dataset_config,
                    )
                    runs[ctx.unique_id] = ctx

        else:
            print(f"{expt_dir} - Not found!")

        return cls.of(
            expt_dir=expt_dir,
            dataset_config=dataset_config,
            runs=runs,
        )

    # Dict-like behavior
    def __getitem__(self, key: str) -> SingleRunContext:
        return self.runs[key]

    def __contains__(self, key: str) -> bool:
        return key in self.runs

    def keys(self):
        return self.runs.keys()

    def values(self):
        return self.runs.values()

    def items(self):
        return self.runs.items()

    def __len__(self):
        return len(self.runs)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, "expt")
    from dataset import SummEval

    dir_path = "../outputs/1_t"
    dataset_config = {
        "task_output_formats": SummEval.task_output_formats,
        "task_losses": SummEval.task_losses,
    }

    ctx = ExptRunContext.build(
        expt_dir=dir_path,
        dataset_configuration=dataset_config,
    )

    print(ctx.runs.keys())
