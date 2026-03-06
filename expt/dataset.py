import ast
import os
from typing import ClassVar, List, Optional

import pandas as pd
from datasets import load_dataset
from huggingface_hub import login

from prompt_moo.data_input import Dataset
from prompt_moo.data_structures import Task


class SummEval(Dataset):
    dataset_name: ClassVar[str] = "SummEval"
    train_size: ClassVar[int] = 160
    test_size: ClassVar[int] = 480
    input_cols: ClassVar[List[str]] = ["machine_summary", "text"]
    gt_cols: ClassVar[List[str]] = ["fluency", "relevance", "coherence", "consistency"]
    base_file_path: ClassVar[str] = "./cleaned_df.csv"

    # Task output format specifications for prompt generation
    task_output_formats: ClassVar[dict] = {
        "fluency": "An Integer between 1 to 5",                             # "1|2|3|4|5",
        "relevance": "An Integer between 1 to 5",                           # "1|2|3|4|5",
        "coherence": "An Integer between 1 to 5",                           # "1|2|3|4|5",
        "consistency": "An Integer between 1 to 5",                         # "1|2|3|4|5",
    }

    tasks: ClassVar[List[Task]] = [
        Task(
            task_name="fluency",
            task_description="Evaluate the fluency and readability of the summary",
            task_instruction="Rate the fluency of this summary on a scale from 1 (very poor) to 5 (excellent). Consider grammar, flow, and readability.",
            gt_col="fluency",
        ),
        Task(
            task_name="relevance",
            task_description="Evaluate whether the summary captures the most important information from the source text",
            task_instruction="Rate the relevance of this summary on a scale from 1 (not relevant) to 5 (highly relevant). Consider if it captures key information.",
            gt_col="relevance",
        ),
        Task(
            task_name="coherence",
            task_description="Evaluate the logical structure and organization of the summary",
            task_instruction="Rate the coherence of this summary on a scale from 1 (incoherent) to 5 (very coherent). Consider logical flow and organization.",
            gt_col="coherence",
        ),
        Task(
            task_name="consistency",
            task_description="Evaluate whether the summary is factually consistent with the source text",
            task_instruction="Rate the consistency of this summary on a scale from 1 (inconsistent) to 5 (fully consistent). Check for factual alignment with source.",
            gt_col="consistency",
        ),
    ]

    def setup(self, base_dir: Optional[str] = None):
        base_dir = base_dir or self.data_dir
        ## Load data:
        df = pd.read_csv(self.base_file_path)

        def parse_annotation(s):
            d = ast.literal_eval(s)
            return [d.get(col, None) for col in self.gt_cols]

        gt_values = df["expert_annotation"].apply(parse_annotation).tolist()
        gt_df = pd.DataFrame(gt_values, columns=self.gt_cols).apply(
            pd.to_numeric, errors="coerce"
        )

        df = pd.concat([df, gt_df], axis=1)

        cols_to_keep = self.input_cols + self.gt_cols
        df = df[cols_to_keep]

        df = df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        train_df = df.iloc[: self.train_size].reset_index(drop=True)
        test_df = df.iloc[
            self.train_size : self.train_size + self.test_size
        ].reset_index(drop=True)

        os.makedirs(os.path.join(base_dir, self.dataset_name), exist_ok=True)

        train_df.to_parquet(self.train_path(base_dir))
        test_df.to_parquet(self.test_path(base_dir))

        print(
            f"Saved train ({self.train_size}) and test ({self.test_size}) splits to {base_dir}/{self.dataset_name}."
        )


class WildGuard(Dataset):
    dataset_name: ClassVar[str] = "WildGuard"
    train_size: ClassVar[int] = 500
    test_size: ClassVar[int] = 500
    input_cols: ClassVar[List[str]] = ["prompt", "response"]
    gt_cols: ClassVar[List[str]] = [
        "prompt_harm_label",
        "response_harm_label",
        "response_refusal_label",
    ]

    # Task output format specifications for prompt generation
    task_output_formats: ClassVar[dict] = {
        "prompt_harm_label": '"unharmful" | "harmful"',
        "response_harm_label": '"unharmful" | "harmful"',
        "response_refusal_label": '"compliance" | "refusal"',
    }

    tasks: ClassVar[List[Task]] = [
        Task(
            task_name="prompt_harm_label",
            task_description="Classify whether the user prompt contains harmful or unsafe content",
            task_instruction="Classify the prompt as harmful or benign. Return 'harmful' if it requests unsafe, illegal, or unethical content, otherwise 'benign'.",
            gt_col="prompt_harm_label",
        ),
        Task(
            task_name="response_harm_label",
            task_description="Classify whether the model response contains harmful or unsafe content",
            task_instruction="Classify the response as harmful or benign. Return 'harmful' if it contains unsafe, illegal, or unethical content, otherwise 'benign'.",
            gt_col="response_harm_label",
        ),
        Task(
            task_name="response_refusal_label",
            task_description="Classify whether the model response appropriately refuses to answer harmful requests",
            task_instruction="Classify whether the response is a refusal or compliance. Return 'refusal' if it declines to answer, 'compliance' if it provides requested information.",
            gt_col="response_refusal_label",
        ),
    ]

    def setup(self, base_dir: Optional[str] = None):
        base_dir = base_dir or self.data_dir
        login(token=os.getenv("HF_TOKEN"))

        train = load_dataset(
            "allenai/wildguardmix",
            "wildguardtrain",
            token=os.getenv("HF_TOKEN"),
        )["train"].to_pandas()
        test = load_dataset(
            "allenai/wildguardmix",
            "wildguardtest",
            token=os.getenv("HF_TOKEN"),
        )["test"].to_pandas()

        cols_to_keep = self.input_cols + self.gt_cols
        train_df = train[cols_to_keep]
        test_df = test[cols_to_keep]

        ## Drop rows as some entries are NaN
        train_df = train_df.dropna().reset_index(drop=True)
        test_df = test_df.dropna().reset_index(drop=True)

        train_df = train_df.sample(frac=1, random_state=self.seed).reset_index(
            drop=True
        )
        test_df = test_df.sample(frac=1, random_state=self.seed).reset_index(drop=True)

        train_df = train_df.head(self.train_size)
        test_df = test_df.head(self.test_size)

        os.makedirs(os.path.join(base_dir, self.dataset_name), exist_ok=True)

        train_df.to_parquet(self.train_path(base_dir))
        test_df.to_parquet(self.test_path(base_dir))

        print(
            f"WildGuard: Saved train ({self.train_size}) and test ({len(test_df)}) splits to {base_dir}/{self.dataset_name}."
        )


class BRIGHTER(Dataset):
    dataset_name: ClassVar[str] = "BRIGHTER"
    train_size: ClassVar[int] = 2768  # All of eng train split
    test_size: ClassVar[int] = 2767  # All of eng test split
    input_cols: ClassVar[List[str]] = ["text"]
    gt_cols: ClassVar[List[str]] = [
        "anger",
        "fear",
        "joy",
        "sadness",
        "surprise",
    ]

    # Task output format specifications for prompt generation
    task_output_formats: ClassVar[dict] = {
        "anger": "An Integer between 0 to 3",                              # "0|1|2|3",
        "fear": "An Integer between 0 to 3",                               # "0|1|2|3",
        "joy": "An Integer between 0 to 3",                                # "0|1|2|3",
        "sadness": "An Integer between 0 to 3",                            # "0|1|2|3",
        "surprise": "An Integer between 0 to 3",                           # "0|1|2|3",
    }

    tasks: ClassVar[List[Task]] = [
        Task(
            task_name="anger",
            task_description="Evaluate the intensity of anger expressed in the text",
            task_instruction="Rate the anger intensity on a scale from 0 (no anger) to 3 (high anger). Consider frustration, hostility, and negative emotions.",
            gt_col="anger",
        ),
        Task(
            task_name="fear",
            task_description="Evaluate the intensity of fear expressed in the text",
            task_instruction="Rate the fear intensity on a scale from 0 (no fear) to 3 (high fear). Consider anxiety, worry, and apprehension.",
            gt_col="fear",
        ),
        Task(
            task_name="joy",
            task_description="Evaluate the intensity of joy expressed in the text",
            task_instruction="Rate the joy intensity on a scale from 0 (no joy) to 3 (high joy). Consider happiness, pleasure, and positive emotions.",
            gt_col="joy",
        ),
        Task(
            task_name="sadness",
            task_description="Evaluate the intensity of sadness expressed in the text",
            task_instruction="Rate the sadness intensity on a scale from 0 (no sadness) to 3 (high sadness). Consider sorrow, grief, and melancholy.",
            gt_col="sadness",
        ),
        Task(
            task_name="surprise",
            task_description="Evaluate the intensity of surprise expressed in the text",
            task_instruction="Rate the surprise intensity on a scale from 0 (no surprise) to 3 (high surprise). Consider unexpectedness and astonishment.",
            gt_col="surprise",
        ),
    ]

    def setup(self, base_dir: Optional[str] = None):
        base_dir = base_dir or self.data_dir
        # Load BRIGHTER dataset from Hugging Face
        print("Loading BRIGHTER dataset from Hugging Face...")

        # Load train and test splits for English
        train_data = load_dataset(
            "brighter-dataset/BRIGHTER-emotion-intensities", "eng", split="train"
        )
        test_data = load_dataset(
            "brighter-dataset/BRIGHTER-emotion-intensities", "eng", split="test"
        )

        # Convert to pandas
        train_df = train_data.to_pandas()
        test_df = test_data.to_pandas()

        # Keep only necessary columns
        cols_to_keep = self.input_cols + self.gt_cols
        train_df = train_df[cols_to_keep]
        test_df = test_df[cols_to_keep]

        # Drop any rows with NaN
        train_df = train_df.dropna().reset_index(drop=True)
        test_df = test_df.dropna().reset_index(drop=True)

        # Shuffle
        train_df = train_df.sample(frac=1, random_state=self.seed).reset_index(
            drop=True
        )
        test_df = test_df.sample(frac=1, random_state=self.seed).reset_index(drop=True)

        # Save to parquet
        os.makedirs(os.path.join(base_dir, self.dataset_name), exist_ok=True)
        train_df.to_parquet(self.train_path(base_dir))
        test_df.to_parquet(self.test_path(base_dir))

        print(
            f"BRIGHTER: Saved train ({len(train_df)}) and test ({len(test_df)}) splits to {base_dir}/{self.dataset_name}."
        )
