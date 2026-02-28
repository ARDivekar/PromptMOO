"""
Setup script to download and prepare all datasets for PromptMOO experiments.

This script downloads and preprocesses:
- SummEval: Summary quality evaluation
- WildGuard: Safety classification
- BRIGHTER: Emotion intensity detection

Usage:
    python setup_datasets.py [--datasets DATASET1 DATASET2 ...]

Examples:
    python setup_datasets.py                    # Setup all datasets
    python setup_datasets.py --datasets BRIGHTER  # Setup only BRIGHTER
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dataset import BRIGHTER, SummEval, WildGuard


def setup_dataset(dataset_name: str, base_dir: str = ".") -> None:
    """Setup a specific dataset."""
    dataset_map = {
        "SummEval": SummEval,
        "WildGuard": WildGuard,
        "BRIGHTER": BRIGHTER,
    }

    if dataset_name not in dataset_map:
        print(f"❌ Unknown dataset: {dataset_name}")
        print(f"   Available: {', '.join(dataset_map.keys())}")
        return False

    dataset_cls = dataset_map[dataset_name]

    try:
        print(f"\n{'=' * 60}")
        print(f"Setting up {dataset_name}...")
        print(f"{'=' * 60}")
        dataset_cls.setup(base_dir=base_dir)
        print(f"✅ {dataset_name} setup complete!")
        return True
    except Exception as e:
        print(f"❌ Failed to setup {dataset_name}: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Setup datasets for PromptMOO experiments"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["SummEval", "WildGuard", "BRIGHTER"],
        choices=["SummEval", "WildGuard", "BRIGHTER"],
        help="Which datasets to setup (default: all)",
    )
    parser.add_argument(
        "--base-dir",
        default=".",
        help="Base directory to save datasets (default: current directory)",
    )

    args = parser.parse_args()

    print("PromptMOO Dataset Setup")
    print("=" * 60)
    print(f"Datasets to setup: {', '.join(args.datasets)}")
    print(f"Base directory: {args.base_dir}")

    results = {}
    for dataset_name in args.datasets:
        success = setup_dataset(dataset_name, args.base_dir)
        results[dataset_name] = success

    print(f"\n{'=' * 60}")
    print("Setup Summary")
    print(f"{'=' * 60}")
    for dataset_name, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        print(f"  {dataset_name:15s}: {status}")

    total_success = sum(results.values())
    total = len(results)
    print(f"\nTotal: {total_success}/{total} datasets setup successfully")

    if total_success < total:
        print("\n⚠️  Some datasets failed to setup. Check errors above.")
        sys.exit(1)
    else:
        print("\n🎉 All datasets ready! You can now run Runner.ipynb")
