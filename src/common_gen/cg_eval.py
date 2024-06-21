"""
This script does evaluation for common_gen.

For each method, it saves a file called {method}_{metric}.json, where {method} is the method name and {metric} is the metric name. 

Example command: python3 -m src.common_gen.cg_eval --model falcon-1b --data_config_path dataset_configs/common_gen_1_shot_2_prompts.yaml
"""

import argparse
import json
from pathlib import Path

from src.console import console
from src.evaluate.metrics import *
from src.evaluate.scorer import *
from src.utils import (construct_predictions_dir_path, load_data_config,
                       load_hf_dataset)

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="LLM to use")
parser.add_argument("--device", default="cuda", type=str, help="Device to use")
parser.add_argument(
    "--data_config_path",
    type=str,
    help="Path to config file. This should be a yaml file",
)
parser.add_argument(
    "--hf_cache_dir",
    default="cache",
    type=str,
    help="Directory to cache HF datasets to",
)
parser.add_argument(
    "--results_dir",
    default="generative_ensembles_data/benchmark_results",
    type=str,
    help="Results directory",
)
parser.add_argument(
    "--redo",
    action="store_true",
    help="Redo evaluation even if results already exist. Otherwise, we only evaluate methods/metrics which aren't already evaluated.",
)


def main(args):
    # Load data config, prompts, and create output directory
    data_config = load_data_config(args)
    predictions_dir = construct_predictions_dir_path(data_config, args)
    print(predictions_dir)

    # Load scores if they already exist
    scorer = Scorer(predictions_dir=predictions_dir, data_config=data_config, args=args)

    # # Load dataset
    dataset = load_hf_dataset(
        dataset_name=data_config["dataset"],
        is_train=False,
        n_samples=data_config["test_size"],
        hf_cache_dir=args.hf_cache_dir,
    )
    references = get_references(dataset, data_config)

    predictions_files = list(predictions_dir.glob("*_test.json"))
    for predictions_fpath in predictions_files:
        for metric in data_config["metrics"]:
            scorer.score_method(
                predictions_fpath=predictions_fpath,
                metric=metric,
                references=references,
            )

    # Compute train individual scores
    train_dataset = load_hf_dataset(
        dataset_name=data_config["dataset"],
        is_train=True,
        n_samples=data_config["train_size"],
        hf_cache_dir=args.hf_cache_dir,
    )
    train_references = get_references(train_dataset, data_config)

    predictions_file = predictions_dir / "individual_train.json"
    for metric in data_config["metrics"]:
        scorer.score_train(
            predictions_fpath=predictions_file,
            metric=metric,
            references=train_references,
        )

    console.log(json.dumps(scorer.summary_scores, indent=4))

    # Save scores
    scorer.save()


if __name__ == "__main__":
    args = parser.parse_args()
    console.log(args)
    main(args)