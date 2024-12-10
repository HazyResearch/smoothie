"""
The MixInstruct dataset contains generations from 11 different models for a set of instruction, and ChatGPT pairwise comparisons over the answers.

This script does the following: 

1. Saves MixInstruct dataset in the desired format. 
2. Saves generations in the desired format.

Sample command:

python -m src.setup_mix_instruct \
    --hf_cache_dir cache \
    --data_dir mix_instruct_test/datasets \
    --results_dir mix_instruct_test/multi_model_results
"""


import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import yaml

from src.console import console
from src.data_utils import (
    construct_processed_dataset_paths,
    generate_prompt,
    get_embedding_inputs,
    get_reference,
    load_hf_dataset,
)
from src.utils import construct_generations_path
from src.ensembles import MIX_INSTRUCT_GROUPS

parser = argparse.ArgumentParser()
parser.add_argument(
    "--hf_cache_dir",
    default="cache",
    type=str,
    help="Directory to cache HF datasets to",
)
parser.add_argument(
    "--data_dir",
    default="smoothie_data/datasets",
    type=str,
    help="Directory with data files",
)
parser.add_argument(
    "--results_dir",
    type=str,
    help="Directory to save results to",
    required=True,
)
parser.add_argument(
    "--redo",
    action="store_true",
)


def process_reference(
    cmp_dict: Dict[str, int], row_models_to_generations: Dict[str, str]
) -> Dict[str, int]:
    """
    Process a dictionary of pairwise comparisons. Returns ranking of models.

    Args:
        cmp_dict (Dict[str, int]): model_a, model_b -> "A is better" or "B is better"
        row_models_to_generations (Dict[str, str]): model_name -> generation

    Returns:
        rank_dict (Dict[str, int]): model_generation -> rank
    """
    wins_dict = {}
    for key, value in cmp_dict.items():
        model_a, model_b = key.split(",")
        if model_a not in wins_dict:
            wins_dict[model_a] = 0
        if model_b not in wins_dict:
            wins_dict[model_b] = 0
        if value == "A is better":
            wins_dict[model_a] += 1
        elif value == "B is better":
            wins_dict[model_b] += 1

    sorted_models = sorted(wins_dict.items(), key=lambda x: x[1], reverse=True)

    model_name_rank_dict = {}
    current_rank = 1
    previous_wins = None

    for i, (model, wins) in enumerate(sorted_models):
        if wins != previous_wins:
            current_rank = i + 1
        model_name_rank_dict[model] = current_rank
        previous_wins = wins

    # Build model generation -> rank dict
    rank_dict = {}
    for model, generation in row_models_to_generations.items():
        rank_dict[generation] = model_name_rank_dict[model]

    return rank_dict


def main(args):
    args.dataset_config = "dataset_configs/mix_instruct.yaml"
    args.multi_model = True
    args.multi_prompt = False
    args.n_generations = 1
    args.test = False
    args.model_group = "mi_all"

    # Load yaml configs file
    data_config = yaml.load(
        Path(args.dataset_config).read_text(), Loader=yaml.FullLoader
    )
    console.log(data_config)

    train_data_fpath, test_data_fpath = construct_processed_dataset_paths(args)
    if train_data_fpath.exists() and test_data_fpath.exists() and not args.redo:
        console.log(
            f"Processed datasets {train_data_fpath} and {test_data_fpath} already exist. Skipping."
        )
        return

    train_df, test_df = load_hf_dataset(data_config, args.hf_cache_dir)

    # Filter out "null" rows in `cmp_results` column for test
    test_df = test_df[test_df["cmp_results"] != "null"]

    console.log(f"Train size: {len(train_df)}")
    console.log(f"Test size: {len(test_df)}")

    train_samples = []
    train_generations = {}  # model_name -> generations
    for i, row in train_df.iterrows():
        # Process row - no rank available for train
        embedding_input = (
            row["instruction"].strip() + "\n\n" + row["input"].strip()
        ).strip()
        train_samples.append(
            {
                "task_name": "mix_instruct",
                "embedding_input": embedding_input,
                "multi_model_prompt": embedding_input,
            }
        )

        # Get generations
        for candidate in row["candidates"]:
            model_name = candidate["model"]
            text = candidate["text"]
            if model_name not in train_generations:
                train_generations[model_name] = []
            train_generations[model_name].append(text)

    # Save train samples as jsonl
    with open(train_data_fpath, "w") as f:
        for sample in train_samples:
            f.write(json.dumps(sample) + "\n")
    console.log(f"Saved train samples to {train_data_fpath}")

    # Save train generations
    for model_name, generations in train_generations.items():
        train_output_fpath, _ = construct_generations_path(
            data_config=data_config, model=model_name, args=args
        )
        out_dict = {"generations": generations}
        train_output_fpath.write_text(json.dumps(out_dict, indent=4))
        console.log(f"Saved train generations for {model_name} to {train_output_fpath}")

    test_samples = []
    test_generations = {}  # model_name -> generations
    for i, row in test_df.iterrows():
        # Get generations
        row_models_to_generations = {}  # model_name -> generation for current row
        for candidate in row["candidates"]:
            model_name = candidate["model"]
            text = candidate["text"]
            row_models_to_generations[model_name] = text
            if model_name not in test_generations:
                test_generations[model_name] = []
            test_generations[model_name].append(text)

        # Process row
        embedding_input = (
            row["instruction"].strip() + "\n\n" + row["input"].strip()
        ).strip()
        ranks = process_reference(eval(row["cmp_results"]), row_models_to_generations)
        test_samples.append(
            {
                "task_name": "mix_instruct",
                "embedding_input": embedding_input,
                "multi_model_prompt": embedding_input,
                "reference": ranks,
            }
        )

    # Save test samples as jsonl
    with open(test_data_fpath, "w") as f:
        for sample in test_samples:
            f.write(json.dumps(sample) + "\n")
    console.log(f"Saved test samples to {test_data_fpath}")

    # Save test generations
    for model_name, generations in test_generations.items():
        _, test_output_fpath = construct_generations_path(
            data_config=data_config, model=model_name, args=args
        )
        out_dict = {"generations": generations}
        test_output_fpath.write_text(json.dumps(out_dict, indent=4))
        console.log(f"Saved test generations for {model_name} to {test_output_fpath}")


if __name__ == "__main__":
    console.log("#" * 30)
    args = parser.parse_args()
    console.log(args)
    main(args)
