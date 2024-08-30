"""
This script creates and processed datasets for different tasks.


Currently supported single-task datasets:
    - squad
    - trivia_qa
    - xsum
    - cnn_dailymail
    - definition_extraction
    - e2e_nlg
    - web_nlg

Currently supported multi-task datasets:
    - acc_group (squad, trivia_qa, definition_extraction)
    - rouge2_group (cnn_dailymail, xsum, e2e_nlg, web_nlg)
"""

import argparse
import json
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

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_config",
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
    "--data_dir",
    default="smoothie_data/datasets",
    type=str,
    help="Directory with data files",
)
parser.add_argument(
    "--prompt_templates_dir",
    default="prompt_templates",
    type=str,
    help="Directory with prompt templates",
)
parser.add_argument(
    "--redo",
    action="store_true",
)


def load_task_dataset(
    config: Dict, args: argparse.Namespace
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load a dataset for a task.

    Args:
        config (Dict): task config
        args (argparse.Namespace): command line arguments

    Returns:
        pd.DataFrame: dataset
    """
    # Extract config name from file name
    config_name = Path(args.dataset_config).stem

    # Load dataset
    train_df, test_df = load_hf_dataset(
        config=config,
        hf_cache_dir=args.hf_cache_dir,
    )
    console.log(f"Loaded {config_name} with {len(train_df)} training examples.")
    console.log(f"Loaded {config_name} with {len(test_df)} testing examples.")

    # Generate multi-prompt prompts
    fpath = Path(args.prompt_templates_dir) / f"{config['multi_prompt_template']}.json"
    multi_prompt_templates = json.loads(fpath.read_text())
    console.log(f"Multi prompt templates:", multi_prompt_templates)

    for idx, prompt_template in enumerate(multi_prompt_templates):
        train_df[f"multi_prompt_{idx}"] = train_df.apply(
            lambda row: generate_prompt(
                config=config, prompt_template=prompt_template, row=row
            ),
            axis=1,
        )
        test_df[f"multi_prompt_{idx}"] = test_df.apply(
            lambda row: generate_prompt(
                config=config, prompt_template=prompt_template, row=row
            ),
            axis=1,
        )

    # Generate multi-model prompts
    fpath = (
        Path(args.prompt_templates_dir) / f"{config['multi_model_prompt_template']}.txt"
    )
    multi_model_prompt_template = fpath.read_text()
    console.log("Multi model prompt template:", multi_model_prompt_template)
    train_df["multi_model_prompt"] = train_df.apply(
        lambda row: generate_prompt(
            config=config, prompt_template=multi_model_prompt_template, row=row
        ),
        axis=1,
    )
    test_df["multi_model_prompt"] = test_df.apply(
        lambda row: generate_prompt(
            config=config, prompt_template=multi_model_prompt_template, row=row
        ),
        axis=1,
    )

    # Add embedding input text
    train_df["embedding_input"] = train_df.apply(
        lambda row: get_embedding_inputs(config=config, row=row), axis=1
    )
    test_df["embedding_input"] = test_df.apply(
        lambda row: get_embedding_inputs(config=config, row=row), axis=1
    )

    # Add reference
    train_df["reference"] = train_df.apply(
        lambda row: get_reference(config=config, row=row), axis=1
    )
    test_df["reference"] = test_df.apply(
        lambda row: get_reference(config=config, row=row), axis=1
    )

    # Add task name to each row
    train_df["task_name"] = config["dataset"]
    test_df["task_name"] = config["dataset"]

    # Remove columns that are not needed
    columns_to_keep = [
        "task_name",
        "reference",
        "embedding_input",
        "multi_model_prompt",
    ]
    columns_to_keep.extend(
        [f"multi_prompt_{idx}" for idx in range(len(multi_prompt_templates))]
    )
    train_df = train_df[columns_to_keep]
    test_df = test_df[columns_to_keep]

    return train_df, test_df


def main(args):
    # Load yaml configs file
    data_config = yaml.load(
        Path(args.dataset_config).read_text(), Loader=yaml.FullLoader
    )
    console.log(data_config)

    train_fpath, test_fpath = construct_processed_dataset_paths(args)
    if train_fpath.exists() and test_fpath.exists() and not args.redo:
        console.log(
            f"Processed datasets {train_fpath} and {test_fpath} already exist. Skipping."
        )
        return

    if "tasks" in data_config:
        # Multitask dataset
        config_name = Path(args.dataset_config).stem
        task_yaml_files = data_config["tasks"]
        train_dfs, test_dfs = [], []
        for task_yaml_file in task_yaml_files:
            task_config = yaml.load(
                Path(task_yaml_file).read_text(), Loader=yaml.FullLoader
            )
            console.log(task_config)
            train_df, test_df = load_task_dataset(config=task_config, args=args)
            train_dfs.append(train_df)
            test_dfs.append(test_df)

        # Concatenate dataframes
        train_df = pd.concat(train_dfs, axis=0)
        test_df = pd.concat(test_dfs, axis=0)
        console.log(f"Concatenated training dataframes with {len(train_df)} examples.")
        console.log(f"Concatenated testing dataframes with {len(test_df)} examples.")

    else:
        # Single task dataset
        train_df, test_df = load_task_dataset(config=data_config, args=args)

    # Add index to each row, corresponding to the number of the row
    train_df["idx"] = train_df.reset_index().index
    test_df["idx"] = test_df.reset_index().index

    # Save dataframes as JSONL files
    train_df.to_json(train_fpath, orient="records", lines=True)
    console.log(f"Saved training data to {train_fpath}")
    test_df.to_json(test_fpath, orient="records", lines=True)
    console.log(f"Saved testing data to {test_fpath}")


if __name__ == "__main__":
    console.log("#" * 30)
    args = parser.parse_args()
    console.log(args)
    main(args)
