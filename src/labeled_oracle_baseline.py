"""
This script implements the Best-on-Val baseline, which selects the best model on the validation set.

In the code, we refer to this as the "labeled oracle" baseline.

Sample command:
python -m src.labeled_oracle \
    --dataset_config dataset_configs/squad.yaml \
    --data_dir "smoothie_data/datasets" \
    --results_dir test_results_multimodel \
    --multi_model
"""

import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)

import argparse
import json

import jsonlines
import numpy as np
import pandas as pd

from src.console import console
from src.data_utils import construct_processed_dataset_paths
from src.ensembles import GSM_8K_GROUPS, MIX_INSTRUCT_GROUPS, MODEL_GROUPS
from src.evaluate.evaluate import evaluate_predictions
from src.utils import (
    check_args,
    clean_generations,
    construct_labeled_oracle_predictions_path,
    get_references,
    load_data_config,
    load_predictions,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", type=str, help="LLM to use. Only used if not multi-model."
)
parser.add_argument(
    "--dataset_config",
    type=str,
    help="Path to config file. This should be a yaml file",
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
    help="Results directory",
)
parser.add_argument(
    "--label_train_n_trials",
    default=10,
    type=int,
    help="Number of trials to run for train oracle sampling method.",
)
parser.add_argument(
    "--label_train_sample_size",
    default=50,
    type=int,
    help="Number of trials to run for train oracle sampling method.",
)
parser.add_argument(
    "--redo",
    action="store_true",
    help="Redo evaluation even if results already exist. Otherwise, we only evaluate methods/metrics which aren't already evaluated.",
)
parser.add_argument(
    "--multi_prompt",
    action="store_true",
)
parser.add_argument(
    "--multi_model",
    action="store_true",
)
parser.add_argument(
    "--n_generations",
    default=1,
    type=int,
    help="If not equal to 1, we replace k-nearest neighbors smoothing with computation over the n_generations per sample",
)
parser.add_argument(
    "--seed",
    default=42,
    type=int,
)

# Mapping from task name to metric
TASK2METRIC = {
    "cnn_dailymail": "rouge2",
    "definition_extraction": "definition_extraction_acc",
    "e2e_nlg": "rouge2",
    "squad": "squad_acc",
    "trivia_qa": "trivia_qa_acc",
    "web_nlg": "rouge2",
    "xsum": "rouge2",
    "mix_instruct": "mix_instruct_rank",
    "gsm8k": "gsm8k_acc",
}


def run_labeled_oracle(
    args: argparse.Namespace,
    data_config: dict,
    model_group_name: str,
    model_group: list,
):
    """
    Run the labeled oracle baseline.

    Args:
        args (argparse.Namespace): arguments from the command line
        data_config (dict): data config
        model_group_name (str): name of the model group
        model_group (list): list of models
    """
    output_fpath = construct_labeled_oracle_predictions_path(
        data_config=data_config,
        model_group_name=model_group_name,
        args=args,
    )
    predictions_dir = output_fpath.parent

    if output_fpath.exists() and not args.redo:
        console.log(f"Results file already exists at {output_fpath}. Skipping.")
        return

    train_generations = load_predictions(
        data_config=data_config,
        split="train",
        models=model_group,
        args=args,
    )
    test_generations = load_predictions(
        data_config=data_config,
        split="test",
        models=model_group,
        args=args,
    )

    train_data_path, _ = construct_processed_dataset_paths(args)
    with jsonlines.open(train_data_path) as file:
        train_dataset = list(file.iter())

    train_references = np.array(get_references(train_dataset), dtype=object)
    train_tasks = np.array([train_df["task_name"] for train_df in train_dataset])

    labeled_oracle_generations = []  # List of generations for each trial
    for _ in range(args.label_train_n_trials):
        all_generator_scores = []  # List of scores for each generator

        # Select a random subset of the training data
        sampled_train_indices = np.random.choice(
            len(train_dataset), args.label_train_sample_size
        )
        sampled_train_references = train_references[sampled_train_indices]
        sampled_train_tasks = train_tasks[sampled_train_indices]
        generator_scores = []
        for generator_idx in range(train_generations.shape[1]):
            sampled_generations = train_generations[:, generator_idx][
                sampled_train_indices
            ]
            generator_scores.append(
                evaluate_predictions(
                    dataset=data_config["dataset"],
                    generations=sampled_generations,
                    references=sampled_train_references,
                    task_names=sampled_train_tasks,
                    metric_map=TASK2METRIC,
                )
            )
        best_generator_idx = np.argmax(generator_scores)
        best_generator_test_generations = test_generations[:, best_generator_idx]
        labeled_oracle_generations.append(best_generator_test_generations.tolist())

    results = {
        "generations": labeled_oracle_generations,
    }
    console.log(f"Saving results to {output_fpath}")
    output_fpath.write_text(json.dumps(results, default=int, indent=4))


def main(args):
    np.random.seed(args.seed)
    console.log(f"Setting random seed to {args.seed}")
    check_args(args)
    data_config = load_data_config(args)
    if args.multi_model:
        if data_config["dataset"] == "mix_instruct":
            model_groups = MIX_INSTRUCT_GROUPS
        elif data_config["dataset"] == "gsm8k":
            model_groups = GSM_8K_GROUPS
        else:
            model_groups = MODEL_GROUPS

        for model_group in model_groups:
            run_labeled_oracle(
                args=args,
                data_config=data_config,
                model_group_name=model_group,
                model_group=model_groups[model_group],
            )

    else:
        run_labeled_oracle(
            args=args,
            data_config=data_config,
            model_group_name="",
            model_group=[args.model],
        )


if __name__ == "__main__":
    args = parser.parse_args()
    console.log(args)
    main(args)
