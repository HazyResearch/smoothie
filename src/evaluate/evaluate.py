"""
This script evaluates predictions for multi-model and multi-prompt tasks. It computes dataset-level scores for each method.

Sample command: 
python -m src.evaluate \
    --dataset_config dataset_configs/cnn_dailymail.yaml \
    --results_dir smoothie_results/multi_model \ 
    --multi_model \
    --redo
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Union

import jsonlines
import numpy as np

from src.console import console
from src.data_utils import construct_processed_dataset_paths
from src.ensembles import GSM_8K_GROUPS, MIX_INSTRUCT_GROUPS, MODEL_GROUPS
from src.evaluate.metrics import METRIC_FUNCS, MULTI_MODEL_TASK2METRIC
from src.utils import (
    check_args,
    clean_generations,
    construct_method_predictions_dir_path,
    get_references,
    load_data_config,
    load_predictions,
)

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="LLM to use")
parser.add_argument(
    "--dataset_config",
    type=str,
    help="Path to config file. This should be a yaml file.",
)
parser.add_argument(
    "--data_dir",
    default="smoothie_data/datasets",
    type=str,
    help="Directory with data files.",
)
parser.add_argument(
    "--multi_prompt",
    action="store_true",
    help="Whether to evaluate multi-prompt results.",
)
parser.add_argument(
    "--multi_model",
    action="store_true",
    help="Whether to evaluate multi-model results.",
)
parser.add_argument(
    "--results_dir",
    type=str,
    help="Directory to save results to.",
)
parser.add_argument(
    "--redo",
    action="store_true",
    help="Redo evaluation even if results already exist.",
)


def evaluate_predictions(
    dataset: str,
    generations: List[str],
    references: Union[str, List[str]],
    task_names: List[str],
    metric_map: Dict,
) -> float:
    """
    Evaluate predictions using the specified metric. Returns the average score across all samples for a given dataset.

    Args:
        dataset: dataset name
        generations: List of generations
        references: List of references. Can be a list of strings or a list of lists of strings
        task_names: List of task names
        metric_map: Dictionary mapping dataset names to metric names

    Returns:
        float: Score
    """
    # Convert everything to numpy arrays for easier indexing
    task_names = np.array(task_names)
    generations = np.array(generations)
    references = np.array(references, dtype=object)

    # Add to scores based on task
    tasks = np.unique(task_names)
    scores = []
    for task_name in tasks:
        metric_func = METRIC_FUNCS[metric_map[task_name]]
        task_idxs = np.where(task_names == task_name)[0]
        # console.log(task_name, task_idxs)
        task_generations = generations[task_idxs]
        if dataset not in ["mix_instruct"]:
            cleaned_generations = clean_generations(task_generations)
        else:
            cleaned_generations = task_generations

        task_references = references[task_idxs]
        scores.extend(
            metric_func(generations=cleaned_generations, references=task_references)
        )

    assert len(scores) == len(generations)
    return np.mean(scores)


def evaluate_multi_model_task(
    args: argparse.Namespace, data_config: Dict, model_group_name: str, models: List[str]
):
    """
    Evaluate predictions for multi-model experiments.

    Args:
        args (argparse.Namespace): arguments
        data_config (Dict): data config
        model_group_name (str): model group name
        models (List[str]): models
    """
    multitask = "tasks" in data_config
    predictions_dir = construct_method_predictions_dir_path(
        data_config=data_config, args=args, model_group_name=model_group_name
    )

    _, test_data_path = construct_processed_dataset_paths(args)
    with jsonlines.open(test_data_path) as file:
        test_dataset = list(file.iter())

    task_names = [sample["task_name"] for sample in test_dataset]

    if multitask:
        scores = {"ensemble": {}, "smoothie": {}}
    else:
        scores = {
            metric: {"ensemble": {}, "smoothie": {}}
            for metric in data_config["metrics"]
        }

    predictions_files = list(predictions_dir.glob("*_test.json"))

    for predictions_fpath in predictions_files:
        fname = predictions_fpath.stem

        if fname.endswith("_train"):
            # Ignore train files
            continue

        if "_gens_" in fname and "smoothie" not in fname:
            continue

        # Method name is everything up to the last underscore
        method = "_".join(fname.split("_")[:-1])

        # Extract references
        references = get_references(test_dataset)
        predictions_dict = json.loads(predictions_fpath.read_text())

        if (
            fname.startswith("labeled_oracle")
            or fname.startswith("pick_random")
            or fname.startswith("labeled_knn")
        ):
            # Predictions for these methods take the shape (n_trials, n_samples). We report the average across trials.
            trial_generations = predictions_dict["generations"]
            if multitask:
                trial_scores = []
                for generations in trial_generations:
                    trial_scores.append(
                        evaluate_predictions(
                            dataset=data_config["dataset"],
                            generations=generations,
                            references=references,
                            task_names=task_names,
                            metric_map=MULTI_MODEL_TASK2METRIC,
                        )
                    )
                mean_score = np.mean(trial_scores)
                scores[method] = mean_score
            else:
                for metric in data_config["metrics"]:
                    trial_scores = []
                    for generations in trial_generations:
                        trial_scores.append(
                            evaluate_predictions(
                                dataset=data_config["dataset"],
                                generations=generations,
                                references=references,
                                task_names=task_names,
                                metric_map={data_config["dataset"]: metric},
                            )
                        )
                    mean_score = np.mean(trial_scores)
                    scores[metric][method] = mean_score
        else:
            generations = predictions_dict["generations"]
            if multitask:
                score = evaluate_predictions(
                    dataset=data_config["dataset"],
                    generations=generations,
                    references=references,
                    task_names=task_names,
                    metric_map=MULTI_MODEL_TASK2METRIC,
                )
                if fname.startswith("smoothie"):
                    scores["smoothie"][method] = score
                elif fname.startswith("pair_rm"):
                    scores[method] = score
                else:
                    scores["ensemble"][method] = score
            else:
                for metric in data_config["metrics"]:
                    score = evaluate_predictions(
                        dataset=data_config["dataset"],
                        generations=generations,
                        references=references,
                        task_names=task_names,
                        metric_map={data_config["dataset"]: metric},
                    )
                    if fname.startswith("smoothie"):
                        scores[metric]["smoothie"][method] = score
                    elif fname.startswith("pair_rm"):
                        scores[metric][method] = score
                    else:
                        scores[metric]["ensemble"][method] = score

    # Compute scores for each model in the model group
    generations = load_predictions(
        data_config=data_config,
        args=args,
        models=models,
        split="test",
    )
    for idx in range(generations.shape[1]):
        if multitask:
            score = evaluate_predictions(
                dataset=data_config["dataset"],
                generations=generations[:, idx],
                references=references,
                task_names=task_names,
                metric_map=MULTI_MODEL_TASK2METRIC,
            )
            scores["ensemble"][models[idx]] = score
        else:
            for metric in data_config["metrics"]:
                score = evaluate_predictions(
                    dataset=data_config["dataset"],
                    generations=generations[:, idx],
                    references=references,
                    task_names=task_names,
                    metric_map={data_config["dataset"]: metric},
                )
                scores[metric]["ensemble"][models[idx]] = score

    # Save scores
    scores_path = predictions_dir / "scores.json"
    console.log(f"Saving scores to {scores_path}")
    scores_path.write_text(json.dumps(scores, indent=4))


def evaluate_multi_prompt_task(args: argparse.Namespace, data_config: Dict):
    """
    Evaluate predictions for multi-prompt experiments.

    Args:
        args (argparse.Namespace): arguments
        data_config (Dict): data config
    """
    multitask = False  # We don't run the multi-prompt experiments on the multi-task datasets
    predictions_dir = construct_method_predictions_dir_path(
        data_config=data_config, args=args, model_group_name=""
    )

    _, test_data_path = construct_processed_dataset_paths(args)
    with jsonlines.open(test_data_path) as file:
        test_dataset = list(file.iter())

    task_names = [sample["task_name"] for sample in test_dataset]

    # Construct scores dict. We save a single dictionary with scores for all metrics and methods.
    scores = {
        metric: {"ensemble": {}, "smoothie": {}}
        for metric in data_config["metrics"]
    }

    # We load all predictions files in the results directory. The name of the file indicates the method.
    predictions_files = list(predictions_dir.glob("*_test.json"))
    for predictions_fpath in predictions_files:
        fname = predictions_fpath.stem

        # Ignore train files
        if fname.endswith("_train"):
            continue

        # Method name is everything up to the last underscore
        method = "_".join(fname.split("_")[:-1])
        console.log(fname, method)
        if "_gens_" in fname and "smoothie" not in fname:
            continue

        # Extract references
        references = get_references(test_dataset)
        predictions_dict = json.loads(predictions_fpath.read_text())

        if fname.startswith("labeled_oracle") or fname.startswith("pick_random"):
            # Predictions from labeled oracle and pick_random baseline take the shape (n_trials, n_samples). We report the average across trials.
            trial_generations = predictions_dict["generations"]
            for metric in data_config["metrics"]:
                trial_scores = []
                for generations in trial_generations:
                    trial_scores.append(
                        evaluate_predictions(
                            dataset=data_config["dataset"],
                            generations=generations,
                            references=references,
                            task_names=task_names,
                            metric_map={data_config["dataset"]: metric},
                        )
                    )
                mean_score = np.mean(trial_scores)
                scores[metric][method] = mean_score
        elif fname.startswith("individual_"):
            # Files starting with "individual_" contain generations for each prompt. We compute the average score for each prompt.
            # Note: there should only be one file in the directory with this naming convention.
            generations = np.array(predictions_dict["generations"])
            for metric in data_config["metrics"]:
                for prompt_idx in range(generations.shape[1]):
                    prompt_scores = evaluate_predictions(
                        dataset=data_config["dataset"],
                        generations=generations[:, prompt_idx],
                        references=references,
                        task_names=task_names,
                        metric_map=MULTI_MODEL_TASK2METRIC,
                    )

                    mean_score = np.mean(prompt_scores)
                    scores[metric]["ensemble"][f"prompt_{prompt_idx}"] = mean_score
        else:
            generations = predictions_dict["generations"]
            for metric in data_config["metrics"]:
                score = evaluate_predictions(
                    dataset=data_config["dataset"],
                    generations=generations,
                    references=references,
                    task_names=task_names,
                    metric_map={data_config["dataset"]: metric},
                )
                if fname.startswith("smoothie"):
                    scores[metric]["smoothie"][method] = score
                else:
                    scores[metric]["ensemble"][method] = score

    # Save scores
    scores_path = predictions_dir / "scores.json"
    console.log(f"Saving scores to {scores_path}")
    scores_path.write_text(json.dumps(scores, indent=4))


def main(args):
    args.n_generations = 1
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
            evaluate_multi_model_task(
                args=args,
                data_config=data_config,
                model_group_name=model_group,
                models=model_groups[model_group],
            )
    else:
        evaluate_multi_prompt_task(
            args=args,
            data_config=data_config,
        )


if __name__ == "__main__":
    args = parser.parse_args()
    console.log(args)
    main(args)
