"""
This script evaluates individual methods. It saves scores on a per-sample basis.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Union
import numpy as np
import jsonlines

from src.console import console
from src.data_utils import construct_processed_dataset_paths
from src.evaluate.metrics import METRIC_FUNCS, MULTI_MODEL_TASK2METRIC
from src.utils import (
    check_args,
    clean_generations,
    construct_method_predictions_dir_path,
    load_predictions,
    get_references,
    load_data_config,
)
from src.ensembles import MODEL_GROUPS, MIX_INSTRUCT_GROUPS, GSM_8K_GROUPS

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="LLM to use")
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
    "--multi_prompt",
    action="store_true",
)
parser.add_argument(
    "--multi_model",
    action="store_true",
)
parser.add_argument(
    "--results_dir",
    type=str,
    help="Results directory",
)
parser.add_argument(
    "--redo",
    action="store_true",
    help="Redo evaluation even if results already exist. Otherwise, we only evaluate methods/metrics which aren't already evaluated.",
)



def evaluate_predictions_individual(
    dataset: str,
    generations: List[str],
    references: Union[str, List[str]],
    task_names: List[str],
    metric_map: Dict,
) -> List[float]:
    """
    Evaluate predictions using the specified metric.

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
    scores = np.zeros(len(generations))
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
        scores[task_idxs] = metric_func(generations=cleaned_generations, references=task_references)

    assert len(scores) == len(generations)
    return scores.tolist()


def evaluate_multi_model_task(args, data_config, model_group_name, models):
    """
    Evaluate predictions for a single task.

    Args:
        args: arguments
        data_config: data config
        model_group_name: model group name
        models: models
    """
    multitask = "tasks" in data_config
    predictions_dir = construct_method_predictions_dir_path(
        data_config=data_config, 
        args=args, 
        model_group_name=model_group_name
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
        if fname.startswith("labeled_oracle") or fname.startswith("pick_random") or fname.startswith("labeled_knn"):
            continue
        
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

  
        generations = predictions_dict["generations"]
        if multitask:
            sample_scores = evaluate_predictions_individual(
                dataset=data_config["dataset"],
                generations=generations,
                references=references,
                task_names=task_names,
                metric_map=MULTI_MODEL_TASK2METRIC,
            )
            if fname.startswith("smoothie"):
                scores["smoothie"][method] = sample_scores
            elif fname.startswith("pair_rm"):
                scores[method] = sample_scores
            else:
                scores["ensemble"][method] = sample_scores
        else:
            for metric in data_config["metrics"]:
                sample_scores = evaluate_predictions_individual(
                    dataset=data_config["dataset"],
                    generations=generations,
                    references=references,
                    task_names=task_names,
                    metric_map={data_config["dataset"]: metric},
                )
                if fname.startswith("smoothie"):
                    scores[metric]["smoothie"][method] = sample_scores
                elif fname.startswith("pair_rm"):
                    scores[metric][method] = sample_scores
                else:
                    scores[metric]["ensemble"][method] = sample_scores

    # Compute scores for each model in the model group
    generations = load_predictions(
        data_config=data_config,
        args=args,
        models=models,
        split="test",
    )
    for idx in range(generations.shape[1]):
        if multitask:
            sample_scores = evaluate_predictions_individual(
                dataset=data_config["dataset"],
                generations=generations[:, idx],
                references=references,
                task_names=task_names,
                metric_map=MULTI_MODEL_TASK2METRIC,
            )
            scores["ensemble"][models[idx]] = sample_scores
        else:
            for metric in data_config["metrics"]:
                sample_scores = evaluate_predictions_individual(
                    dataset=data_config["dataset"],
                    generations=generations[:, idx],
                    references=references,
                    task_names=task_names,
                    metric_map={data_config["dataset"]: metric},
                )
                scores[metric]["ensemble"][models[idx]] = sample_scores
    

    # Save scores
    scores_path = predictions_dir / "sample_scores.json"
    console.log(f"Saving scores to {scores_path}")
    scores_path.write_text(json.dumps(scores, indent=4))


def evaluate_multi_prompt_task(args, data_config):
    """
    Evaluate predictions for a single task.

    Args:
        args: arguments
        data_config: data config
    """
    multitask = "tasks" in data_config
    predictions_dir = construct_method_predictions_dir_path(
        data_config=data_config, 
        args=args, 
        model_group=""
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
        
        # Method name is everything up to the last underscore
        method = "_".join(fname.split("_")[:-1])

        if "_gens_" in fname and "smoothie" not in fname:
            continue

        # Extract references
        references = get_references(test_dataset)
        predictions_dict = json.loads(predictions_fpath.read_text())

        if fname.startswith("labeled_oracle") or fname.startswith("pick_random"):
            # Predictions from labeled oracle and pick_random baseline take the shape (n_trials, n_samples). We report the average.
            trial_generations = predictions_dict["generations"]
            if multitask:
                trial_scores = []
                for generations in trial_generations:
                    trial_scores.append(
                        evaluate_predictions(
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
                                generations=generations,
                                references=references,
                                task_names=task_names,
                                metric_map={data_config["dataset"]: metric},
                            )
                        )
                    mean_score = np.mean(trial_scores)
                    scores[metric][method] = mean_score
        elif fname.startswith("individual_"):
            generations = np.array(predictions_dict["generations"])
            for metric in data_config["metrics"]:
                for prompt_idx in range(generations.shape[1]):
                    prompt_scores =  evaluate_predictions(
                        generations=generations[:, prompt_idx],
                        references=references,
                        task_names=task_names,
                        metric_map=MULTI_MODEL_TASK2METRIC,
                    )
                    
                    mean_score = np.mean(prompt_scores)
                    scores[metric]["ensemble"][f"prompt_{prompt_idx}"] = mean_score
        else:
            generations = predictions_dict["generations"]
            if multitask:
                score = evaluate_predictions(
                    generations=generations,
                    references=references,
                    task_names=task_names,
                    metric_map=MULTI_MODEL_TASK2METRIC,
                )
                if fname.startswith("smoothie"):
                    scores["smoothie"][method] = score
                else:
                    scores["ensemble"][method] = score
            else:
                for metric in data_config["metrics"]:
                    score = evaluate_predictions(
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

    # Load data config, prompts, and create output directory
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
                models=model_groups[model_group]
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