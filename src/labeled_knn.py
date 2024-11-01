"""
This script implements the labeled KNN method for the multi-model setting.
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
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm

from src.console import console
from src.data_utils import construct_processed_dataset_paths
from src.evaluate.evaluate import evaluate_predictions
from src.utils import (
    check_args,
    construct_labeled_knn_predictions_path,
    load_data_config,
    load_predictions,
    get_references,
    Embedder
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
    "--results_dir",
    default="generative_ensembles_data/multi_model_results",
    type=str,
    help="Directory to save results to",
)
parser.add_argument(
    "--redo",
    action="store_true",
    help="Redo the generation if the results file already exists",
)
parser.add_argument(
    "--label_train_sample_size",
    default=50,
    type=int,
    help="Number of trials to run for train knn sampling method.",
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
parser.add_argument(
    "--label_train_n_trials",
    default=10,
    type=int,
    help="Number of trials to run for train oracle sampling method.",
)
parser.add_argument(
    "--embedding_model",
    type=str,
    choices=["all-mpnet-base-v2", "bge-small-en-v1.5"],
    help="Model to use for embedding generations."
)


TASK2METRIC = {
    "cnn_dailymail": "rouge2",
    "definition_extraction": "definition_extraction_acc",
    "e2e_nlg": "rouge2",
    "squad": "squad_acc",
    "trivia_qa": "trivia_qa_acc",
    "web_nlg": "rouge2",
    "xsum": "rouge2",
    "gsm8k": "gsm8k_acc"
}


def run_labeled_knn(args, data_config, model_group_name, model_group, embedder):
    """
    Run the labeled knn baseline.

    Args:
        args (argparse.Namespace): arguments from the command line
        data_config (dict): data config
        model_group (str): name of the model group
        embedder (Embedder): embedder
    """
    output_fpath = construct_labeled_knn_predictions_path(
        data_config=data_config,
        model_group_name=model_group_name,
        args=args
    )
    predictions_dir = output_fpath.parent
    
    # Check if the results file already exists
    if output_fpath.exists() and not args.redo:
        console.log(f"Results file already exists at {output_fpath}. Skipping.")
        return 

    # Load generations
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

    # Load dataset
    train_data_path, test_data_pth = construct_processed_dataset_paths(args)
    with jsonlines.open(train_data_path) as file:
        train_dataset = list(file.iter())

    with jsonlines.open(test_data_pth) as file:
        test_dataset = list(file.iter())

    train_references = np.array(get_references(train_dataset), dtype=object)
    train_tasks = np.array([train_df["task_name"] for train_df in train_dataset])

    # Embed dataset
    train_input_embeddings = embedder.embed_dataset(train_dataset)
    test_input_embeddings = embedder.embed_dataset(test_dataset)

    # Run labeled-knn on multiple trials, where each trial samples a different "train" subset
    final_generations = [] # Final generations. Shape: (args.label_train_n_trials, n_test)
    for _ in range(args.label_train_n_trials):
        sampled_idxs = np.random.choice(
            len(train_generations), args.label_train_sample_size
        )
        sampled_train_references = [train_references[idx] for idx in sampled_idxs]
        sampled_train_tasks = [train_tasks[idx] for idx in sampled_idxs]
        sampled_train_embeddings = train_input_embeddings[sampled_idxs]

        # Compute scores for each model over the train sample.
        model_scores = [] # shape: (n_models)
        for prompt_idx in range(train_generations.shape[1]):
            sampled_generations = train_generations[sampled_idxs, prompt_idx]
            model_scores.append(
                evaluate_predictions(
                    dataset=data_config["dataset"],
                    generations=sampled_generations,
                    references=sampled_train_references,
                    task_names=sampled_train_tasks,
                    metric_map=TASK2METRIC,
                )
            )
        model_scores = np.array(model_scores)
        

        # Train KNN
        nbrs = NearestNeighbors(n_neighbors=20, algorithm="auto")
        nbrs.fit(sampled_train_embeddings)

        # Find the k nearest neighbors
        _, sample_train_indices = nbrs.kneighbors(test_input_embeddings)

        # Compute best model on average over these indices
        trial_generations = []
        for idx in range(len(sample_train_indices)):
            max_idxs = np.where(model_scores == model_scores.max())[0]
            selected_idx = np.random.choice(max_idxs) # If there is a tie
            trial_generations.append(test_generations[idx, selected_idx])
        final_generations.append(trial_generations)

    results = {
        "generations": final_generations,
    }
    console.log(f"Saving results to {output_fpath}")
    output_fpath.write_text(json.dumps(results, default=int, indent=4))


def main(args):
    np.random.seed(args.seed)
    console.log(f"Setting random seed to {args.seed}")
    check_args(args)
    data_config = load_data_config(args)
    embedder = Embedder(model_name=args.embedding_model)
    if args.multi_model:
        if data_config["dataset"] == "mix_instruct":
            model_groups = MIX_INSTRUCT_GROUPS
        elif data_config["dataset"] == "gsm8k":
            model_groups = GSM_8K_GROUPS
        else:
            model_groups = MODEL_GROUPS

        for model_group in model_groups:
            run_labeled_knn(
                args=args, 
                data_config=data_config, 
                model_group_name=model_group,
                model_group=model_groups[model_group],
                embedder=embedder
            )

    else:
        run_labeled_oracle(
            args=args, 
            data_config=data_config, 
            model_group=""
        )
    

if __name__ == "__main__":
    args = parser.parse_args()
    console.log(args)
    main(args)