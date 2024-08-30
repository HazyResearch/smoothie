"""
This script implements the labeled KNN method for the multi-model setting.
"""

import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)

import argparse
import json

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm

from src.console import console
from src.data_utils import construct_processed_dataset_paths
from src.evaluate.metrics import *
from src.utils import (
    construct_labeled_knn_predictions_path,
    clean_generations,
    load_data_config,
    load_predictions,
    get_references,
)

parser = argparse.ArgumentParser()
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
    "--model_group",
    help="The models to use for predictions",
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


TASK2METRIC = {
    "cnn_dailymail": METRIC_FUNCS["rouge2"],
    "definition_extraction": METRIC_FUNCS["definition_extraction_acc"],
    "e2e_nlg": METRIC_FUNCS["rouge2"],
    "squad": METRIC_FUNCS["squad_acc"],
    "trivia_qa": METRIC_FUNCS["trivia_qa_acc"],
    "web_nlg": METRIC_FUNCS["rouge2"],
    "xsum": METRIC_FUNCS["rouge2"],
}


def main(args):
    np.random.seed(args.seed)
    if not args.multi_model or args.multi_prompt:
        raise ValueError("Labeled KNN baseline only supported for --multi_model.")
    # Load data config, prompts, and create output directory
    data_config = load_data_config(args)
    console.log(data_config)
    output_fpath = construct_labeled_knn_predictions_path(data_config, None, args)
    predictions_dir = output_fpath.parent
    # Check if the results file already exists
    if output_fpath.exists() and not args.redo:
        console.log(f"Results file already exists at {output_fpath}. Skipping.")
        return

    # Load individual train and test generations.
    train_generations = load_predictions(predictions_dir, "train", args)
    test_generations = load_predictions(predictions_dir, "test", args)
    n_models = test_generations.shape[1]

    # Load train and test dataset
    train_data_path, test_data_path = construct_processed_dataset_paths(args)
    train_dataset = pd.read_csv(train_data_path)
    test_dataset = pd.read_csv(test_data_path)

    train_references = get_references(train_dataset)

    # Embed train and test generations
    model_name = "all-mpnet-base-v2"
    model = SentenceTransformer(model_name)
    console.log(f"Loaded embedding model: {model_name}")
    train_dataset_embeddings = model.encode(train_dataset["embedding_input"])
    test_dataset_embeddings = model.encode(test_dataset["embedding_input"])

    # Evaluate train generations
    if data_config["dataset"] not in TASK2METRIC:
        raise ValueError(f"Dataset {data_config['dataset']} not supported.")
    metric_func = TASK2METRIC[data_config["dataset"]]

    generations = []  # Final generations. Shape: (10, n_test)
    for _ in tqdm(range(10)):
        # Sample labeled train set
        sampled_idxs = np.random.choice(
            len(train_generations), args.label_train_sample_size
        )
        sampled_references = [train_references[idx] for idx in sampled_idxs]
        sampled_train_embeddings = train_dataset_embeddings[sampled_idxs]

        # Compute scores for each model
        model_scores = []
        for prompt_idx in range(n_models):
            sampled_generations = train_generations[sampled_idxs, prompt_idx]
            cleaned_generations = clean_generations(sampled_generations, data_config)
            scores = metric_func(cleaned_generations, sampled_references)
            model_scores.append(scores)
        model_scores = np.array(model_scores).T  # shape: (sample_size, n_models)

        # Train KNN
        nbrs = NearestNeighbors(n_neighbors=20, algorithm="auto")
        nbrs.fit(sampled_train_embeddings)

        # Find the k nearest neighbors
        _, sample_train_indices = nbrs.kneighbors(test_dataset_embeddings)

        # Compute best model on average over these indices
        trial_generations = []
        for idx in range(len(sample_train_indices)):
            sample_model_scores = model_scores[sample_train_indices[idx]].mean(axis=0)
            max_idxs = np.where(sample_model_scores == sample_model_scores.max())[0]
            selected_idx = np.random.choice(max_idxs)
            trial_generations.append(test_generations[idx, selected_idx])
        generations.append(trial_generations)

    results = {
        "generations": generations,
    }
    console.log(f"Saving results to {output_fpath}")
    output_fpath.write_text(json.dumps(results, default=int, indent=4))


if __name__ == "__main__":
    console.log("#" * 30)
    args = parser.parse_args()
    console.log(args)
    main(args)
