"""
This script implements Smoothie train-time. In this version we only use generations from a holdout train set.
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

from src.console import console
from src.constants import *
from src.data_utils import construct_processed_dataset_paths
from src.model import Smoothie
from src.utils import (check_args, construct_smoothie_train_time_predictions_path,
                       embed_individual_generations, load_data_config,
                       load_predictions)

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="LLM to use")
parser.add_argument("--device", default="cuda", type=str, help="Device to use")
parser.add_argument("--dataset_config", type=str, help="Path to the data yaml config.")
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
)
parser.add_argument(
    "--redo", action="store_true", help="Redo the generation if the file already exists"
)
parser.add_argument(
    "--test",
    action="store_true",
    help="Runs the script in test mode. This will only generate predictions for two samples.",
)
parser.add_argument(
    "--type",
    choices=["sample_dependent", "sample_independent"],
    required=True,
    help="The type of Smoothie to use. See file docstring for more information.",
)
parser.add_argument("--k", help="Nearest neighbors size", type=int)
parser.add_argument(
    "--model_group",
    help="The models to use for predictions if we are doing multi-model",
)
parser.add_argument(
    "--multi_prompt",
    action="store_true",
)
parser.add_argument(
    "--multi_model",
    action="store_true",
)


def main(args):
    args.n_generations = 1 # only support 1 for now

    check_args(args)
    data_config = load_data_config(args)
    output_fpath = construct_smoothie_train_time_predictions_path(data_config, args.model, args)
    predictions_dir = output_fpath.parent
    if output_fpath.exists() and not args.redo:
        console.log(f"Results file already exists at {output_fpath}. Skipping.")
        return

    # Load data and get embeddings for train and test x
    model_name = "all-mpnet-base-v2"
    model = SentenceTransformer(model_name)
    console.log(f"Loaded embedding model: {model_name}")

    train_dataset_path, test_dataset_path = construct_processed_dataset_paths(args)
    with jsonlines.open(train_dataset_path) as file:
        train_dataset = list(file.iter())
    train_inputs = [sample["embedding_input"] for sample in train_dataset]
    train_input_embeddings = model.encode(train_inputs)

    with jsonlines.open(test_dataset_path) as file:
        test_dataset = list(file.iter())
    test_inputs = [sample["embedding_input"] for sample in test_dataset]
    test_input_embeddings = model.encode(test_inputs)


    # Compute embeddings for train generations
    train_generations_for_smoothie = load_predictions(
        predictions_dir, "train", args, for_selection=False
    )
    smoothie_embeddings = embed_individual_generations(
        individual_generations=train_generations_for_smoothie, model_name=model_name
    )

    n_samples = len(test_dataset)
    n_voters = smoothie_embeddings.shape[1]
    embed_dim = smoothie_embeddings.shape[2]

    if args.type == "sample_dependent":
        # use KNN
        nbrs = NearestNeighbors(n_neighbors=args.k, algorithm="auto")
        nbrs.fit(
            train_input_embeddings
        )  # not the same as smoothie_embeddings! only kernel-smooth based on x similarity

        _, test_indices = nbrs.kneighbors(test_input_embeddings)

        smoothie_dataset_weights = []
        for sample_idx in range(n_samples):
            if args.k == 1:
                embs_per_sample = smoothie_embeddings[sample_idx].reshape(
                    (1, n_voters, -1)
                )
            else:
                embs_per_sample = smoothie_embeddings[test_indices[sample_idx]]
            smoothie = Smoothie(n_voters=n_voters, dim=embed_dim)
            smoothie.fit(embs_per_sample)
            smoothie_dataset_weights.append(smoothie.theta)
        smoothie_dataset_weights = np.array(smoothie_dataset_weights)
    else:
        # learn a single set of weights for all samples
        smoothie = Smoothie(n_voters=n_voters, dim=embed_dim)
        smoothie.fit(smoothie_embeddings)
        smoothie_dataset_weights = np.tile(smoothie.theta, (n_samples, 1))


    # Select test generations based on smoothie weights
    test_generations_for_selection = load_predictions(predictions_dir, "test", args)
    dataset_texts = []
    for sample_idx in range(n_samples):
        max_idx = smoothie_dataset_weights[sample_idx].argmax()
        text = test_generations_for_selection[sample_idx][max_idx]
        dataset_texts.append(text)

        if args.test and sample_idx == 1:
            break

    results = {
        "generations": dataset_texts,
        "smoothie_weights": smoothie_dataset_weights.tolist(),
    }
    console.log(f"Saving results to {output_fpath}")
    output_fpath.write_text(json.dumps(results, default=int, indent=4))


if __name__ == "__main__":
    console.log("#" * 30)
    args = parser.parse_args()
    console.log(args)
    main(args)
