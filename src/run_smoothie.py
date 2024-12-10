"""
This script runs the different versions of Smoothie. There are two flags which control the version: --type and --regime.

--type can be either "sample_dependent" or "sample_independent". When set to "sample_indepedent", this runs the version of Smoothie described in the paper as "Smoothie-Global". In this version, we learn a single set of weights for all samples. When set to "sample_dependent", this runs the version of Smoothie described as "Smoothie-Local". In this version, we learn a set of weights for each sample.

--regime can be either "train_time" or "test_time". When set to "train_time", we have access to generations from multiple models at train time, and we learn Smoothie weights using these generations. We assume at test time that we do not have access to generations from multiple models, and thus use the Smoothie weights learned on the training data to select which model to produce a generation from. When set to "test_time", we assume that we do not have access to generations from multiple models over a training set, but do have access to generations from multiple models at test time. Thus, we learn Smoothie weights over these test-time generations.

The paper mainly focuses on the "test_time" regime. 

Sample command:
python -m src.run_smoothie \
    --dataset_config dataset_configs/squad.yaml \
    --results_dir test_results_multimodel \
    --type sample_independent \
    --regime test_time \
    --embedding_model "all-mpnet-base-v2"
"""

import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)


import argparse
import json
from typing import Dict

import jsonlines
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from src.console import console
from src.constants import *
from src.data_utils import construct_processed_dataset_paths
from src.ensembles import GSM_8K_GROUPS, MIX_INSTRUCT_GROUPS, MODEL_GROUPS
from src.model import Smoothie
from src.utils import (
    Embedder,
    check_args,
    construct_smoothie_predictions_path,
    load_data_config,
    load_predictions,
)

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
parser.add_argument(
    "--use_full_text_embeddings",
    action="store_true",
    help="If set to true, Smoothie operates on embeddings of [input text, generation text]. Otherwise, Smoothie uses the embedding of the generation text only.",
)
parser.add_argument(
    "--k",
    type=int,
    help="Nearest neighborhood size. Only used if --type is set to sample_dependent.",
)
parser.add_argument(
    "--n_generations",
    default=1,
    type=int,
    help="If not equal to 1, we replace k-nearest neighbors smoothing with computation over the n_generations per sample",
)
parser.add_argument(
    "--multi_prompt",
    action="store_true",
    help="If set to true, runs Smoothie in multi-prompt mode.",
)
parser.add_argument(
    "--multi_model",
    action="store_true",
    help="If set to true, runs Smoothie in multi-model mode.",
)
parser.add_argument(
    "--regime",
    choices=["train_time", "test_time"],
    required=True,
    help="Whether to run Smoothie in train-time or test-time regime.",
)
parser.add_argument(
    "--embedding_model",
    type=str,
    choices=["all-mpnet-base-v2", "bge-small-en-v1.5"],
    help="Model to use for embedding generations.",
)


def train_time_smoothie(
    args: argparse.Namespace,
    data_config: dict,
    model_group_name: str,
    model_group: list,
    embedder: Embedder,
):
    """
    Runs version of Smoothie where we have access to generations from multiple models at train time.

    Args:
        args (argparse.Namespace): arguments from the command line
        model_group (str): name of the model group
        embedder (Embedder): embedder to use
    """

    output_fpath = construct_smoothie_predictions_path(
        data_config=data_config,
        model=args.model,
        model_group_name=model_group_name,
        args=args,
    )
    if output_fpath.exists() and not args.redo:
        console.log(f"Results file already exists at {output_fpath}. Skipping.")
        return

    # Load data and get embeddings for train and test x
    console.log(f"Loaded embedding model: {embedder.model_name}")

    train_dataset_path, test_dataset_path = construct_processed_dataset_paths(args)
    with jsonlines.open(train_dataset_path) as file:
        train_dataset = list(file.iter())
    train_input_embeddings = embedder.embed_dataset(train_dataset)

    with jsonlines.open(test_dataset_path) as file:
        test_dataset = list(file.iter())
    test_input_embeddings = embedder.embed_dataset(test_dataset)

    # Compute embeddings for train generations
    train_generations_for_smoothie = load_predictions(
        data_config=data_config,
        split="train",
        models=model_group,
        args=args,
        for_selection=False,
    )
    clean = data_config["dataset"] not in ["mix_instruct", "alpaca", "gsm8k"]
    smoothie_embeddings = embedder.embed_individual_generations(
        individual_generations=train_generations_for_smoothie, clean=clean
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
    test_generations_for_selection = load_predictions(
        data_config=data_config,
        split="test",
        models=model_group,
        args=args,
    )
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


def test_time_smoothie(
    args: argparse.Namespace,
    data_config: dict,
    model_group_name: str,
    model_group: list,
    embedder: Embedder,
):
    """
    Runs version of Smoothie where we have access to generations from multiple models at test time.

    Args:
        args (argparse.Namespace): arguments from the command line
        model_group (str): name of the model group
        embedder (Embedder): embedder to use
    """

    output_fpath = construct_smoothie_predictions_path(
        data_config=data_config,
        model=args.model,
        model_group_name=model_group_name,
        args=args,
    )
    if output_fpath.exists() and not args.redo:
        console.log(f"Results file already exists at {output_fpath}. Skipping.")
        return

    _, test_dataset_path = construct_processed_dataset_paths(args)
    with jsonlines.open(test_dataset_path) as file:
        test_dataset = list(file.iter())
    test_input_embeddings = embedder.embed_dataset(test_dataset)

    test_generations_for_smoothie = load_predictions(
        data_config=data_config,
        split="test",
        models=model_group,
        args=args,
        for_selection=False,
    )
    test_generations_for_selection = load_predictions(
        data_config=data_config,
        split="test",
        models=model_group,
        args=args,
    )

    if args.use_full_text_embeddings:
        # use full text embeddings as input to Smoothie.
        # convert test_generations_for_smoothie to have test_input prepended.
        smoothie_text = []
        assert len(test_inputs) == len(test_generations_for_smoothie)
        for i, gens_per_sample in enumerate(test_generations_for_smoothie):
            smoothie_text.append(
                test_inputs[i] + " " + gen_per_model_per_sample
                for gen_per_model_per_sample in gens_per_sample
            )
        smoothie_text = np.array(smoothie_text)
        assert smoothie_text.shape == test_generations_for_smoothie.shape
    else:
        smoothie_text = test_generations_for_smoothie

    clean = data_config["dataset"] not in ["mix_instruct", "alpaca", "gsm8k"]
    smoothie_embeddings = embedder.embed_individual_generations(
        individual_generations=smoothie_text,
        clean=clean,
    )

    n_samples = int(len(smoothie_embeddings) / args.n_generations)
    n_voters = smoothie_embeddings.shape[1]
    embed_dim = smoothie_embeddings.shape[2]

    if args.type == "sample_dependent":
        if args.n_generations == 1:
            # use KNN
            nbrs = NearestNeighbors(n_neighbors=args.k, algorithm="auto")
            nbrs.fit(
                test_input_embeddings
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
            # use n_generations per sample to do estimation
            smoothie_dataset_weights = []
            for sample_idx in range(n_samples):
                embs_per_sample = smoothie_embeddings[
                    sample_idx
                    * args.n_generations : (sample_idx + 1)
                    * args.n_generations
                ]
                smoothie = Smoothie(n_voters=n_voters, dim=embed_dim)
                smoothie.fit(embs_per_sample)
                smoothie_dataset_weights.append(smoothie.theta)
            smoothie_dataset_weights = np.array(smoothie_dataset_weights)
    else:
        # learn a single set of weights for all samples
        smoothie = Smoothie(n_voters=n_voters, dim=embed_dim)
        smoothie.fit(smoothie_embeddings)
        smoothie_dataset_weights = np.tile(smoothie.theta, (n_samples, 1))

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


def main(args):
    # Load args and data config
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
            if args.regime == "train_time":
                train_time_smoothie(
                    args=args,
                    data_config=data_config,
                    model_group_name=model_group,
                    model_group=model_groups[model_group],
                    embedder=embedder,
                )
            else:
                test_time_smoothie(
                    args=args,
                    data_config=data_config,
                    model_group_name=model_group,
                    model_group=model_groups[model_group],
                    embedder=embedder,
                )
    else:
        if args.regime == "train_time":
            train_time_smoothie(
                args=args,
                data_config=data_config,
                model_group_name="",
                model_group=[args.model],
                embedder=embedder,
            )
        else:
            test_time_smoothie(
                args=args,
                data_config=data_config,
                model_group_name="",
                model_group=[args.model],
                embedder=embedder,
            )


if __name__ == "__main__":
    console.log("#" * 30)
    args = parser.parse_args()
    console.log(args)
    main(args)
