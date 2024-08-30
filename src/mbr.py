"""
This script implements a MBR (bayes minimum risk) baseline. 
"""

import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)


import argparse
import json
import jsonlines
import numpy as np
from sentence_transformers import SentenceTransformer


from sklearn.metrics.pairwise import cosine_similarity
from src.console import console
from src.constants import *
from src.data_utils import construct_processed_dataset_paths
from src.utils import (
    check_args,
    construct_mbr_predictions_path,
    embed_individual_generations,
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
    "--model_group",
    help="The models to use for predictions if we are doing multi-model",
)
parser.add_argument(
    "--type",
    choices=["sample_dependent", "sample_independent"],
    required=True,
    help="The type of Smoothie to use. See file docstring for more information.",
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
)
parser.add_argument(
    "--multi_model",
    action="store_true",
)


def main(args):
    check_args(args)
    data_config = load_data_config(args)
    output_fpath = construct_mbr_predictions_path(data_config, args.model, args)
    predictions_dir = output_fpath.parent
    if output_fpath.exists() and not args.redo:
        console.log(f"Results file already exists at {output_fpath}. Skipping.")
        return

    _, test_dataset_path = construct_processed_dataset_paths(args)
    with jsonlines.open(test_dataset_path) as file:
        test_dataset = list(file.iter())

    test_generations = load_predictions(predictions_dir, "test", args)

    model_name = "all-mpnet-base-v2"
    console.log(f"Loaded embedding model: {model_name}")

    smoothie_embeddings = embed_individual_generations(
        individual_generations=test_generations, model_name=model_name
    )

    n_samples = int(len(smoothie_embeddings) / args.n_generations)
    n_voters = smoothie_embeddings.shape[1]

    dataset_texts = []
    if args.type == "sample_dependent":
        for sample_idx in range(n_samples):
            embs_per_sample = smoothie_embeddings[sample_idx].reshape(n_voters, -1)
            sim = cosine_similarity(embs_per_sample)
            sim_vec = sim.sum(axis=1)

            max_idx = sim_vec.argmax()
            text = test_generations[sample_idx][max_idx]
            dataset_texts.append(text)
    else:
        sim_total = np.zeros((n_voters, n_voters))
        for sample_idx in range(n_samples):
            embs_per_sample = smoothie_embeddings[sample_idx]

            sim = cosine_similarity(embs_per_sample)
            sim_total += sim

        sim_vec = sim_total.sum(axis=1)
        max_idx = sim_vec.argmax()

        dataset_texts = test_generations[:, max_idx].tolist()

    results = {
        "generations": dataset_texts,
    }
    console.log(f"Saving results to {output_fpath}")
    output_fpath.write_text(json.dumps(results, default=int, indent=4))


if __name__ == "__main__":
    console.log("#" * 30)
    args = parser.parse_args()
    console.log(args)
    main(args)
