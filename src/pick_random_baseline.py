"""
This script implements the pick-random baseline, which randomly selects one of the individual generations to return.
"""

import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)

import argparse
import json

import numpy as np
from tqdm.auto import tqdm

from src.console import console
from src.utils import (
    check_args,
    construct_pick_random_predictions_path,
    load_data_config,
    load_predictions,
)
from src.ensembles import MODEL_GROUPS

parser = argparse.ArgumentParser()

parser.add_argument("--model", default=None, type=str, help="LLM to use")
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
    help="Directory to save results to",
)
parser.add_argument(
    "--redo",
    action="store_true",
    help="Redo the generation if the results file already exists",
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

def run_pick_random_baseline(args, data_config, model_group):
    """
    Run the pick-random baseline.

    Args:
        args (argparse.Namespace): arguments from the command line
        data_config (dict): data config
        model_group (str): name of the model group
    """
    output_fpath = construct_pick_random_predictions_path(
        data_config=data_config,
        model_group=model_group,
        args=args,
    )
    if output_fpath.exists() and not args.redo:
        console.log(f"Results file already exists at {output_fpath}. Skipping.")
        return

    test_generations = load_predictions(
        data_config=data_config,
        split="test",
        model_group=model_group,
        args=args,
    )

    sequence_texts = []
    for _ in range(10):
        # we do pick-random ten times to reduce noise
        trial_generations = []
        for sample_idx in range(len(test_generations)):
            # Select a random generation from the individual generations.
            generation_idx = np.random.randint(test_generations.shape[1])
            generation = test_generations[sample_idx][generation_idx]
            trial_generations.append(generation)
        sequence_texts.append(trial_generations)

    # Save to file
    results = {"generations": sequence_texts}
    
    # Create directory for output path if it doesn't exist
    output_fpath.parent.mkdir(parents=True, exist_ok=True)
    output_fpath.write_text(json.dumps(results, indent=4))
    console.log(f"Results saved to {output_fpath}")


def main(args):
    check_args(args)
    np.random.seed(args.seed)
    data_config = load_data_config(args)
    if args.multi_model:
        for model_group in MODEL_GROUPS.keys():
            run_pick_random_baseline(
                args=args, 
                data_config=data_config, 
                model_group=model_group
            )

    else:
        run_pick_random_baseline(
            args=args, 
            data_config=data_config, 
            model_group=""
        )


if __name__ == "__main__":
    console.log("#" * 30)
    args = parser.parse_args()
    console.log(args)
    main(args)
