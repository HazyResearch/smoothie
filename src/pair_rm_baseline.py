"""
This script implements the PairRM baseline.
"""

import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)

import argparse
import json
import llm_blender
import jsonlines 

import numpy as np
from tqdm.auto import tqdm

from src.console import console
from src.utils import (
    check_args,
    construct_pair_rm_predictions_path,
    load_data_config,
    load_predictions,
)
from src.data_utils import construct_processed_dataset_paths
from src.ensembles import MODEL_GROUPS, MIX_INSTRUCT_GROUPS, GSM_8K_GROUPS

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

# Load PairRM model
blender = llm_blender.Blender()
blender.loadranker("llm-blender/PairRM") # load ranker checkpoint


def run_pair_rm_baseline(args, data_config, model_group_name, model_group):
    """
    Run the PairRM baseline baseline.

    Args:
        args (argparse.Namespace): arguments from the command line
        data_config (dict): data config
        model_group (str): name of the model group
    """
    output_fpath = construct_pair_rm_predictions_path(
        data_config=data_config,
        model_group_name=model_group_name,
        args=args,
    )
    if output_fpath.exists() and not args.redo:
        console.log(f"Results file already exists at {output_fpath}. Skipping.")
        return

    # Load generations
    test_generations = load_predictions(
        data_config=data_config,
        split="test",
        models=model_group,
        args=args,
    )

    # Load samples. We need this because the PairRM model takes as input teh prompt used.
    _, test_data_fpath = construct_processed_dataset_paths(args)
    with jsonlines.open(test_data_fpath) as file:
        dataset = list(file.iter())

    # Construct a list of all candidate inputs and generations
    all_candidate_inputs = []
    all_generations = []
    for sample_idx, sample in enumerate(dataset):
        # Load candidate inputs
        if args.multi_model:
            candidate_input = sample["multi_model_prompt"]
        else:
            mp_keys = sorted([k for k in sample.keys() if "multi_prompt" in k])
            candidate_input = sample[mp_keys[0]]  # Take the first prompt

        all_candidate_inputs.append(candidate_input)
        all_generations.append(test_generations[sample_idx])

    # Use PairRM to select the best generation for all samples at once
    all_ranks = blender.rank(
        all_candidate_inputs,
        all_generations,
        return_scores=False,
        batch_size=32
    )

    sequence_texts = []
    for sample_idx, ranks in enumerate(all_ranks):
        # Get all the idxs where the ranks[idx] = 1
        idxs = [i for i in range(len(ranks)) if ranks[i] == 1]
        # Select a random idx from the idxs
        best_generation = test_generations[sample_idx][np.random.choice(idxs)]
        sequence_texts.append(best_generation)

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

        if data_config["dataset"] == "mix_instruct":
            model_groups = MIX_INSTRUCT_GROUPS
        elif data_config["dataset"] == "gsm8k":
            model_groups = GSM_8K_GROUPS
        else:
            model_groups = MODEL_GROUPS

        for model_group in model_groups:
            run_pair_rm_baseline(
                args=args, 
                data_config=data_config, 
                model_group_name=model_group,
                model_group=model_groups[model_group]
            )

    else:
        run_pair_rm_baseline(
            args=args, 
            data_config=data_config, 
            model_group=""
        )


if __name__ == "__main__":
    console.log("#" * 30)
    args = parser.parse_args()
    console.log(args)
    main(args)
