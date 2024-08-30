"""


"""

import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)


import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.console import console
from src.constants import *
from src.model import Smoothie

parser = argparse.ArgumentParser()
parser.add_argument(
    "--outputs_dir",
    default="alpaca/algorithm_outputs",
    type=str,
    help="Output directory",
)
parser.add_argument(
    "--n_trials",
    default=10,
    type=int,
    help="Number of trials to run. Each trial we sample k models.",
)
parser.add_argument(
    "--k", default=5, type=int, help="Number of models to sample for each trial"
)


def main(args):
    # Load outputs of individual models
    data_dir = Path("alpaca/downloaded_outputs")
    outputs = []
    for output_file in data_dir.glob("*.json"):
        with open(output_file, "r") as f:
            outputs.append(json.load(f))
            generator_name = outputs[-1][0]["generator"]
            console.log(f"Loaded {len(outputs[-1])} outputs for {generator_name}.")
    generator_names = np.array([output[0]["generator"] for output in outputs])

    # Get generations
    generations = []
    for i in range(len(outputs)):
        generations.append([output["output"] for output in outputs[i]])
    generations = np.array(generations).T
    console.log(f"Loaded generations of shape: {generations.shape}")

    # Get instructions
    instructions = [sample["instruction"] for sample in outputs[0]]
    console.log(f"Loaded instructions of shape: {len(instructions)}")
    console.log(f"Instructions: {instructions[:5]}")
    n_samples = len(instructions)

    # Set up model
    model = SentenceTransformer("all-mpnet-base-v2")

    # Compute embeddings of instructions
    instruction_embeddings = model.encode(instructions)
    instruction_embeddings = np.array(instruction_embeddings)

    # Compute embeddings of generations
    generations_embeddings = model.encode(generations.flatten())
    generations_embeddings = generations_embeddings.reshape(
        generations.shape[0], generations.shape[1], -1
    )
    console.log(
        f"Computed embeddings of generations of shape: {generations_embeddings.shape}"
    )

    # Fit KNN and find the k nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=21, algorithm="auto").fit(
        instruction_embeddings
    )
    _, instruction_nn_indices = nbrs.kneighbors(instruction_embeddings)
    instruction_nn_indices = instruction_nn_indices[:, 1:]

    # Run trials
    for trial_num in range(args.n_trials):
        # Sample generators to use in trial
        np.random.seed(trial_num)
        gen_idxs = np.random.choice(len(outputs), args.k, replace=False)
        console.log(f"Trial: {trial_num}. Sampled models: {generator_names[gen_idxs]}")

        # Construct trial generations and embeddings
        trial_generations = generations[:, gen_idxs]
        trial_generations_embeddings = generations_embeddings[:, gen_idxs, :]
        trial_models = generator_names[gen_idxs]
        assert trial_generations.shape == (n_samples, args.k)
        assert trial_generations_embeddings.shape == (n_samples, args.k, 768)

        # Create pick_random baseline by randomly selecting one of the k generations for each sample
        pick_random_outputs = []
        for sample_idx in range(n_samples):
            selected_generator_idx = np.random.choice(args.k)
            pick_random_outputs.append(
                {
                    "instruction": instructions[sample_idx],
                    "output": trial_generations[sample_idx, selected_generator_idx],
                    "generator": f"pick_random_{trial_num}",
                    "models_in_trial": str(trial_models.tolist()),
                    "selected_model": trial_models[selected_generator_idx],
                }
            )

        # Save pick_random outputs
        pick_random_output_file = (
            Path(args.outputs_dir) / f"pick_random_{trial_num}.json"
        )
        pick_random_output_file.parent.mkdir(parents=True, exist_ok=True)
        pick_random_output_file.write_text(json.dumps(pick_random_outputs, indent=4))

        # Compute Smoothie Dependent weights
        smoothie_outputs = []
        for sample_idx in range(len(instructions)):
            # Idxs of nearest neighbors (based on instruction)
            sample_nn_idxs = instruction_nn_indices[sample_idx]

            # Embeddings of generations in trial from corresponding nearest neighbors
            generation_embeds = trial_generations_embeddings[sample_nn_idxs]
            assert generation_embeds.shape == (20, args.k, 768)

            smoothie = Smoothie(
                n_voters=generation_embeds.shape[1], dim=generation_embeds.shape[2]
            )
            smoothie.fit(generation_embeds)
            best_gen_idx = smoothie.theta.argmax()
            smoothie_outputs.append(
                {
                    "instruction": instructions[sample_idx],
                    "output": trial_generations[sample_idx, best_gen_idx],
                    "generator": f"smoothie_{trial_num}",
                    "models_in_trial": str(trial_models.tolist()),
                    "selected_model": trial_models[best_gen_idx],
                    "smoothie_weights": str(smoothie.theta.tolist()),
                }
            )

        # Save Smoothie to file
        smoothie_output_file = Path(args.outputs_dir) / f"smoothie_{trial_num}.json"
        smoothie_output_file.parent.mkdir(parents=True, exist_ok=True)
        smoothie_output_file.write_text(json.dumps(smoothie_outputs, indent=4))

        # Compute Smoothie independent weights
        smoothie = Smoothie(
            n_voters=trial_generations_embeddings.shape[1],
            dim=trial_generations_embeddings.shape[2],
        )
        smoothie.fit(trial_generations_embeddings)
        best_gen_idx = smoothie.theta.argmax()
        smoothie_outputs = []
        for sample_idx in range(len(instructions)):
            smoothie_outputs.append(
                {
                    "instruction": instructions[sample_idx],
                    "output": trial_generations[sample_idx, best_gen_idx],
                    "generator": f"smoothie_independent_{trial_num}",
                    "models_in_trial": str(trial_models.tolist()),
                    "selected_model": trial_models[best_gen_idx],
                    "smoothie_weights": str(smoothie.theta.tolist()),
                }
            )

        # Save Smoothie to file
        smoothie_output_file = (
            Path(args.outputs_dir) / f"smoothie_independent_{trial_num}.json"
        )
        smoothie_output_file.parent.mkdir(parents=True, exist_ok=True)
        smoothie_output_file.write_text(json.dumps(smoothie_outputs, indent=4))


if __name__ == "__main__":
    console.log("#" * 30)
    args = parser.parse_args()
    console.log(args)
    main(args)
