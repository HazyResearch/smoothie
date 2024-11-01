"""
This script generates predictions for the Alpaca dataset.

NOTE:
    - Alpaca has no train split, so we don't run train-time smoothie.
"""

import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)


import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm

from src.console import console
from src.constants import *
from src.model import Smoothie
from src.utils import Embedder

parser = argparse.ArgumentParser()
parser.add_argument(
    "--outputs_dir",
    default="alpaca/algorithm_outputs",
    type=str,
    help="Output directory",
)
parser.add_argument(
    "--n_models", default=5, type=int, help="Number of models to sample for each trial"
)
parser.add_argument(
    "--embedding_model",
    type=str,
    choices=["all-mpnet-base-v2", "bge-small-en-v1.5"],
    help="Model to use for embedding generations."
)
parser.add_argument("--k", help="Nearest neighbors size", type=int)

ENSEMBLES = [
    ['Meta-Llama-3-70B-Instruct', 'mistral-large-2402', 'yi-large-preview', 'Storm-7B', 'gemini-pro'],
    ['Meta-Llama-3-70B-Instruct', 'Storm-7B', 'FsfairX-Zephyr-Chat-v0.1', 'yi-large-preview', 'Nanbeige-Plus-Chat-v0.1'],
    ['yi-large-preview', 'gemini-pro', 'claude-2', 'Nanbeige-Plus-Chat-v0.1', 'Ein-70B-v0.1'],
    ['claude-2', 'yi-large-preview', 'gemini-pro', 'Meta-Llama-3-70B-Instruct', 'Storm-7B'],
    ['Qwen1.5-110B-Chat', 'mistral-large-2402', 'yi-large-preview', 'Storm-7B', 'Meta-Llama-3-70B-Instruct'],
    ['Storm-7B', 'claude-2', 'Meta-Llama-3-70B-Instruct', 'yi-large-preview', 'Ein-70B-v0.1'],
    ['mistral-large-2402', 'gemini-pro', 'Ein-70B-v0.1', 'Nanbeige-Plus-Chat-v0.1', 'FsfairX-Zephyr-Chat-v0.1'],
    ['mistral-large-2402', 'claude-2', 'Nanbeige-Plus-Chat-v0.1', 'Meta-Llama-3-70B-Instruct', 'gemini-pro'],
    ['mistral-large-2402', 'FsfairX-Zephyr-Chat-v0.1', 'Storm-7B', 'Nanbeige-Plus-Chat-v0.1', 'Meta-Llama-3-70B-Instruct'],
    ['mistral-large-2402', 'yi-large-preview', 'Ein-70B-v0.1', 'Meta-Llama-3-70B-Instruct', 'gemini-pro']
]

def main(args):

    # Load embedder
    embedder = Embedder(model_name=args.embedding_model)

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


    # Compute embeddings of instructions. embedder.embed_dataset assumes inputs are 
    # a list of dicts where the text to be embedded is under the key 'embedding_input'
    instruction_embeddings = embedder.embed_dataset([{"embedding_input": instruction} for instruction in instructions])

    # Compute embeddings of generations
    generations_embeddings = embedder.embed_individual_generations(
        individual_generations=generations,
        clean=False, # We don't need to clean Alpaca generations
    )
    console.log(
        f"Computed embeddings of generations of shape: {generations_embeddings.shape}"
    )

    # Fit KNN and find the k nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=args.k, algorithm="auto").fit(
        instruction_embeddings
    )
    _, instruction_nn_indices = nbrs.kneighbors(instruction_embeddings)

    # Run trials
    for trial_num in range(len(ENSEMBLES)):
        # Get the ensemble for this trial
        ensemble = ENSEMBLES[trial_num]
        
        # Find the indices of the generators in the ensemble
        gen_idxs = np.array([np.where(generator_names == model)[0][0] for model in ensemble])
        
        # Ensure we have the correct number of models
        args.n_models = len(ensemble)
        assert len(gen_idxs) == args.n_models, f"Expected {args.n_models} models, but got {len(gen_idxs)}"
        
        console.log(f"Trial {trial_num}: Using models {ensemble}")

        # Construct trial generations and embeddings
        trial_generations = generations[:, gen_idxs]
        trial_generations_embeddings = generations_embeddings[:, gen_idxs, :]
        trial_models = generator_names[gen_idxs]
        assert trial_generations.shape == (n_samples, args.n_models)
        #assert trial_generations_embeddings.shape == (n_samples, args.n_models, -1), trial_generations_embeddings.shape


        # Create pick_random baseline by randomly selecting one of the k generations for each sample
        pick_random_outputs = []
        for sample_idx in range(n_samples):
            selected_generator_idx = np.random.choice(len(ensemble))
            pick_random_outputs.append(
                {
                    "instruction": instructions[sample_idx],
                    "output": trial_generations[sample_idx, selected_generator_idx],
                    "generator": f"pick_random_trial_num={trial_num}",
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
            #assert generation_embeds.shape == (args.k, args.n_models, -1), generation_embeds.shape

            smoothie = Smoothie(
                n_voters=generation_embeds.shape[1], dim=generation_embeds.shape[2]
            )
            smoothie.fit(generation_embeds)
            best_gen_idx = smoothie.theta.argmax()
            smoothie_outputs.append(
                {
                    "instruction": instructions[sample_idx],
                    "output": trial_generations[sample_idx, best_gen_idx],
                    "generator": f"smoothie_{args.embedding_model}_n_neighbors={args.k}_trial_num={trial_num}",
                    "models_in_trial": str(trial_models.tolist()),
                    "selected_model": trial_models[best_gen_idx],
                    "smoothie_weights": str(smoothie.theta.tolist()),
                }
            )

        # Save Smoothie to file
        smoothie_output_file = Path(args.outputs_dir) / f"smoothie_{args.embedding_model}_{trial_num}_n_neighbors={args.k}_trial_num={trial_num}.json"
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
                    "generator": f"smoothie_independent_{args.embedding_model}_trial_num={trial_num}",
                    "models_in_trial": str(trial_models.tolist()),
                    "selected_model": trial_models[best_gen_idx],
                    "smoothie_weights": str(smoothie.theta.tolist()),
                }
            )

        # Save Smoothie to file
        smoothie_output_file = (
            Path(args.outputs_dir) / f"smoothie_independent_{args.embedding_model}_{trial_num}_n_neighbors={args.k}_trial_num={trial_num}.json"
        )
        smoothie_output_file.parent.mkdir(parents=True, exist_ok=True)
        smoothie_output_file.write_text(json.dumps(smoothie_outputs, indent=4))


if __name__ == "__main__":
    console.log("#" * 30)
    args = parser.parse_args()
    console.log(args)
    main(args)
