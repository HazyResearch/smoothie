"""
This script generates random ensembles of language models for experimentation.

It defines a list of predefined language models and creates 40 random ensembles,
each containing 3 to 7 models. The script ensures that each ensemble is unique
and prints the resulting ensembles in a dictionary format.

Key components:
1. A list of predefined language models (MODELS)
2. A function to generate random, unique ensembles (generate_ensembles)
3. Code to create and print 40 random ensembles with a fixed random seed for reproducibility

The output is formatted as a Python dictionary named ENSEMBLES, where each key
is an ensemble name (e.g., 'ensemble_1') and the corresponding value is a list
of model names in that ensemble.
"""

MODELS = [
    "mistral-7b",
    "llama-2-7b",
    "vicuna-7b",
    "gemma-7b",
    "nous-capybara",
    "pythia-2.8b",
    "gemma-2b",
    "incite-3b",
    "dolly-3b",
]

import random

# Set a fixed seed for reproducibility
random.seed(42)


def generate_ensembles(models):
    ensembles = {}
    ensemble_count = 1

    for size in range(3, 8):
        generated = 0
        while generated < 10:
            ensemble = tuple(sorted(random.sample(models, size)))
            if ensemble not in ensembles.values():
                ensembles[f"ensemble_{ensemble_count}"] = list(ensemble)
                ensemble_count += 1
                generated += 1

    return ensembles


# Generate ensemble
random_ensembles = generate_ensembles(MODELS)

# Print the ensembles
print("ENSEMBLES = {")
for key, value in random_ensembles.items():
    print(f"\t'{key}': {value},")
print("}")
