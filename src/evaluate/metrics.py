"""
This script implements evaluation metrics for different tasks.
"""

import time
from collections import Counter
from typing import List, Union

import numpy as np
from tqdm.auto import tqdm

from evaluate import load


def compute_rouge2_score(
    generations: List[str], references: Union[List[List[str]], List[str]]
) -> List[float]:
    """
    Computes ROUGE-2 score over a list of generations and references.

    Args:
        generations (List[str]): List of generated texts
        references (Union[List[List[str]], List[str]]): List of reference texts

    Returns:
        List[float]: List of ROUGE-2 scores
    """
    rouge = load("rouge")
    results = rouge.compute(
        predictions=generations, references=references, use_aggregator=False
    )
    return results["rouge2"]


def squad_acc(generations: List[str], references: List[str]) -> List[int]:
    """
    Computes accuracy for SQuAD dataset. A generation is correct if it contains the reference answer.

    Args:
        generations (List[str]): List of generated texts
        references (List[str]): List of reference texts

    Returns:
        List[int]: List of 0s and 1s, where 1 indicates a correct generation and 0 indicates an incorrect generation
    """
    correct = []
    for gen, ref in zip(generations, references):
        if ref.lower() in gen.lower():
            correct.append(1)
        else:
            correct.append(0)
    return correct


def trivia_qa_acc(generations: List[str], references: List[List[str]]) -> List[int]:
    """
    Computes accuracy for TriviaQA dataset. A generation is correct if it contains any of the reference answers.

    Args:
        generations (List[str]): List of generated texts
        references (List[List[str]]): List of lists of reference texts

    Returns:
        List[int]: List of 0s and 1s, where 1 indicates a correct generation and 0 indicates an incorrect generation
    """
    correct = []
    for gen, refs in zip(generations, references):
        gen_lower = gen.lower()
        if any(ref.lower() in gen_lower for ref in refs):
            correct.append(1)
        else:
            correct.append(0)
    return correct


def definition_extraction_acc(
    generations: List[str], references: List[str]
) -> List[int]:
    """
    Computes accuracy for definition extraction dataset. A generation is correct if it contains the reference answer.

    Args:
        generations (List[str]): List of generated texts
        references (List[str]): List of reference texts

    Returns:
        List[int]: List of 0s and 1s, where 1 indicates a correct generation and 0 indicates an incorrect generation
    """
    correct = []
    for gen, ref in zip(generations, references):
        gen_lower = gen.lower()
        if ref.lower() in gen_lower:
            correct.append(1)
        else:
            correct.append(0)
    return correct


def mix_instruct_rank(
    generations: List[str], references: List[Dict[str, int]]
) -> List[int]:
    """
    Returns the rank of each generation. For mix-instruct, `references` is a list of dictionaries (where the ith dictionary corresponds to the ith sample in the dataset), and each dictionary maps model generations to corresponding ranks. See src/setup_mix_instruct.py for more details.

    Ranks are computed using ChatGPT. See the original paper for more details: https://arxiv.org/abs/2306.02561.

    Args:
        generations (List[str]): List of generated texts
        references (List[Dict[str, int]]): List of dictionaries mapping model generations to ranks

    Returns:
        List[int]: List of ranks
    """
    ranks = []
    for gen, ref_dict in zip(generations, references):
        ranks.append(ref_dict[gen])
    return ranks


def gsm8k_acc(generations: List[str], references: List[str]) -> List[int]:
    """
    Computes accuracy for GSM8K dataset. To isolate the answer from the generation, we split the generation into sentences and check if any of the sentences contain the substring "the answer is". If so, we check if the reference answer is in this sentence.

    Args:
        generations (List[str]): List of generated texts
        references (List[str]): List of reference texts

    Returns:
        List[int]: List of 0s and 1s, where 1 indicates a correct generation and 0 indicates an incorrect generation
    """
    correct = []
    for gen, ref in zip(generations, references):
        gen_sentences = gen.split(".")
        for sentence in gen_sentences:
            if "the answer is" in sentence:
                gen_lower = sentence.lower().replace(",", "")
                ref_lower = ref.lower().replace(",", "")
                if ref_lower in gen_lower:
                    correct.append(1)
                else:
                    correct.append(0)
                break
        else:
            correct.append(0)
    return correct


METRIC_FUNCS = {
    "rouge2": compute_rouge2_score,
    "squad_acc": squad_acc,
    "trivia_qa_acc": trivia_qa_acc,
    "definition_extraction_acc": definition_extraction_acc,
    "mix_instruct_rank": mix_instruct_rank,
    "gsm8k_acc": gsm8k_acc,
}

MULTI_MODEL_TASK2METRIC = {
    "cnn_dailymail": "rouge2",
    "definition_extraction": "definition_extraction_acc",
    "e2e_nlg": "rouge2",
    "squad": "squad_acc",
    "trivia_qa": "trivia_qa_acc",
    "web_nlg": "rouge2",
    "xsum": "rouge2",
}
