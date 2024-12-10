"""
This script produces generations for the train and test splits of a dataset for different models. See args for information on parameters.

Sample command:
python -m src.get_generations --model "pythia-70m" \
    --dataset_config dataset_configs/squad.yaml \
    --data_dir "smoothie_data/datasets" \
    --device "cpu" \
    --hf_cache_dir "cache" \
    --results_dir test_results_multimodel \
    --redo \
    --test \
    --n_generations 1 \
    --temperature 0.0 \
    --seed 42 \
    --multi_model

python -m src.get_generations --model "pythia-70m" \
    --dataset_config dataset_configs/squad.yaml \
    --data_dir "smoothie_data/datasets" \
    --device "cpu" \
    --hf_cache_dir "cache" \
    --results_dir test_results_multiprompt \
    --redo \
    --test \
    --n_generations 1 \
    --temperature 0.0 \
    --seed 42 \
    --multi_prompt   
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import jsonlines
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from src.console import console
from src.constants import HF_MODEL_MAX_LENGTHS, HF_MODELS
from src.data_utils import construct_processed_dataset_paths
from src.utils import check_args, construct_generations_path, load_data_config

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="LLM to use")
parser.add_argument("--device", default="cuda", type=str, help="Device to use")
parser.add_argument(
    "--dataset_config",
    type=str,
    help="Path to dataset config file. This should be a yaml file.",
)
parser.add_argument(
    "--data_dir",
    default="smoothie_data/datasets",
    type=str,
    help="Directory with data files",
)
parser.add_argument(
    "--hf_cache_dir",
    default="cache",
    type=str,
    help="Directory to cache HF datasets to",
)
parser.add_argument(
    "--results_dir",
    type=str,
    help="Directory to save results to",
)
parser.add_argument(
    "--redo", action="store_true", help="If set, overwrite existing predictions."
)
parser.add_argument(
    "--test",
    action="store_true",
    help="Runs the script in test mode. This will only generate predictions for two samples.",
)
parser.add_argument(
    "--n_generations",
    default=1,
    type=int,
    help="For each model we produce n_generations per sample. Default is 1.",
)
parser.add_argument(
    "--temperature",
    default=0.0,
    type=float,
    help="Temperature for generations. Only used when n_generations > 1.",
)
parser.add_argument(
    "--seed",
    default=42,
    type=int,
    help="Random seed if n_generations > 1.",
)
parser.add_argument(
    "--multi_prompt",
    action="store_true",
)
parser.add_argument(
    "--multi_model",
    action="store_true",
)


def generate_predictions(
    args: argparse.Namespace,
    data_config: Dict,
    data_path: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    output_fpath: Path,
):
    """
    Generate predictions over a dataset using a model.

    This function generates predictions for each sample in the dataset using the specified model and tokenizer.
    It saves a json file with the following format:
    ```
    {
        "generations": [list of generated sequences]
    }
    ```

    Args:
        args (argparse.Namespace): Arguments from the command line, including configurations for generation.
        data_config (dict): Data configuration dictionary containing dataset-specific settings.
        data_path (str): Path to the data file containing the dataset to generate predictions for.
        model (transformers.PreTrainedModel): The pre-trained language model (LLM) used for generating predictions.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer associated with the pre-trained model.
        output_fpath (Path): The file path where the generated predictions will be saved.

    Returns:
        None
    """
    # Check if the results file already exists
    if output_fpath.exists() and not args.redo:
        console.log(f"Results file {output_fpath} already exists. Skipping.")
        return
    else:
        console.log(f"Will save results to: {output_fpath}")

    # load the data
    with jsonlines.open(data_path) as file:
        dataset = list(file.iter())

    if args.n_generations > 1:
        gen_params = {"temperature": args.temperature, "do_sample": True}
    else:
        gen_params = {
            "do_sample": False,
        }

    sequence_texts = []
    progress_bar = tqdm(range(len(dataset)))
    for sample_idx, sample in enumerate(dataset):
        if args.multi_model:
            prompt = sample["multi_model_prompt"]
            texts = generate_per_sample_single_prompt(
                data_config=data_config,
                args=args,
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                gen_params=gen_params,
            )
            sequence_texts.extend(texts)  # one big list

        else:
            mp_keys = sorted([k for k in sample.keys() if "multi_prompt" in k])
            prompts = [sample[k] for k in mp_keys]
            texts = generate_per_sample_multi_prompt(
                data_config=data_config,
                args=args,
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                gen_params=gen_params,
            )
            sequence_texts.append(texts)  # ensures that we have a list of lists

        progress_bar.update(1)
        if args.test and sample_idx == 1:
            break

    results = {
        "generations": sequence_texts,
    }
    output_fpath.write_text(json.dumps(results, indent=4))


def generate_per_sample_single_prompt(
    data_config: Dict,
    args: argparse.Namespace,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    gen_params: Dict,
) -> List[str]:
    """
    Returns a generation for a single sample with a single prompt. If args.n_generations > 1, returns a list.
    """

    sequence_texts = []
    prompt_encodings = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=HF_MODEL_MAX_LENGTHS[args.model],
    ).to(args.device)

    for i in range(args.n_generations):
        output = model.generate(
            **prompt_encodings,
            max_new_tokens=data_config["max_new_tokens"],
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
            output_scores=True,
            **gen_params,
        )

        # Get the token ids corresponding to the generation. output["sequences"] is a tensor of shape (batch_size, seq_len)
        sequence_texts.append(
            tokenizer.decode(get_generation_output(prompt_encodings, output))
        )

    # returns a list of args.n_generations outputs

    return sequence_texts


def generate_per_sample_multi_prompt(
    data_config: Dict,
    args: argparse.Namespace,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    gen_params: Dict,
) -> List[str]:
    """
    Returns a list of generations corresponding to multiple prompts for a single sample.

    Args:
        data_config (Dict): the data config
        args (argparse.Namespace): the arguments from the command line
        model (AutoModelForCausalLM): the model
        tokenizer (AutoTokenizer): the tokenizer
        prompts (List[str]): the prompts
        gen_params (Dict): the generation parameters

    Returns:
        List[str]: the generations. The length of the list is n_prompts * n_generations.
    """
    sequence_texts = (
        []
    )  # will be a list: p1 output1, p1 output2, ..., p2 output1, p2 output2, ...
    for prompt_idx in range(len(prompts)):
        texts = generate_per_sample_single_prompt(
            data_config=data_config,
            args=args,
            model=model,
            tokenizer=tokenizer,
            prompt=prompts[prompt_idx],
            gen_params=gen_params,
        )
        sequence_texts.extend(texts)

    return sequence_texts


def get_generation_output(input: Dict, output: Dict) -> List[str]:
    """
    By default, Huggingface returns the prompt + the generated text. This function
    returns only the generated text.

    Args:
        input (Dict): the input encodings
        output (Dict): the output encodings

    Returns:
        List[str]: the token ids of the generation
    """
    input_len = input["input_ids"].shape[1]
    return output["sequences"][0, input_len:].detach().to("cpu").tolist()


def load_hf_model(
    args: argparse.Namespace,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a HuggingFace model and tokenizer.

    Args:
        args (argparse.Namespace): arguments from the command line

    Returns:
        Tuple[AutoModelForCausalLM, AutoTokenizer]: the model and tokenizer
    """
    model = AutoModelForCausalLM.from_pretrained(
        HF_MODELS[args.model],
        cache_dir=args.hf_cache_dir,
        trust_remote_code=True,
    )
    model.to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        HF_MODELS[args.model],
        cache_dir=args.hf_cache_dir,
        truncation_side="left",
        trust_remote_code=True,
    )
    return model, tokenizer


def main(args: argparse.Namespace):
    """
    Main function.
    """
    # Load the data config
    data_config = load_data_config(args)

    # Construct the paths to the output files
    train_output_fpath, test_output_fpath = construct_generations_path(
        data_config=data_config, model=args.model, args=args
    )
    if train_output_fpath.exists() and test_output_fpath.exists() and not args.redo:
        console.log(
            f"Results file {train_output_fpath} and {test_output_fpath} already exists. Skipping."
        )
        return
    else:
        console.log(
            f"Will save results to: {train_output_fpath} and {test_output_fpath}"
        )

    # Get paths to the processed dataset files
    train_data_fpath, test_data_fpath = construct_processed_dataset_paths(args)

    # Load the model and tokenizer
    model, tokenizer = load_hf_model(args=args)

    # Set the seed if we are doing multiple generations
    if args.n_generations > 1:
        set_seed(args.seed)
        assert args.temperature != 0

    generate_predictions(
        args=args,
        data_config=data_config,
        data_path=train_data_fpath,
        model=model,
        tokenizer=tokenizer,
        output_fpath=train_output_fpath,
    )
    generate_predictions(
        args=args,
        data_config=data_config,
        data_path=test_data_fpath,
        model=model,
        tokenizer=tokenizer,
        output_fpath=test_output_fpath,
    )


if __name__ == "__main__":
    console.log("#" * 30)
    args = parser.parse_args()
    console.log(args)
    main(args)
