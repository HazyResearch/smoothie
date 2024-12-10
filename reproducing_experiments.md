# Reproducing paper experiments

We assume the Huggingface dataset repository has been cloned to `$DATA_DIR`.

## AlpacaEval experiments

The Huggingface dataset repository contains the generations from 10 different models on the AlpacaEval dataset. These were downloaded off of the [leaderboard](https://tatsu-lab.github.io/alpaca_eval/). They can be found [here](https://huggingface.co/datasets/hazyresearch/smoothie_data/tree/main/alpaca/downloaded_outputs).

To produce Smoothie and baseline predictions, run the following command:

```bash
> python3 -m src.alpaca_generate_predictions --embedding_model all-mpnet-base-v2 --k 1
```

This will save predictions for Smoothie and the baseline methods in `$DATA_DIR/alpaca/algorithm_outputs`. To run the evaluation, navigate to the `$DATA_DIR/alpaca` directory and run the following command:

```bash
> ./run_alpaca.sh
```

This will write the results to `$DATA_DIR/alpaca/leaderboard.csv`.

## Multi-model experiments

Generations, method outputs, and scores for the multi-model experiments can be found [here](https://huggingface.co/datasets/hazyresearch/smoothie_data/tree/main/multi_model_results). Every subdirectory contains results for a different dataset. Within each dataset directory is a subdirectory called `model_gens` containing train and test generations for each model. The generations are saved in json files with the following format: `{model_name}_train.json` and `{model_name}_test.json`. Also within each dataset directory are subdirectories corresponding to different ensembles. The "name" of the ensemble is the name of the subdirectory. A mapping of ensemble names to the models they contain can be found in `src/ensembles.py`. The paper reports results on three ensembles. It reports primary results on an ensemble of 3B and 7B models (respectively denoted as `3b_ensemble` and `7b_ensemble`). It also reports secondary results on randomly constructed ensembles of varying sizes (denoted as `ensemble_{i}`). 

Within each ensemble directory are the predictions for Smoothie variants and baseline methods and the scores for each method. The scores are saved in a json file called `scores.json`. The predictions are saved in json files with the following format: 

| Method | Prediction File Format |
|--------|------------------------|
| pick-random | `pick_random_test.json` |
| PairRM | `pair_rm_test.json` |
| Best-on-Val | `labeled_oracle_test.json` |
| Labeled-KNN | `labeled_knn_test.json` |
| Smoothie-Local (Test-time) with k neighbors | `smoothie_sample_dependent_test_time_{k}_{embedding_model}_test.json` |
| Smoothie-Global (Test-time) | `smoothie_sample_independent_test_time_{embedding_model}_test.json` |
| Smoothie-Local (Train-time) with k neighbors | `smoothie_sample_dependent_train_time_{k}_{embedding_model}_train.json` |
| Smoothie-Global (Train-time) | `smoothie_sample_independent_train_time_{embedding_model}_train.json` |

To reproduce the experiments on the NLG datasets, run the following command:

```bash
> ./replication_scripts/multimodel_nlg_exps.sh
```
This script will preprocess datasets, generate predictions from models, and evaluate the predictions. Please see the script for more details regarding file path placeholders.

To reproduce the multi-model experiments on GSM8K, run the following command:

```bash
> ./replication_scripts/gsm8k.sh
```

To reproduce the multi-model experiments on MIXInstruct, run the following command:

```bash
> ./replication_scripts/mix_instruct_exps.sh
```

## Reproducing Multi-prompt experiments

Generations, method outputs, and scores for the multi-model experiments can be found [here](https://huggingface.co/datasets/hazyresearch/smoothie_data/tree/main/multi_prompt_results). Every subdirectory contains results for a different dataset, and within each dataset directory are subdirectories corresponding to different models. Generations from baseline methods follow the same format as the multi-model experiments. Generations from the different prompts are saved to `individual_train.json` and `individual_test.json` within each model directory.

To reproduce the multi-prompt experiments, run the following command:

```bash
> ./replication_scripts/multiprompt_exps.sh
```
