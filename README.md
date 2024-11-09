# smoothie

This repository contains replication code for the following paper:

> Smoothie: Label Free Language Model Routing
> Neel Guha*, Mayee Chen*, Trevor Chow, Ishan Khare, Christopher Ré
> NeurIPS 2024

It contains code for replicating all experiments in the paper. If you would like to use Smoothie in any pipeline, we recommend using version.

This is the codebase for Smoothie. It allows you to both use Smoothie, and reproduce the experiments in the paper. 



where `$HUGGINGFACE_TOKEN` is your HuggingFace token.

## Dependencies

Install the dependencies using the following commands:

```
> conda create -n "smoothie" python=3.10 -y
> conda activate smoothie
> pip install -r requirements.txt
```

## Data and model generations

We store all datasets, predictions, and results from the paper in a [HuggingFace dataset repository](https://huggingface.co/datasets/hazyresearch/smoothie_data/). You can download the dataset from HuggingFace by running the following command:

```bash
> huggingface-cli login --token $HUGGINGFACE_TOKEN --add-to-git-credential
> git clone https://huggingface.co/datasets/hazyresearch/smoothie_data
```

## Reproducing the paper

The repository contains the following folders:

- `alpaca`: Contains the code and data for the AlpacaEval experiments.
- `dataset_configs`: Contains the configuration files for all single-task and multi-task datasets.
- `plots`: Contains plots for the paper.
- `prompt_templates`: Contains the prompt templates for all single-task and multi-task datasets.
- `replication_scripts`: Contains bash scripts for running all experiments in the paper.
- `src`: Contains the source code for formatting datasets, getting generations, running routing methods, and evaluating results.
- `tables`: Contains tables for the paper.
- `tutorials`: Contains tutorials for using Smoothie.

### Formatting datasets

#### Single-task datasets
`dataset_configs` contains the configuration files for all single-task datasets. Each configuration file contains the following fields:

```yaml 
# Name of the dataset
dataset: squad
# Name of the prompt template to use for the multi-prompt setting. There should be a file called squad_multi_prompt.json in prompt_templates. It should contain a list of f-strings.
multi_prompt_template: squad_multi_prompt
# Name of the prompt template to use for the multi-model setting. There should be a file called squad_multi_model.txt in prompt_templates. It should contain a single f-string.
multi_model_prompt_template: squad_multi_model
# Maximum number of new tokens to generate
max_new_tokens: 20
# Train and test sizes
train_size: 250
test_size: 1000
# Metrics to compute for evaluation
metrics:
  - squad_acc
```

The script `src/make_dataset.py` downloads the datasets from Huggingface, creates the train and test splits, generates prompts for each row, and saves the resulting splits to disk as jsonl files. To create the train and test splits for a dataset, run the following command:

```bash
> python -m src.make_dataset --dataset_config $config_filename
```

Train and test splits for datasets are saved to `$HF_DATASETS_DIR/datasets/` as `${config_filename}_train.jsonl` and `${config_filename}_test.jsonl`. Each line corresponds to a sample, and contains the following fields:
- `idx`: Unique identifier for the sample
- `task_name`: The name of the task
- `reference`: The gold reference text for the sample.
- `embedding_input`: Text input used when computing lookup embeddings.
- `multi_model_prompt`: The prompt used for the multi-model setting.
- `multi_prompt_{i}`: The prompt used for the i-th prompt in the multi-prompt setting.

### Multi-task datasets

For multi-task datasets, each configuration file instead only contains the following fields:

```yaml
# This file contains the configuration for the accuracy group of tasks
tasks:
  - dataset_configs/squad.yaml # Configuration file for the first task
  - dataset_configs/trivia_qa.yaml # Configuration file for the second task
  - dataset_configs/definition_extraction.yaml # Configuration file for the third task
# Maximum number of new tokens to generate
max_new_tokens: 40
```

We use the script `src/make_multi_task_dataset.py` to create the train and test splits for multi-task datasets as well:

```bash
> python -m src.make_dataset --dataset_config dataset_configs/acc_group.yaml
```

Train and test splits for datasets are also saved to `$HF_DATASETS_DIR/datasets/` as `${config_filename}_train.csv` and `${config_filename}_test.csv`. Each csv file contains the following columns:
- `idx`: Unique identifier for the sample
- `task_name`: The name of the task
- `reference`: The gold reference text for the sample.
- `embedding_input`: Text input used when computing lookup embeddings.
- `multi_model_prompt`: The prompt used for the multi-model setting.

Note that compared to single-task datasets, multi-task datasets do not contain the `multi_prompt_{i}` columns.

### Producing generations 

Once a dataset has been created, we need to produce the generations for this dataset. Given a list of models (multi-model) or a list of prompts (multi-prompt), we create individual json files containing the generations for each model/prompt. Note that all valid models are defined in `src/constants.py`.

#### Multi-model

In the multi-model setting, `src/get_generations.py` is called with the `--multi_model` flag, and a group of models is specified with `--model_group`. Currently, we have a `3b` and `7b` model group defined in `MODEL_GROUPS` in `src/utils.py`. 

```bash
> python -m src.get_generations \
    --dataset_config $config_filename \
    --model_group $model_group \
    --results_dir $RESULTS_DIR \
    --multi_model
```

For each `model_name` in the `model_group`, train and test generations are saved to `$RESULTS_DIR/${config_filename}/${model_group}/${model_name}_train.json` and `$RESULTS_DIR/${config_filename}/${model_group}/${model_name}_test.json`, respectively.

#### Multi-prompt

In the multi-prompt setting, `src/get_generations.py` is called with the `--multi_prompt` flag, and the model to use is specified with `--model`, which takes as argument any of the models in `HF_MODELS` in `src/constants.py`.

```bash
> python -m src.get_generations \
    --dataset_config $config_filename \
    --model $model \
    --results_dir $RESULTS_DIR \
    --multi_prompt
```

Train and test generations are saved to `$RESULTS_DIR/${config_filename}/${model}/individual_train.json` and `$RESULTS_DIR/${config_filename}/${model}/individual_test.json`, respectively. Since we store the generations for multiple prompts in one file, the generations are stored as a list of lists (n_samples x n_prompts).

### Routing methods

Once individual model/prompt generations are created, we have several methods that we can use to select a generation per sample:
* Smoothie (`src/run_smoothie.py`): our main algorithm. `--type` accepts as argument either `sample_dependent` or `sample_independent`, which controls if we use the dependent (different model/prompt per sample) or independent (one model/prompt for the entire dataset) version of Smoothie. In Smoothie-dependent, `--k` is the parameter for the k-nearest neighbors per test sample used as input to Smoothie to determine what the sample should be routed to.
  ```bash
  > python -m src.run_smoothie \
    --dataset_config $config_filename \ 
    --model_group $model_group \
    --results_dir $RESULTS_DIR \
    --multi_model \
    --type sample_dependent \
    --k $k \ 
  ```

  Outputs are saved to `$RESULTS_DIR/${config_filename}/${model_group}/smoothie_${type}_${model_group}_${k}_test.json`, which contains both the selected generation per sample as well as the Smoothie weights on each generation per sample.


* Pick Random (`src/pick_random_baseline.py`): this approach selects a random generation for each sample. The procedure is repeated 10 times to reduce noise. 
  ```bash
  > python -m src.pick_random_baseline \
    --dataset_config $config_filename \ 
    --model_group $model_group \
    --results_dir $RESULTS_DIR \
    --multi_model \
  ```

  Outputs are saved to `$RESULTS_DIR/${config_filename}/${model_group}/pick_random_${model_group}_test.json`, which contains a list of (n_trials, n_samples) selected generations.
  
* Labeled Oracle (`src/labeled_oracle.py`): this approach uses a subset of the train split of generations to determine which model/prompt is the best. The procedure is repeated 10 times to reduce noise.
  ```bash
  > python -m src.labeled_oracle \
    --dataset_config $config_filename \ 
    --model_group $model_group \
    --results_dir $RESULTS_DIR \
    --multi_model \
  ```

  Outputs are saved to `$RESULTS_DIR/${config_filename}/${model_group}/labeled_oracle_${model_group}_test.json`, which contains a list of (n_trials, n_samples) selected generations.

* Labeled KNN (`src/labeled_knn.py`): for each test sample, this approach examines the k nearest neighbors in a subset of the training dataset and routes the test sample to the model/prompt that does best on a majority of the neighbors. The procedure is repeated 10 times to reduce noise, and k is set to 20. 
  ```bash
  > python -m src.labeled_knn \
    --dataset_config $config_filename \ 
    --model_group $model_group \
    --results_dir $RESULTS_DIR \
    --multi_model \
  ```

  Outputs are saved to `$RESULTS_DIR/${config_filename}/${model_group}/labeled_knn_${model_group}_test.json`, which contains a list of (n_trials, n_samples) selected generations. Labeled KNN is not supported for the multi-prompt version yet.


For the multi-prompt versions, replace `--model_group $model_group` with `--model $model`, `--multi_model` with `--multi_prompt`, and set `$RESULTS_DIR` accordingly. Outputs are saved to `$RESULTS_DIR/${config_filename}/${model}/${method_name}_test.json`. 

### Evaluation

We use `src/evaluate/evaluate.py` to compare each method's selected generations to the reference generations. Depending on the task, we use a different evaluation metric (see `MULTI_MODEL_TASK2METRIC` in `src/evaluate/metrics.py`). For the pick random, labeled oracle, and labeled KNN methods, we average the performance over the trials.  

To evaluate in the multi-model setting, run:
```bash
python -m src.evaluate.evaluate \
    --dataset_config $dataset_config \
    --model_group $model_group \
    --multi_model \
    --results_dir $RESULTS_DIR
```

To evaluate in the multi-prompt setting, run:
```bash
python -m src.evaluate.evaluate \
    --dataset_config $dataset_config \
    --model $model \
    --multi_prompt \
    --results_dir $RESULTS_DIR
```

This will output a file named `scores.json` in the folder with all the generations and routing methods' outputs.

### Alpaca results

We run experiments showing that Smoothie can be used to select the best instruction-following model from a set of models. We use the Alpaca dataset for this purpose. We download predictions of 10 different models from the Alpaca leaderboard. JSON files containing the predictions are stored in `alpaca/downloaded_outputs/`. Each JSON file is a list, where each entry is a dictionary containing the following fields:

```json
{
    "dataset":"helpful_base",
    "instruction":"What are the names of some famous actors that started their careers on Broadway?",
    "output":"Here are some famous actors who got their start on Broadway:\n\n- Hugh Jackman - He made his Broadway debut as Gaston in Beauty and the Beast in 1994. He went on to star in shows like The Boy From Oz and The Music Man.\n\n- Neil Patrick Harris - He starred in musicals like Rent and Cabaret early in his career in the late 1990s before becoming known for TV shows like Doogie Howser M.D. and How I Met Your Mother.\n\n- Audra McDonald - She has won a record 6 Tony Awards for shows like Carousel, Master Class, and Lady Day at Emerson's Bar & Grill. She's also been on TV shows like Private Practice. \n\n- Kristin Chenoweth - She made her Broadway debut in 1997's Steel Pier and is known for roles in Wicked and You're a Good Man, Charlie Brown. She later starred in TV shows like Pushing Daisies and Glee.\n\n- Jesse Tyler Ferguson - Before his role on Modern Family, he performed in Broadway shows like On the Town and The 25th Annual Putnam County Spelling Bee in the mid-2000s.\n\n- Cynthia Erivo - She broke out as Celie in the Broadway revival of The Color Purple. She went on to win a Tony and star in movies like Harriet.\n\n- Sutton Foster - She headlined shows like Thoroughly Modern Millie, Anything Goes, and Violet before becoming known for her TV work on Bunheads and Younger.",
    "generator":"claude-2"
  }
```

`alpaca/generate_predictions.py` generates predictions for Smoothie and baselines across multiple trials. 

```bash
> python -m src.generate_predictions --n_trials 10 --k 5
```
where `n_trials` is the number of trials to run, and `k` is the number of models to randomly select in each trial. For trial {i}, the following files are saved to `alpaca/algorithm_outputs/`:

- `pick_random_{i}.json`: The predictions of the random-selection baseline. This corresponds to randomly selecting a model from the ensemble for each sample.
- `smoothie_{i}.json`: The predictions of Smoothie using contextual weights.
- `smoothie_independent_{i}.json`: The predictions of Smoothie using sample independent weights.

Each JSON file is a list, where each entry is a dictionary. The fields across the three files are identical. Here is an example from `smoothie_1.json`:

```json
{
        "instruction": "What are the names of some famous actors that started their careers on Broadway?",
        "output": "\nTransitioning from the realm of Broadway theater to the bright lights of Hollywood is a dream for many stage actors. This transition not only marks a change in performance venue but often signifies a leap towards greater fame and a broader audience reach. Several renowned actors began their careers on Broadway before capturing the hearts of movie and television audiences worldwide. Here are some notable names:\n\n1. **Robert Redford** - Known for his leading roles in films like \"Butch Cassidy and the Sundance Kid\" and \"The Sting,\" Redford started his career on Broadway, winning his first Tony Award for his performance in \"Out of Our League\" in 1962.\n\n2. **Meryl Streep** - Often regarded as the greatest actress of her generation, Streep's Broadway debut was in the 1971 production of \"The Play What I Wrote.\" She has since garnered three Tony Awards for her work on Broadway.\n\n3. **Al Pacino** - A method actor known for his intense performances, Pacino made his Broadway debut in 1967 with \"Does a Tiger Wear a Necktie?\" His film career took off with \"The Godfather\" series.\n\n4. **Dustin Hoffman** - Hoffman's Broadway debut was in the 1960 play \"Eh?\" by Samuel Beckett. He later became famous for films such as \"Rain Man\" and \"All the President's Men.\"\n\n5. **Jennifer Lawrence** - While primarily recognized for her roles in films like \"The Hunger Games\" series and \"Silver Linings Playbook,\" Lawrence began her career on Broadway, notably appearing in \"Time Stands Still\" in 2010.\n\n6. **Liam Neeson** - Known for action roles in films like \"Taken\" and \"Star Wars,\" Neeson started his career in theater and made his Broadway debut in 1983 with \"Anthony and Cleopatra.\"\n\n7. **Hugh Jackman** - Before becoming globally recognized as Wolverine in the X-Men series, Jackman won a Tony Award for his role in the 2004 Broadway musical \"The Boy from Oz.\"\n\n8. **Scarlett Johansson** - While primarily known for her roles in Marvel films as Black Widow, Johansson has also performed on Broadway, notably in \"A View from the Bridge\" in 2010.\n\nThese actors' transitions from Broadway to Hollywood highlight the versatility and depth of talent that theater can nurture, setting a foundation for their successful careers in film and television.\n",
        "generator": "smoothie_1", 
        "models_in_trial": "['Meta-Llama-3-70B-Instruct', 'Storm-7B', 'FsfairX-Zephyr-Chat-v0.1', 'yi-large-preview', 'Nanbeige-Plus-Chat-v0.1']",
        "selected_model": "Nanbeige-Plus-Chat-v0.1",
        "smoothie_weights": "[0.1926124005325325, 0.1591649898806868, 0.14405633409506835, 0.21382343940368917, 0.29034283608802314]"
    }
```
where:
- `instruction`: The instruction for the sample.
- `output`: The output text generated by the model.
- `generator`: The name of the method that generated the output.
- `models_in_trial`: The models that were used in the trial.
- `selected_model`: The model that was selected by Smoothie.
- `smoothie_weights`: The weights assigned to each model by Smoothie.

To evaluate the predictions, run the following command:

```bash
> cd alpaca
> ./run_alpaca.sh
```

This script runs `alpaca_eval` on all the original model predictions, and all Smoothie/baseline predictions. The results are stored to `alpaca/leaderboard.csv`.


## To run on new datasets/prompts/models

To add a new dataset, add it to `HF_TEST_DATASETS` and `HF_TRAIN_DATASETS` in `src/constants.py`, and specify any particular details for loading in `src/data_utils.py`. 
To add corresponding prompts, go to `prompt_templates/` and add `${config_filename}_multi_model.txt` in the multi-model setting, and add `${config_filename}_multi_prompt.txt` in the multi-prompt setting.
Then, create a new .yaml file in `dataset_configs` with the dataset name and the prompt filenames above. 

To add a new model, add it to `HF_MODELS` in `src/constants.py`. 


## Using Smoothie

In `tutorials/tutorial.ipynb`, we walk through how to use the Smoothie algorithm. The tutorial can be easily adapted for your use case given that you provide a .jsonl file with the dataset inputs, and several json files each containing a different model/prompt's generations.

If interested in the mathematical derivation of Smoothie, check out `tutorials/algorithm.ipynb`.