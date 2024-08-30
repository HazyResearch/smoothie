#!/bin/bash

# Replication script for multiple prompt experiments
# To run: ./replication_scripts/multiprompt_exps.sh

RESULTS_DIR="smoothie_data/multi_prompt_results"

# Dataset configs to run
data_configs=(
    "dataset_configs/squad.yaml"
    #"dataset_configs/trivia_qa.yaml"
    #"dataset_configs/definition_extraction.yaml"
    #"dataset_configs/cnn_dailymail.yaml"
    #"dataset_configs/e2e_nlg.yaml"
    #"dataset_configs/xsum.yaml"
    #"dataset_configs/web_nlg.yaml"
)

# Model
model="falcon-1b"

for dataset_config in "${data_configs[@]}"; do
    echo "Processing dataset config: $dataset_config"

    #python -m src.make_dataset \
    #    --dataset_config $dataset_config

    #python -m src.get_generations \
    #    --dataset_config $dataset_config \
    #    --model $model \
    #    --results_dir $RESULTS_DIR \
    #    --multi_prompt

    # Pick random baseline
    python -m src.pick_random_baseline \
        --dataset_config $dataset_config \
        --model $model \
        --results_dir $RESULTS_DIR \
        --multi_prompt --redo

    # Labeled oracle
    python -m src.labeled_oracle \
        --dataset_config $dataset_config \
        --model $model \
        --results_dir $RESULTS_DIR \
        --multi_prompt --redo

    # Smoothie sample independent
    #python -m src.run_smoothie \
    #    --dataset_config $dataset_config \
    #    --model $model \
    #    --results_dir $RESULTS_DIR \
    #    --multi_prompt \
    #    --type sample_independent --redo

    # Smoothie sample dependent
    #python -m src.run_smoothie \
    #    --dataset_config $dataset_config \
    #    --model $model \
    #    --results_dir $RESULTS_DIR \
    #    --multi_prompt \
    #    --type sample_dependent \
    #    --k 20 --redo

    # Evaluate
    python -m src.evaluate.evaluate \
        --dataset_config $dataset_config \
        --multi_prompt \
        --model $model \
        --results_dir $RESULTS_DIR --redo

    echo "Finished processing $dataset_config"
    echo "----------------------------------------"
done