#!/bin/bash

# Replication script for multiple model experiments
# To run: ./replication_scripts/multimodel_exps.sh

RESULTS_DIR="smoothie_data/multi_model_results"

# Array of dataset configs
dataset_configs=(
    "dataset_configs/squad.yaml"
    #"dataset_configs/trivia_qa.yaml"
    #"dataset_configs/definition_extraction.yaml"
    #"dataset_configs/cnn_dailymail.yaml"
    #"dataset_configs/e2e_nlg.yaml"
    #"dataset_configs/xsum.yaml"
    #"dataset_configs/web_nlg.yaml"
    #"dataset_configs/acc_group.yaml"
    #"dataset_configs/rouge2_group.yaml"
)


for dataset_config in "${dataset_configs[@]}"
do
    #echo "Processing dataset config: $dataset_config"

    #python -m src.make_dataset \
    #    --dataset_config $dataset_config

    #python -m src.get_generations \
    #    --dataset_config $dataset_config \
    #    --model_group $model_group \
    #    --results_dir $RESULTS_DIR \
    #    --multi_model

    # Pick random baseline
    python -m src.pick_random_baseline \
        --dataset_config $dataset_config \
        --results_dir $RESULTS_DIR \
        --multi_model --redo

    # Labeled oracle
    python -m src.labeled_oracle \
        --dataset_config $dataset_config \
        --results_dir $RESULTS_DIR \
        --multi_model \
        --redo

    # Smoothie sample independent
    #python -m src.run_smoothie \
    #    --dataset_config $dataset_config \
    #    --model_group $model_group \
    #    --results_dir $RESULTS_DIR \
    #    --multi_model \
    #    --type sample_independent --redo

    # Smoothie sample dependent
    #python -m src.run_smoothie \
    #    --dataset_config $dataset_config \
    #    --model_group $model_group \
    #    --results_dir $RESULTS_DIR \
    #    --multi_model \
    #    --type sample_dependent \
    #    --k 20 --redo

    # Evaluate
    python -m src.evaluate.evaluate \
        --dataset_config $dataset_config \
        --results_dir $RESULTS_DIR --redo

    echo "Finished processing $dataset_config"
    echo "----------------------------------------"
done

echo "All dataset configs processed."