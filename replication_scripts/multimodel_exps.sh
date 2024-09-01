#!/bin/bash

# Replication script for multiple model experiments
# To run: ./replication_scripts/multimodel_exps.sh

RESULTS_DIR="smoothie_data/multi_model_results"

# Array of dataset configs
dataset_configs=(
    "dataset_configs/squad.yaml"
    "dataset_configs/trivia_qa.yaml"
    "dataset_configs/definition_extraction.yaml"
    "dataset_configs/cnn_dailymail.yaml"
    "dataset_configs/e2e_nlg.yaml"
    "dataset_configs/xsum.yaml"
    "dataset_configs/web_nlg.yaml"
    "dataset_configs/acc_group.yaml"
    "dataset_configs/rouge2_group.yaml"
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
        --multi_model

    # Labeled oracle
    python -m src.labeled_oracle \
        --dataset_config $dataset_config \
        --results_dir $RESULTS_DIR \
        --multi_model 

    # Array of embedding models
    embedding_models=("bge-small-en-v1.5" "all-mpnet-base-v2")

    for embedding_model in "${embedding_models[@]}"
    do
        # Smoothie sample independent / test time
        python -m src.run_smoothie \
            --dataset_config $dataset_config \
            --results_dir $RESULTS_DIR \
            --multi_model \
            --type sample_independent \
            --regime test_time \
            --embedding_model $embedding_model

        # Smoothie sample dependent / train time
        python -m src.run_smoothie \
            --dataset_config $dataset_config \
            --results_dir $RESULTS_DIR \
            --multi_model \
            --type sample_independent \
            --regime train_time \
            --embedding_model $embedding_model

        # Smoothie sample dependent / test time
        python -m src.run_smoothie \
            --dataset_config $dataset_config \
            --results_dir $RESULTS_DIR \
            --multi_model \
            --type sample_dependent \
            --regime test_time \
            --k 1 \
            --embedding_model $embedding_model
        
        # Smoothie sample dependent / train time with k = 20
        python -m src.run_smoothie \
            --dataset_config $dataset_config \
            --results_dir $RESULTS_DIR \
            --multi_model \
            --type sample_dependent \
            --regime train_time \
            --k 20 \
            --embedding_model $embedding_model

        # Smoothie sample dependent / train time with k = 50
        python -m src.run_smoothie \
            --dataset_config $dataset_config \
            --results_dir $RESULTS_DIR \
            --multi_model \
            --type sample_dependent \
            --regime train_time \
            --k 50 \
            --embedding_model $embedding_model
    done


    # Evaluate
    python -m src.evaluate.evaluate \
        --dataset_config $dataset_config \
        --results_dir $RESULTS_DIR --redo --multi_model

    echo "Finished processing $dataset_config"
    echo "----------------------------------------"
done

echo "All dataset configs processed."