#!/bin/bash

# Replication script for multiple prompt experiments

# Where to save the results
RESULTS_DIR="smoothie_data/multi_prompt_results"

# Where to cache the HF datasets
HF_CACHE_DIR="cache"

# Where to save the processed datasets
DATA_DIR="smoothie_data/datasets"

# Random seed
SEED=42

# Device to use
DEVICE="cuda"

# Dataset configs to run
data_configs=(
    "dataset_configs/squad.yaml"
    "dataset_configs/trivia_qa.yaml"
    "dataset_configs/definition_extraction.yaml"
    "dataset_configs/cnn_dailymail.yaml"
    "dataset_configs/e2e_nlg.yaml"
    "dataset_configs/xsum.yaml"
    "dataset_configs/web_nlg.yaml"
)

# Model
model="llama-2-7b"

for dataset_config in "${data_configs[@]}"; do
    echo "Processing dataset config: $dataset_config"

    # Dataset processing
    python -m src.make_dataset \
        --dataset_config $dataset_config \
        --hf_cache_dir $HF_CACHE_DIR \
        --data_dir $DATA_DIR

    python -m src.get_generations \
        --model $model \
        --dataset_config $dataset_config \
        --data_dir $DATA_DIR \
        --device $DEVICE \
        --hf_cache_dir $HF_CACHE_DIR \
        --results_dir $RESULTS_DIR \
        --n_generations 1 \
        --temperature 0.0 \
        --seed $SEED \
        --multi_prompt

    # Pick random baseline
    python -m src.pick_random_baseline \
        --dataset_config $dataset_config \
        --results_dir $RESULTS_DIR \
        --multi_prompt 

    # Best-on-val baseline
    python -m src.labeled_oracle_baseline \
        --dataset_config $dataset_config \
        --results_dir $RESULTS_DIR \
        --data_dir $DATA_DIR \
        --multi_prompt

    # Run Smoothie-Global. In code we refer to this as smoothie_independent
    python -m src.run_smoothie \
        --dataset_config $dataset_config \
        --results_dir $RESULTS_DIR \
        --multi_prompt \
        --type sample_independent \
        --regime test_time \
        --embedding_model "all-mpnet-base-v2"

    # Run Smoothie-Local. In code we refer to this as smoothie_dependent
    python -m src.run_smoothie \
        --dataset_config $dataset_config \
        --results_dir $RESULTS_DIR \
        --multi_prompt \
        --type sample_dependent \
        --regime test_time \
        --k 1 \
        --embedding_model "all-mpnet-base-v2"

    # Evaluate
    python -m src.evaluate.evaluate \
        --dataset_config $dataset_config \
        --multi_prompt \
        --model $model \
        --results_dir $RESULTS_DIR --redo

    echo "Finished processing $dataset_config"
    echo "----------------------------------------"
done