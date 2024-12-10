#!/bin/bash

set -x  # Enable command tracing

# Replication script for multiple model experiments on NLG datasets
# To run: ./replication_scripts/multimodel_exps.sh

# Where to save the results
RESULTS_DIR="smoothie_data/multi_model_results"

# Where to cache the HF datasets
HF_CACHE_DIR="cache"

# Where to save the processed datasets
DATA_DIR="smoothie_data/datasets"

# Random seed
SEED=42

# Device to use
DEVICE="cuda"

# Array of dataset configs
dataset_configs=(
    "dataset_configs/acc_group.yaml"
    "dataset_configs/rouge2_group.yaml"
    "dataset_configs/squad.yaml"
    "dataset_configs/trivia_qa.yaml"
    "dataset_configs/definition_extraction.yaml"
    "dataset_configs/cnn_dailymail.yaml"
    "dataset_configs/e2e_nlg.yaml"
    "dataset_configs/xsum.yaml"
    "dataset_configs/web_nlg.yaml"
)


for dataset_config in "${dataset_configs[@]}"
do
    echo "Processing dataset config: $dataset_config"

    # Dataset processing
    python -m src.make_dataset \
        --dataset_config $dataset_config \
        --hf_cache_dir $HF_CACHE_DIR \
        --data_dir $DATA_DIR

    # Array of models. 
    models=(
        "dolly-3b"
        "incite-3b"
        "pythia-2.8b"
        "gemma-2b"
        "llama-2-7b"
        "mistral-7b"
        "vicuna-7b"
        "gemma-7b"
        "nous-capybara"
    )

    for model in "${models[@]}"
    do
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
            --multi_model
    done

    # Pick-random baseline
    python -m src.pick_random_baseline \
        --dataset_config $dataset_config \
        --results_dir $RESULTS_DIR \
        --multi_model

    # Best-on-val baseline
    python -m src.labeled_oracle_baseline \
        --dataset_config $dataset_config \
        --results_dir $RESULTS_DIR \
        --data_dir $DATA_DIR \
        --multi_model 
    
    # PairRM
    python -m src.pair_rm_baseline \
        --dataset_config $dataset_config \
        --results_dir $RESULTS_DIR \
        --multi_model

    # Run labeled knn baseline. 
    python -m src.labeled_knn_baseline \
        --dataset_config $dataset_config \
        --results_dir $RESULTS_DIR \
        --embedding_model "all-mpnet-base-v2" \
        --multi_model

    # Run Smoothie-Global. In code we refer to this as smoothie_independent
    python -m src.run_smoothie \
        --dataset_config $dataset_config \
        --results_dir $RESULTS_DIR \
        --multi_model \
        --type sample_independent \
        --regime test_time \
        --embedding_model "all-mpnet-base-v2"

    # Run Smoothie-Local. In code we refer to this as smoothie_dependent
    python -m src.run_smoothie \
        --dataset_config $dataset_config \
        --results_dir $RESULTS_DIR \
        --multi_model \
        --type sample_dependent \
        --regime test_time \
        --k 1 \
        --embedding_model "all-mpnet-base-v2"

    # Evaluate
    python -m src.evaluate.evaluate \
        --dataset_config $dataset_config \
        --results_dir $RESULTS_DIR --redo --multi_model

    echo "Finished processing $dataset_config"
    echo "----------------------------------------"
done

set +x  # Disable command tracing
