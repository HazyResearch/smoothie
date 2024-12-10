#!/bin/bash

# Replication script for multiple model experiments on MIXInstruct

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


dataset_config="dataset_configs/mix_instruct.yaml"

python -m src.setup_mix_instruct \
    --results_dir $RESULTS_DIR \
    --hf_cache_dir $HF_CACHE_DIR \
    --data_dir $DATA_DIR

# Pick-random baseline
python -m src.pick_random_baseline \
    --dataset_config $dataset_config \
    --results_dir $RESULTS_DIR \
    --multi_model


# PairRM
python -m src.pair_rm_baseline \
    --dataset_config $dataset_config \
    --results_dir $RESULTS_DIR \
    --multi_model


# Run Smoothie-Local. In code we refer to this as smoothie_dependent
python -m src.run_smoothie \
    --dataset_config $dataset_config \
    --results_dir $RESULTS_DIR \
    --multi_model \
    --type sample_dependent \
    --regime test_time \
    --k 1 \
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

python -m src.evaluate.evaluate \
    --dataset_config $dataset_config \
    --results_dir $RESULTS_DIR --redo --multi_model

