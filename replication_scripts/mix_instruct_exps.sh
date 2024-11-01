#!/bin/bash

# Replication script for multiple model experiments
# To run: ./replication_scripts/multimodel_exps.sh

RESULTS_DIR="smoothie_data/multi_model_results"


dataset_config="dataset_configs/mix_instruct.yaml"

#python -m src.setup_mix_instruct \
#    --results_dir $RESULTS_DIR

# Pick random baseline
python -m src.pick_random_baseline \
    --dataset_config $dataset_config \
    --results_dir $RESULTS_DIR \
    --multi_model

# NOTE: We don't have train annotations for mix instruct so we don't run labeled oracle


# PairRM
python -m src.pair_rm_baseline \
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
        --embedding_model $embedding_model --redo

    # Smoothie sample dependent / train time
    python -m src.run_smoothie \
        --dataset_config $dataset_config \
        --results_dir $RESULTS_DIR \
        --multi_model \
        --type sample_independent \
        --regime train_time \
        --embedding_model $embedding_model --redo

    # Smoothie sample dependent / test time
    python -m src.run_smoothie \
        --dataset_config $dataset_config \
        --results_dir $RESULTS_DIR \
        --multi_model \
        --type sample_dependent \
        --regime test_time \
        --k 1 \
        --embedding_model $embedding_model --redo
    
    # Smoothie sample dependent / train time with k = 20
    python -m src.run_smoothie \
        --dataset_config $dataset_config \
        --results_dir $RESULTS_DIR \
        --multi_model \
        --type sample_dependent \
        --regime train_time \
        --k 20 \
        --embedding_model $embedding_model --redo

    # Smoothie sample dependent / train time with k = 50
    python -m src.run_smoothie \
        --dataset_config $dataset_config \
        --results_dir $RESULTS_DIR \
        --multi_model \
        --type sample_dependent \
        --regime train_time \
        --k 50 \
        --embedding_model $embedding_model --redo
done

python -m src.evaluate.evaluate \
    --dataset_config $dataset_config \
    --results_dir $RESULTS_DIR --redo --multi_model

