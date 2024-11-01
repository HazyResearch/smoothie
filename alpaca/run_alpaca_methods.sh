#!/bin/bash

# Array of embedding models
embedding_models=("all-mpnet-base-v2")

for embedding_model in "${embedding_models[@]}"
do
    python3 -m alpaca.alpaca_generate_predictions --embedding_model $embedding_model --k 1 
done
