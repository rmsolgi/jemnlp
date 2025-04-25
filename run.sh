#!/bin/bash

# Function to find a GPU with the most free memory
find_free_gpu() {
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | sort -nr -k2 | head -n 1 | awk '{print $1}'
}

# Check if "cpu" is passed as the first argument
if [[ "$1" == "cpu" ]]; then
    echo "Forcing CPU execution."
    export CUDA_VISIBLE_DEVICES=""
    shift

# Check if a specific GPU ID is passed
elif [[ "$1" =~ ^[0-9]+$ ]]; then
    export CUDA_VISIBLE_DEVICES=$1
    echo "Using manually specified GPU $1."
    shift

# Otherwise, pick GPU with most free memory
else
    GPU=$(find_free_gpu)
    if [[ -z "$GPU" ]]; then
        echo "No available GPUs found. Exiting."
        exit 1
    fi
    export CUDA_VISIBLE_DEVICES=$GPU
    echo "Using GPU $GPU with the most free memory."
fi


python -m execute.runners.zeroshot


