#!/bin/bash

find_free_gpu() {
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | sort -nr -k2 | head -n 1 | awk '{print $1}'
}

if [[ "$1" == "cpu" ]]; then
    echo "Forcing CPU execution."
    export CUDA_VISIBLE_DEVICES=""
    shift

elif [[ "$1" =~ ^[0-9]+$ ]]; then
    export CUDA_VISIBLE_DEVICES=$1
    echo "Using manually specified GPU $1."
    shift

else
    GPU=$(find_free_gpu)
    if [[ -z "$GPU" ]]; then
        echo "No available GPUs found. Exiting."
        exit 1
    fi
    export CUDA_VISIBLE_DEVICES=$GPU
    echo "Using GPU $GPU with the most free memory."
fi


# python -m execute.runners.zeroshot
python -m execute.runners.llama_base


