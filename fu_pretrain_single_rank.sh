#!/usr/bin/bash

source /local/venv/bin/activate

export NUM_GPUS_PER_NODE=0
export TOKENIZERS_PARALLELISM=true
export PROTONN_DISTRIBUTED_BACKEND=MPI
export WANDB_MODE="online"
export OMP_NUM_THREADS=48 


python -m langmo.training.mlm pretrain.yaml
