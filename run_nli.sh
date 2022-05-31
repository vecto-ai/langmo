#!/bin/bash
#$ -cwd
#$ -l rt_AF=1
#$ -l h_rt=1:00:00
#$ -N finetune
#$ -j y
#$ -o ./logs/$JOB_NAME.o$JOB_ID

# ======== Modules ========
source /etc/profile.d/modules.sh
source modules.sh

NUM_NODES=${NHOSTS}
export NUM_GPUS_PER_NODE=8
NUM_GPUS_PER_SOCKET=$(expr ${NUM_GPUS_PER_NODE} / 2)
NUM_PROCS=$(expr ${NUM_NODES} \* ${NUM_GPUS_PER_NODE})

MPIOPTS="-np ${NUM_PROCS} -map-by ppr:${NUM_GPUS_PER_NODE}:node -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_include bond0"

export TOKENIZERS_PARALLELISM=true
export PL_TORCH_DISTRIBUTED_BACKEND=NCCL

# ======== Main ===========

mpirun ${MPIOPTS} \
    -x TOKENIZERS_PARALLELISM \
    -x NUM_GPUS_PER_NODE \
    -x PL_TORCH_DISTRIBUTED_BACKEND \
    python3 -m langmo.benchmarks.NLI nli.yaml

# horovodrun -np 4 python3 main.py

