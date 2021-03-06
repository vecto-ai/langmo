#!/bin/bash
#$ -cwd
#$ -l rt_AF=1
#$ -l h_rt=24:00:00
#$ -N Pretrain
#$ -j y
#$ -o $JOB_NAME.o$JOB_ID

# ======== Modules ========
source /etc/profile.d/modules.sh
source modules.sh

NUM_NODES=${NHOSTS}
NUM_GPUS_PER_NODE=8
NUM_GPUS_PER_SOCKET=$(expr ${NUM_GPUS_PER_NODE} / 2)
NUM_PROCS=$(expr ${NUM_NODES} \* ${NUM_GPUS_PER_NODE})
TOKENIZERS_PARALLELISM=true

MPIOPTS="-np ${NUM_PROCS} -map-by ppr:${NUM_GPUS_PER_NODE}:node -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_include bond0"

# ======== Main ===========

mpirun ${MPIOPTS} \
    python3 -m langmo.pretraining pretrain.yaml


# horovodrun -np 4 python3 main.py

