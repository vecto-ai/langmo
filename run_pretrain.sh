#!/bin/bash
#$ -cwd
#$ -l rt_F=4
#$ -l h_rt=07:55:00
#$ -N Pretrain
#$ -j y
#$ -o $JOB_NAME.o$JOB_ID

# ======== Modules ========
source /etc/profile.d/modules.sh
source modules.sh

NUM_NODES=${NHOSTS}
NUM_GPUS_PER_NODE=4
NUM_GPUS_PER_SOCKET=$(expr ${NUM_GPUS_PER_NODE} / 2)
NUM_PROCS=$(expr ${NUM_NODES} \* ${NUM_GPUS_PER_NODE})

MPIOPTS="-np ${NUM_PROCS} -map-by ppr:${NUM_GPUS_PER_NODE}:node -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_include bond0"

# ======== Main ===========

mpirun ${MPIOPTS} \
    python3 -m langmo.pretraining pretrain.yaml

# horovodrun -np 4 python3 main.py

