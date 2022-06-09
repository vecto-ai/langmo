#!/bin/bash
#$ -cwd
#$ -l rt_AF=4
#$ -l h_rt=62:00:00
#$ -N Pretrain
#$ -j y
#$ -o ./logs/pretrain/$JOB_NAME.o$JOB_ID

# ======== Modules ========
source /etc/profile.d/modules.sh
source modules.sh

export NUM_GPUS_PER_NODE=8
NUM_NODES=${NHOSTS}
NUM_PROCS=$(expr ${NUM_NODES} \* ${NUM_GPUS_PER_NODE})
# NUM_GPUS_PER_NODE=8
# export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=true
export PL_TORCH_DISTRIBUTED_BACKEND=NCCL
export NCCL_DEBUG=WARN

# MPIOPTS="-np ${NUM_PROCS} -map-by ppr:${NUM_GPUS_PER_NODE}:node -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_include bond0"
MPIOPTS="-np ${NUM_PROCS} -map-by ppr:${NUM_GPUS_PER_NODE}:node -mca pml ob1 -mca btl self,tcp -mca btl_tcp_if_include bond0"

echo ${MPIOPTS}
# ======== Main ===========
# WANDB_MODE=offline
mpirun ${MPIOPTS} \
    -x TOKENIZERS_PARALLELISM \
    -x NCCL_DEBUG \
    -x NUM_GPUS_PER_NODE \
    -x HOROVOD_CACHE_CAPACITY \
    python3 -m langmo.pretraining pretrain_shared.yaml

