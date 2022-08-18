#!/usr/bin/bash
#PJM -L "rscunit=rscunit_ft01,rscgrp=small"
#PJM -L elapse=08:00:00
#PJM -L "node=16"
#PJM --mpi "proc=16"
#PJM -j
#PJM -g ra000012
#PJM -S
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
##PJM --llio sharedtmp-size=80Gi

source /home/ra000012/data/pytorch.1.10/venv/bin/activate

export NUM_GPUS_PER_NODE=0
export TOKENIZERS_PARALLELISM=true
export PL_TORCH_DISTRIBUTED_BACKEND=MPI

# ======== Main ===========

OMP_NUM_THREADS=48 mpirun \
    -x TOKENIZERS_PARALLELISM \
    -x NUM_GPUS_PER_NODE \
    -x PL_TORCH_DISTRIBUTED_BACKEND \
    python3 -m langmo.benchmarks.GLUE finetune.yaml sst2

