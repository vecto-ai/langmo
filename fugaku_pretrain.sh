#!/usr/bin/bash
#PJM -L "rscunit=rscunit_ft01,rscgrp=small"
#PJM -L elapse=08:00:00
#PJM -L "node=8"
#PJM --mpi "proc=32"
#PJM -j
#PJM -g ra000012
#PJM -S
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
##PJM --llio sharedtmp-size=80Gi

source ~/ra000012/venv/bin/activate

# WANDB_MODE=offline
WANDB_MODE=disabled \
# mpirun -x LD_PRELOAD=libtcmalloc.so python3 -m langmo.pretraining pretrain.yaml
mpirun -x LD_PRELOAD=libtcmalloc.so python3 -m langmo.NLI pretrain_minimal_test.yaml

