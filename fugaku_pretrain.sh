#!/bin/bash
#PJM -L "rscunit=rscunit_ft01,rscgrp=small"
#PJM -L elapse=08:00:00
#PJM -L "node=8"
#PJM -j
#PJM -S
#PJM --mpi proc=32
#PJM --llio sharedtmp-size=80Gi

# source ~/mytorch/activate.sh
export PATH=/home/apps/oss/PyTorch-1.7.0/bin:$PATH
export LD_LIBRARY_PATH=/home/apps/oss/PyTorch-1.7.0/lib:$LD_LIBRARY_PATH

WANDB_MODE=dryrun LD_PRELOAD=libtcmalloc.so mpirun -n $PJM_MPI_PROC python3 -m langmo.pretraining pretrain.yaml

