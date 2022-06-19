#!/usr/bin/bash
#PJM -L "rscgrp=small"
#PJM -L elapse=8:00:00
#PJM -L "node=64"
#PJM --mpi "proc=64"
#PJM -j
#PJM -g ra000012
#PJM -S
#PJM -o out-files/%n.%j.stdout
#PJM -e out-files/%n.%j.stderr
#PJM --spath out-files/%n.%j
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
##PJM --llio sharedtmp-size=80Gi
##PJM -L "rscunit=rscunit_ft01,rscgrp=small"

source /home/ra000012/data/pytorch110.Kfast/venv/bin/activate

export WANDB_MODE="disbled"

OMP_NUM_THREADS=48 mpirun \
    -x WANDB_MODE \
    python3 -m langmo.pretraining pretrain.yaml

