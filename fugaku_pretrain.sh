#!/bin/bash
#PJM -L "rscunit=rscunit_ft01,rscgrp=eap-small"
#PJM -L elapse=08:00:00
#PJM -L "node=4"
#PJM -j
#PJM -S

source ~/mytorch/activate.sh
LD_PRELOAD=libtcmalloc.so mpirun -n $PJM_MPI_PROC python3 -m langmo.pretraining pretrain.yaml 

