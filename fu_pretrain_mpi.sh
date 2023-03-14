#!/usr/bin/bash
#PJM -L "rscgrp=large"
#PJM -L elapse=24:00:00
#PJM -L "node=1024"
#PJM --mpi "proc=1024"
#PJM -j
#PJM -g ra000012
#PJM -S
#PJM -o logs/%n.%j.stdout
#PJM -e logs/%n.%j.stderr
#PJM --spath logs/%n.%j.stat
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM --llio localtmp-size=40Gi
#PJM -m b,e
#PJM --mail-list alexander.drozd@gmail.com

#TODO: mpi out path
## -of-proc

export LD_PRELOAD=/usr/lib/FJSVtcs/ple/lib64/libpmix.so
# TODO: think where to add tcmalloc
# LD_PRELOAD=$PREFIX/prefix/lib/libtcmalloc.so
mpirun fu_prepare_venv.sh
mpirun fu_pretrain_single_rank.sh
