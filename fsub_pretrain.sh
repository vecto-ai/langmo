#!/usr/bin/bash

PREFIX="/home/ra000012/data/fj-pytorch-builds/v1.10"
NODES=128
ELAPSE="8:30:00"

jobname="$(basename $0)"
jobname="${jobname%.*}"
jobname="${jobname##*_}"

outprefix="NLP_outs/${jobname}"

# PJM_JOBNAME PJM_JOBID PJM_JOBDIR
[ -e "${outprefix}" ] || mkdir -p "${outprefix}"

cat << EOF | pjsub
#!/usr/bin/bash
#PJM -N $jobname
#PJM -g ra000012
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -L "rscgrp=small"
#PJM -L "elapse=${ELAPSE}"
#PJM -L "node=${NODES}"
#PJM --mpi "proc=${NODES}"
#PJM -j
#PJM -S
#PJM -o ${outprefix}/%j.stdout
#PJM -e ${outprefix}/%j.stderr
#PJM --spath ${outprefix}/%j.stat

# #PJM --llio sharedtmp-size=80Gi

source "$PREFIX/venv/bin/activate"
# export PATH="$PREFIX/opt/bin${PATH:+:${PATH}}"
# export LD_LIBRARY_PATH="$PREFIX/opt/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

OUTDIR="${outprefix}/\${PJM_JOBID}"
[ -e \${OUTDIR} ] || mkdir -p \${OUTDIR}

export NUM_GPUS_PER_NODE=0
export TOKENIZERS_PARALLELISM=true
export PROTONN_DISTRIBUTED_BACKEND=MPI
export WANDB_MODE="disabled"

LD_PRELOAD=$PREFIX/opt/lib/libtcmalloc.so OMP_NUM_THREADS=48 \
    mpirun \
    -of-proc \${OUTDIR}/mpi \
    -x TOKENIZERS_PARALLELISM \
    -x NUM_GPUS_PER_NODE \
    -x PROTONN_DISTRIBUTED_BACKEND \
    -x WANDB_MODE \
    python3 -m langmo.training.mlm pretrain.yaml
EOF
