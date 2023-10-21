#!/usr/bin/bash

set -x

YAML_FILE=$1
NODES=${2:-128}
ELAPSE=${3:-8:30:00}
DATADIR=/home/ra000012/data
GROUP=ra000012
CP=$HOME/bin/mpicp
if [ ! -e ${CP} ]; then
  CP=cp
  echo "WARNING: $HOME/bin/mpicp doesn't exist! (Compile it, it should make things faster, using cp instead)"
fi
LOCAL_PYTORCH_TGZ=${DATADIR}/NLP/local-v1.13-langmo-mod.tgz
X_PARAMS=(-x PJM_LLIO_GFSCACHE=/vol0004)

if [ ! -f $YAML_FILE ]; then
  echo "YAML_FILE: $YAML_FILE (doesn't exist)"
  echo "NODES: $NODES"
  echo "ELAPSE: $ELAPSE"
  echo "USAGE: $0 \$YAML_FILE [ \$NODES ] [ \$ELAPSE ]"
  exit 0
fi

# useful variables: PJM_JOBNAME PJM_JOBID PJM_JOBDIR

# JOBNAME: as it appears in `pjstat`.
JOBNAME="langmo-N${NODES}-${YAML_FILE}"
# outprefix: where all the output (stdout, stderr) is dumped.
OUTPREFIX="${DATADIR}/NLP_outs/pretrain/${JOBNAME}"
mkdir -p "${OUTPREFIX}"

if [ ${NODES} -gt 348 ]; then
    rscgrp="large";
else
    rscgrp="small";
fi
if email=$(git config --get user.email); then
    email_args="-m b,e --mail-list ${email}"
else
    echo "$0 WARNING: git email not set!"
fi

PJSUB_ARGS=(
    -g ${GROUP}
    ${X_PARAMS[@]}
    -N ${JOBNAME}
    -L "rscgrp=${rscgrp}"
    -L "elapse=${ELAPSE}"
    -L "node=${NODES}"
    --mpi "proc=${NODES}"
    -o ${OUTPREFIX}/%j.stdout
    -e ${OUTPREFIX}/%j.stderr
    --spath ${OUTPREFIX}/%j.stat
    --llio localtmp-size=40Gi
    -j -S
    ${email_args}
)

pjsub ${PJSUB_ARGS[@]} << EOF 
#!/usr/bin/bash
set -x

# Prepare output dir!
OUTDIR="${OUTPREFIX}/\${PJM_JOBID}"
mkdir -p \${OUTDIR}

# Prepare venv!
mpirun -of-proc \${OUTDIR}/mpi ${CP} ${LOCAL_PYTORCH_TGZ} /local/
mpirun -of-proc \${OUTDIR}/mpi tar -I pigz -xf /local/$(basename ${LOCAL_PYTORCH_TGZ}) -C /local
source "/local/venv/bin/activate"

# Run langmo
MPIEXEC_ARGS=(
   -of-proc \${OUTDIR}/mpi
   -x NUM_GPUS_PER_NODE=0
   -x TOKENIZERS_PARALLELISM=true
   -x PROTONN_DISTRIBUTED_BACKEND=MPI
   -x WANDB_MODE="disabled"
   -x OMP_NUM_THREADS=48
   -x LD_PRELOAD=/local/opt/lib/libtcmalloc.so
)
mpirun \${MPIEXEC_ARGS[@]} python3 -m langmo.training.mlm ${YAML_FILE}
EOF
