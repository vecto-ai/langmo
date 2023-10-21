#!/usr/bin/bash

# set -x

source fu_common.src

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
JOBNAME="langmo-N${NODES}-$(basename ${YAML_FILE})"
# outprefix: where all the output (stdout, stderr) is dumped.
OUTDIR="${DATADIR}/NLP_outs/pretrain/${JOBNAME}"
mkdir -p "${OUTDIR}"

PJSUB_ARGS=(
    -g ${GROUP}
    ${X_PARAMS[@]}
    -N ${JOBNAME}
    -L "rscgrp=$(get_rscgrp ${NODES})"
    -L "elapse=${ELAPSE}"
    -L "node=${NODES}"
    --mpi "proc=${NODES}"
    -o ${OUTDIR}/%j.stdout
    -e ${OUTDIR}/%j.stderr
    --spath ${OUTDIR}/%j.stat
    --llio localtmp-size=40Gi
    -j -S
    $(get_emailargs)
)

pjsub ${PJSUB_ARGS[@]} << EOF 
#!/usr/bin/bash
set -x

# Prepare output dir!
MPI_OUTDIR="${OUTDIR}/\${PJM_JOBID}"
mkdir -p \${MPI_OUTDIR}

# Prepare venv!
mpirun -of-proc \${MPI_OUTDIR}/mpi ${CP} ${LOCAL_PYTORCH_TGZ} /local/
mpirun -of-proc \${MPI_OUTDIR}/mpi tar -I pigz -xf /local/$(basename ${LOCAL_PYTORCH_TGZ}) -C /local
source "/local/venv/bin/activate"

# Run langmo
MPIEXEC_ARGS=(
   -of-proc \${MPI_OUTDIR}/mpi
   -x NUM_GPUS_PER_NODE=0
   -x TOKENIZERS_PARALLELISM=true
   -x PROTONN_DISTRIBUTED_BACKEND=MPI
   -x WANDB_MODE="disabled"
   -x OMP_NUM_THREADS=48
   -x LD_PRELOAD=/local/opt/lib/libtcmalloc.so
)
mpirun \${MPIEXEC_ARGS[@]} python3 -m langmo.training.mlm ${YAML_FILE}
EOF
