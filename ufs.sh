#!/usr/bin/bash

# set -x

source fu_common.src

MODULE="langmo.training.mlm"
YAML_FILE="pretrain_minimal_test.yaml"
NODES=128
ELAPSE="8:30:00"
DATADIR="/home/ra000012/data"
GROUP="ra000012"
PY_ENV="${DATADIR}/NLP/local-v1.13-langmo-mod.tgz"
OUTPUT="NLP_outs"

if [[ $# -gt 0 ]]; then
  if [[ "$1" =~ ^((-{1,2})([Hh]$|[Hh][Ee][Ll][Pp])|)$ ]]; then
    print_usage; exit 1
  else
    while [[ $# -gt 0 ]]; do
      opt="$1"
      shift;
      current_arg="$1"
      if [[ "$current_arg" =~ ^-{1,2}.* ]]; then
        echo "WARNING: You may have left an argument blank. Double check your command."
      fi
      case "$opt" in
        "-c"|"--config"     ) YAML_FILE="$1"; shift;;
        "-e"|"--elapse"     ) ELAPSE="$1";    shift;;
        "-g"|"--group"      ) GROUP="$1";     shift;;
        "-m"|"--module"     ) MODULE="$1";    shift;;
        "-n"|"--nodes"      ) NODES="$1";     shift;;
        "-o"|"--output"     ) OUTPUT="$1";    shift;;
        "-p"|"--py-env"     ) PY_ENV="$1";    shift;;
        *                   ) echo "ERROR: Invalid option: \""$opt"\"" >&2
                              exit 1;;
      esac
    done
  fi
else
  echo "Using default arguments!"
fi

echo "Module: ${MODULE}"
echo "YAML Config File: ${YAML_FILE}"
echo "Number of Compute Nodes: ${NODES}"
echo "Elapse Time: ${ELAPSE}"
echo "Group: ${GROUP}"
echo "Python Environment: ${PY_ENV}"

CP=$HOME/bin/mpicp
if [ ! -e ${CP} ]; then
  echo "WARNING: ${CP} doesn't exist! (Compile it, it should make things faster, using cp instead)"
  CP=cp
fi

LOCAL_PYTORCH_TGZ=$PY_ENV
X_PARAMS=(-x PJM_LLIO_GFSCACHE=/vol0004)

if [ ! -f $YAML_FILE ]; then
  echo "YAML_FILE: $YAML_FILE (doesn't exist)"
  exit 0
fi

# useful variables: PJM_JOBNAME PJM_JOBID PJM_JOBDIR
JOBNAME="langmo-N${NODES}-$(basename ${YAML_FILE})"  # JOBNAME: as it appears in `pjstat`.
OUTDIR="${DATADIR}/${OUTPUT}/${MODULE}/${JOBNAME}/$(date +%Y%m%d-%H%M%S)" # outprefix: where all the output (stdout, stderr) is dumped.
mkdir -p "${OUTDIR}"

echo "Job Name: $JOBNAME"
echo "Output Directory: $OUTDIR"

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

pjsub "${PJSUB_ARGS[@]}" << EOF
#!/usr/bin/bash

# set -x

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
