#!/usr/bin/bash

NODES=128
ELAPSE="8:30:00"
YAML_FILE=$1
DATADIR=/home/ra000012/data
GROUP=ra000012
X_PARAM="-x PJM_LLIO_GFSCACHE=/vol0004"
# useful variables: PJM_JOBNAME PJM_JOBID PJM_JOBDIR

LOCAL_PYTORCH_TGZ=${DATADIR}/NLP/local-v1.13-langmo-mod.tgz
# jobname: as it appears in `pjstat`.
jobname="langmo-N${NODES}-${YAML_FILE}"
# outprefix: where all the output (stdout, stderr) is dumped.
OUTPREFIX="${DATADIR}/NLP_outs/${jobname}"


# Create ${OUTPREFIX} if it doesn't exist!
[ -e "${OUTPREFIX}" ] || mkdir -p "${OUTPREFIX}"

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
    ${X_PARAM}
    -N ${jobname}
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

cat << EOF | pjsub ${PJSUB_ARGS[@]}
#!/usr/bin/bash

# Prepare venv!
cp ${LOCAL_PYTORCH_TGZ} /local/
cd /local
pigz -dc $(basename ${LOCAL_PYTORCH_TGZ}) | tar xf -

source "/local/venv/bin/activate"
# export PATH="/local/opt/bin${PATH:+:${PATH}}"
# export LD_LIBRARY_PATH="/local/opt/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

OUTDIR="${OUTPREFIX}/\${PJM_JOBID}"
[ -e \${OUTDIR} ] || mkdir -p \${OUTDIR}

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
