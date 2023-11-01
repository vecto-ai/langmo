#!/usr/bin/env python

import yaml
import git
import grp
import os
import argparse
from pathlib import Path


def get_config_data(config_path):
    with open(config_path) as f:
        config_data = yaml.load(f)
    if "path_result" not in config_data:
        raise ValueError(f"'path_result' not found in {config_path}")
    return config_data


def make_config_yaml(config_data):
    decorated_result_path = Path(config_data["path_result"]) / "decorated"
    decorated_result_path.mkdir(exist_ok=True)
    config_data["path_result"] = str(decorated_result_path)
    config_data["create_unique_path"] = False
    with open(decorated_result_path / "config.yaml", "w") as f:
        yaml.dump(config_data, f)
    return config_data


def make_pjsub_script(args, config_data):
    result_path = Path(config_data["path_result"])
    module = "foobar"
    gid = os.stat(result_path).st_gid
    group = grp.getgrgid(gid)[0]
    pjm_lines = [f"-g {group}"]
    try:
        email = git.config.GitConfigParser().get_value("user", "email")
        pjm_lines.append(f"-m b,e --mail-list {email}")
    except Exception as e:
        print(f"Warning: git email not set")

    script_path = result_path / "submit.sh"
    # -x PJM_LLIO_GFSCACHE=/vol0004
    # -N ${JOBNAME}
    # -L "rscgrp=$(get_rscgrp ${NODES})"
    # -L "elapse=${ELAPSE}"
    # -L "node=${NODES}"
    # --mpi "proc=${NODES}"
    # -o ${OUTDIR}/%j.stdout
    # -e ${OUTDIR}/%j.stderr
    # --spath ${OUTDIR}/%j.stat
    # --llio localtmp-size=40Gi
    # -j -S
    lines = ["#!/bin/bash"]
    lines += [f"#PJM {line}" for line in pjm_lines]
    lines += [
        f"""
mpirun -of-proc .../mpi {{CP}} ${{LOCAL_PYTORCH_TGZ}} /local/
mpirun -of-proc .../mpi tar -I pigz -xf /local/$(basename ${{LOCAL_PYTORCH_TGZ}}) -C /local
source "/local/venv/bin/activate"

# Run langmo
MPIEXEC_ARGS=(
   -of-proc .../mpi
   -x NUM_GPUS_PER_NODE=0
   -x TOKENIZERS_PARALLELISM=true
   -x PROTONN_DISTRIBUTED_BACKEND=MPI
   -x WANDB_MODE="disabled"
   -x OMP_NUM_THREADS=48
   -x LD_PRELOAD=/local/opt/lib/libtcmalloc.so
)
mpirun ${{MPIEXEC_ARGS[@]}} python3 -m {args.module} {result_path}/config.yaml
"""
    ]
    body = "\n".join(lines)
    script_path.write_text(body)


def main(args):
    print(args.config)
    config_data = get_config_data(args.config)
    config_data = make_config_yaml(config_data)
    make_pjsub_script(args, config_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("module", type=str)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    main(args)
