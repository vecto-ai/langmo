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
    result_path = Path(config_data["path_result"]) / "decorated"
    result_path.mkdir(exist_ok=True)
    config_data["path_result"] = str(result_path)
    config_data["create_unique_path"] = False
    with open(result_path / "config.yaml", "w") as f:
        yaml.dump(config_data, f)
    return config_data


def get_rscgrp(nodes):
    return "small" if nodes < 384 else "large"


def get_pjm_lines(job_name, result_path):
    gid = os.stat(result_path).st_gid
    group = grp.getgrgid(gid)[0]
    pjm_lines = [
        f"-g {group}",
        f"-x PJM_LLIO_GFSCACHE=/vol0004",
        f"-N {job_name}",
        f"-L rscgrp={get_rscgrp(args.nodes)}",
        f"-L elapse={args.elapse}",
        f"-L node={args.nodes}",
        f"--mpi proc={args.nodes}",
        f"-o {result_path}/%j.stdout",
        f"-e {result_path}/%j.stderr",
        f"--spath {result_path}/%j.stat",
        f"--llio localtmp-size=40Gi",
        f"-j -S",
    ]
    try:
        email = git.config.GitConfigParser().get_value("user", "email")
        pjm_lines.append(f"-m b,e --mail-list {email}")
    except Exception as e:
        print(f"Warning: git email not set")
    return pjm_lines


def make_pjsub_script(args, langmo_args, config_data):
    result_path = Path(config_data["path_result"])
    script_path = result_path / "submit.sh"
    job_name = f"{args.config.with_suffix('')}-{args.nodes}"
    pjm_lines = get_pjm_lines(job_name, result_path)

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
mpirun ${{MPIEXEC_ARGS[@]}} python3 -m {' '.join(langmo_args)} {result_path}/config.yaml
"""
    ]
    body = "\n".join(lines)
    script_path.write_text(body)
    cmd = f"pjsub {script_path}"
    if args.dry_run:
        print(cmd)
    else:
        os.system(cmd)


def main(args, langmo_args):
    print(args.config)
    config_data = get_config_data(args.config)
    config_data = make_config_yaml(config_data)
    make_pjsub_script(args, langmo_args, config_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("-n", "--nodes", type=int, default=128)
    parser.add_argument("-e", "--elapse", type=str, default="12:00:00")
    parser.add_argument("--dry-run", action="store_true")
    args, langmo_args = parser.parse_known_args()
    print(f"args={args}")
    print(f"langmo_args={langmo_args}")

    main(args, langmo_args)
