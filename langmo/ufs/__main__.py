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

    body = f"""#!/bin/bash
#PJM -g {group}
# #PJM ${{X_PARAMS[@]}}
# #PJM -N ${{JOBNAME}}
# #PJM -L "rscgrp=$(get_rscgrp ${{NODES}})"
# #PJM -L "elapse=${{ELAPSE}}"
# #PJM -L "node=${{NODES}}"
# #PJM --mpi "proc=${{NODES}}"
# #PJM -o ${{OUTDIR}}/%j.stdout
# #PJM -e ${{OUTDIR}}/%j.stderr
# #PJM --spath ${{OUTDIR}}/%j.stat
# #PJM --llio localtmp-size=40Gi
# #PJM -j -S
# #PJM $(get_emailargs)
python {module} {result_path}/config.yaml
"""
    script_path = result_path / "submit.sh"
    lines = ["#!/bin/bash"]
    lines += [f"#PJM {line}" for line in pjm_lines]
    lines += [f"python {module}\n"]
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
