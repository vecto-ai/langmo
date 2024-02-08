import os
import stat
import tempfile
from pathlib import Path

import yaml

from langmo.config.base import load_yaml_or_empty
from langmo.training.glue.config import GLUE_TASKS
from langmo.training.qa.config import QATASKS

# TODO: make console logs go to the target dir


def schedule_eval_run(path):
    # TODO: support capping max runs
    # TODO: add some warnings/ howto later
    config = load_yaml_or_empty("./configs/langmo.yaml")
    num_runs = 1
    for _ in range(num_runs):
        if "submit_command" in config:
            cmd_submit = config["submit_command"]
            cmd = f"{cmd_submit} {path}"
        else:
            cmd = path
        print(f"@@@ scheduling: {cmd}")
        os.system(cmd)


def make_executable(path):
    st = os.stat(path)
    os.chmod(path, st.st_mode | stat.S_IEXEC)


def get_value_or_placeholder(config, key):
    value = f"no_{key}"
    if key in config:
        value = str(config[key])
    else:
        pass  # warning
    return value


def get_results_path(user_config, path_snapshot, name_task, salt):
    bs = get_value_or_placeholder(user_config, "batch_size")
    seed = get_value_or_placeholder(user_config, "seed")
    path = path_snapshot / f"eval/{name_task}/lbs{bs}_s{seed}_{salt}"
    print(f"PATH: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def create_config_file(path_config, user_config, path_snapshot):
    # TUDO: add warnings when overwriting
    if path_snapshot.exists():
        user_config["model_name"] = str(path_snapshot / "hf")
    else:
        user_config["model_name"] = str(path_snapshot)
    with open(path_config, "w") as file_config:
        yaml.dump(user_config, file_config)


def create_job_file(path_jobscript, path_config, name_task):
    # read headear
    with open("./configs/auto_finetune.inc") as f:
        job_script_header = f.read()
    available_glue_tasks = list(GLUE_TASKS.keys())
    if name_task == "NLI":
        cmd = f"python3 -m langmo.benchmarks.NLI {path_config}\n"
    elif name_task in available_glue_tasks:
        cmd = f"python3 -m langmo.benchmarks.GLUE {path_config} {name_task}\n"
    elif name_task in QATASKS:
        cmd = f"python3 -m langmo.benchmarks.QA {path_config} {name_task}\n"
    else:
        raise ValueError(
            f"{name_task} is not supported. One among {'|'.join(available_glue_tasks)} or {'|'.join(QATASKS)} should be chosen."
        )
    with open(path_jobscript, "w") as file_jobscript:
        file_jobscript.write(job_script_header)
        file_jobscript.write(cmd)
    make_executable(path_jobscript)


def create_files_and_submit(path_snapshot, name_task, config, out_dir, path_tempfiles):
    # TODO: don't forget to resolve the path
    user_config = load_yaml_or_empty(config)
    salt = tempfile.NamedTemporaryFile().name.split("/")[-1][3:]
    if out_dir is None:
        path_out = get_results_path(user_config, path_snapshot, name_task, salt)
        user_config["create_unique_path"] = False
    else:
        path_out = Path(out_dir)
        user_config["create_unique_path"] = True
    user_config["path_results"] = str(path_out)
    path_out.mkdir(parents=True, exist_ok=True)
    if path_tempfiles:
        path_config = path_tempfiles / f"auto_{name_task}_{salt}.yaml"
        path_jobscript = path_tempfiles / f"auto_{name_task}_{salt}.sh"
    else:
        path_config = path_out / f"auto_{name_task}_{salt}.yaml"
        path_jobscript = path_out / f"auto_{name_task}_{salt}.sh"
    create_config_file(path_config, user_config, path_snapshot)
    create_job_file(path_jobscript, path_config, name_task)
    schedule_eval_run(path_jobscript)
