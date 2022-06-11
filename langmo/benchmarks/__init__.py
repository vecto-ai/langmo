import os
import stat

import yaml
from langmo.config import load_yaml_or_empty

# TODO: make console logs go to the target dir


def schedule_eval_run(path):
    # TODO: support capping max runs
    # TODO: add some warnings/ howto later
    config = load_yaml_or_empty("./configs/langmo.yaml")
    # cmd_submit = "qsub"
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


def create_config_file(path_snapshot, path_config):
    # TUDO: add warnings when overwriting
    user_config = load_yaml_or_empty("./configs/auto_finetune.yaml")
    user_config["model_name"] = str(path_snapshot / 'hf')
    user_config["path_results"] = str(path_snapshot / 'eval')
    user_config["suffix"] = "auto"
    with open(path_config, "w") as file_config:
        yaml.dump(user_config, file_config)


def create_job_file(path_jobscript, path_config, name_task):
    # read headear
    with open("./configs/auto_finetune.inc") as f:
        job_script_header = f.read()
    with open(path_jobscript, "w") as file_jobscript:
        file_jobscript.write(job_script_header)
        if name_task == "NLI":
            cmd = f"python3 -m langmo.benchmarks.NLI {path_config}\n"
        else:
            # TODO: check if task is in supported list to not waste time
            cmd = f"python3 -m langmo.benchmarks.GLUE {path_config} {name_task}\n"
        file_jobscript.write(cmd)
    make_executable(path_jobscript)


def create_files_and_submit(path_snapshot, name_task):
    # TODO: add unique suffix to filenames
    path_eval = path_snapshot / "eval"
    path_eval.mkdir(parents=True, exist_ok=True)
    path_config = path_eval / f"auto_{name_task}.yaml"
    path_jobscript = path_eval / f"auto_{name_task}.sh"
    create_config_file(path_snapshot, path_config)
    create_job_file(path_jobscript, path_config, name_task)
    schedule_eval_run(path_jobscript)
