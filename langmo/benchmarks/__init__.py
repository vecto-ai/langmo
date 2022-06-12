import os
import stat
import tempfile

import yaml

from langmo.config import GLUETASKTOKEYS, load_yaml_or_empty

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


def get_results_path(user_config, path_snapshot, name_task):
    bs = get_value_or_placeholder(user_config, "batch_size")
    seed = get_value_or_placeholder(user_config, "seed")
    random = tempfile.NamedTemporaryFile().name.split("/")[-1]

    path = path_snapshot / f"eval/{name_task}/{bs}_{seed}_{random}"
    print(f"PATH: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def create_config_file(user_config, path_snapshot, path_config, path_out):
    # TUDO: add warnings when overwriting
    user_config["model_name"] = str(path_snapshot / "hf")
    user_config["path_results"] = str(path_out)
    user_config["suffix"] = "auto"
    user_config["create_unique_path"] = False
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
            available_glue_tasks = list(GLUETASKTOKEYS.keys())
            if not name_task in available_glue_tasks:
                raise Exception(
                    f"{name_task} is not supported. One among {'|'.join(available_glue_tasks)} should be chosen."
                )
            cmd = f"python3 -m langmo.benchmarks.GLUE {path_config} {name_task}\n"
        file_jobscript.write(cmd)
    make_executable(path_jobscript)


def create_files_and_submit(path_snapshot, name_task):
    user_config = load_yaml_or_empty("./configs/auto_finetune.yaml")
    path_out = get_results_path(user_config, path_snapshot, name_task)

    path_config = path_out / f"auto_{name_task}.yaml"
    create_config_file(user_config, path_snapshot, path_config, path_out)

    path_jobscript = path_out / f"auto_{name_task}.sh"
    create_job_file(path_jobscript, path_config, name_task)
    schedule_eval_run(path_jobscript)
