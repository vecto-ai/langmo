import os
import stat

# import sys

# TODO: allow user to provide paltform-specific headers
# TODO: this will be probably done through protonn
# TODO: make console logs go to the target dir
header_ABCI = (
    "#!/bin/bash\n"
    "#$ -cwd\n"
    "#$ -l rt_AF=1\n"
    "#$ -l h_rt=00:20:00\n"
    "#$ -N auto-finetune\n"
    "#$ -j y\n"
    "#$ -o ./logs/$JOB_NAME.o$JOB_ID\n\n"
    "source /etc/profile.d/modules.sh\n"
    "source /home/aca10027xu/projects/langmo/modules.sh\n\n"
    "NUM_NODES=${NHOSTS}\n"
    "export NUM_GPUS_PER_NODE=8\n"
    "export WANDB_MODE=offline\n"
    "export TOKENIZERS_PARALLELISM=true\n"
    "export PL_TORCH_DISTRIBUTED_BACKEND=NCCL\n"
    "NUM_GPUS_PER_SOCKET=$(expr ${NUM_GPUS_PER_NODE} / 2)\n"
    "NUM_PROCS=$(expr ${NUM_NODES} \\* ${NUM_GPUS_PER_NODE})\n"
    "MPIOPTS=\"-np ${NUM_PROCS} -map-by ppr:${NUM_GPUS_PER_NODE}:node -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_include bond0\"\n"
)


def schedule_eval_run(path):
    # TODO: support capping max runs
    num_runs = 1
    for _ in range(num_runs):
        # TODO: move group name to header of jobfile
        cmd_submit = "qsub -g gcb50300"
        cmd = f"{cmd_submit} {path}"
        print(f"@@@ scheduling: {cmd}")
        os.system(cmd)


def make_executable(path):
    st = os.stat(path)
    os.chmod(path, st.st_mode | stat.S_IEXEC)


def create_config_file(path, path_config):
    with open(path_config, "w") as file_config:
        file_config.write(f"model_name: {path / 'hf'}\n")
        file_config.write(f"path_results: {path / 'eval'}\n")
        file_config.write(f"suffix: auto\n")
        file_config.write(f"cnt_epochs: 8\n")
        file_config.write(f"batch_size: 128\n")
        file_config.write(f"per_epoch_snapshot: false\n")


def create_files_and_submit(path, name_task):
    path_config = path / "evalconfig.yaml"
    path_jobscript = path / "evaljobscript.sh"
    create_config_file(path, path_config)
    with open(path_jobscript, "w") as file_jobscript:
        # TODO: get platform-specific headers from config
        # e.g. for ABCI the nodes, the groups etc
        # (nonTODO:) expect langmo to be installed by user
        file_jobscript.write(header_ABCI)
        file_jobscript.write("mpirun ${MPIOPTS} \\\n\t")
        file_jobscript.write("-x TOKENIZERS_PARALLELISM \\\n\t")
        file_jobscript.write("-x NUM_GPUS_PER_NODE \\\n\t")
        file_jobscript.write("-x PL_TORCH_DISTRIBUTED_BACKEND \\\n\t")
        file_jobscript.write("-x WANDB_MODE \\\n\t")
        # TODO: this failes if python on login node is different
        # cmd = f"{sys.executable} -m langmo.benchmarks.NLI {path_config}\n"
        if name_task == "NLI":
            cmd = f"python3 -m langmo.benchmarks.NLI {path_config}\n"
        else:
            cmd = f"python3 -m langmo.benchmarks.GLUE {path_config} {name_task}\n"
        file_jobscript.write(cmd)
    make_executable(path_jobscript)
    schedule_eval_run(path_jobscript)
