import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas


def read_eval_run(eval_run_path):
    metadata = json.load(open(eval_run_path / "metadata.json", "r"))
    # @todo(vatai): use steps instead of epochs

    # max search
    max_dict = dict()
    for entry in metadata["train_logs"]:
        # epoch = entry["epoch"]
        for k, v in entry.items():
            if k == "epoch":
                continue
            # save this pair if other than value is need id
            # pair = {"value": v, "epoch": epoch}
            if k not in max_dict:
                max_dict[k] = v  # pair
            elif v > max_dict[k]:  # ["value"]:
                max_dict[k] = v  # pair
    dframe = pandas.json_normalize(max_dict)
    return dframe


def get_snapshot_stats(snapshot_path, task, metric):
    evals_path = Path(snapshot_path) / "eval" / task
    eval_dfs = []
    for eval_run_path in evals_path.glob("*"):
        dframe = read_eval_run(eval_run_path)
        eval_dfs.append(dframe[metric])
    snapshot_df = pandas.concat(eval_dfs)
    return {"mean": snapshot_df.mean(), "var": snapshot_df.var()}


def _get_epoch_step_pair(dir):
    dir = dir.name
    epoch, step = dir.split("_")
    epoch = int(epoch.split("epoch")[1])
    step = int(step.split("step")[1])
    return (epoch, step)


def get_pretrain_run_dframe(path, task, metric):
    dfs = []
    path = Path(path) / "checkpoints"
    # @todo(vatai): this might read stuff in the wrong order!!!
    sorted_paths = sorted(path.glob("*"), key=_get_epoch_step_pair)
    for ckpt_idx, snapshot_path in enumerate(sorted_paths):
        snapshot_stats = get_snapshot_stats(snapshot_path, task, metric)
        dfs.append(pandas.DataFrame(snapshot_stats, index=[ckpt_idx]))
    dframe = pandas.concat(dfs)
    return dframe


def plot_pretrain_run(path, task, metric, fileext="pdf"):
    path = Path(path)
    dframe = get_pretrain_run_dframe(path, task, metric)
    var = dframe["var"]
    mean = dframe["mean"]
    ax = dframe["mean"].plot()
    ax.fill_between(dframe.index, mean - var, mean + var, alpha=0.3)
    plt.savefig(f"{path.name}.{fileext}", bbox_inches="tight")


if __name__ == "__main__":
    assert len(sys.argv) > 1, "Args: path[, task[, metric]]"

    path = sys.argv[1]
    task = "NLI"
    metric = "val_acc_matched"

    if len(sys.argv) > 2:
        task = sys.argv[2]
    if len(sys.argv) > 3:
        metric = sys.argv[3]
    plot_pretrain_run(path, task, metric)
    plot_pretrain_run(path, task, metric, fileext="png")
