#!/usr/bin/env python

import argparse

# import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from protonn.utils import load_json


def gen_checkpoint_paths(path):
    for ckpt in Path(path).joinpath("checkpoints").glob("*"):
        yield ckpt


def read_finetunes(path, name_metric="val_acc_matched"):
    task_runs = path.glob("**/metadata.json")
    for path_run in task_runs:
        data_run = load_json(path_run)
        # print(data_run["train_logs"])
        keys = ["epoch", name_metric]
        # logs_clean = [{key: d[key] for key in keys} for d in data_run["train_logs"]][1:]
        # logs_clean = [{d["epoch"]: d[name_metric]} for d in data_run["train_logs"][1:]]
        logs_clean = [d[name_metric] for d in data_run["train_logs"][1:]]
        # print(logs_clean)
        df = pd.Series(logs_clean)
        # df.set_index("epoch", inplace=True)
        # TODO: returning just the first one for the time being
        # print(df.T)
        return df
        # [0][name_metric]


# def get_pretrain_samples_per_batch(path):
#     pretrain_paths = map(
#         lambda path: path / "metadata.json",
#         gen_pretrain_paths(path),
#     )
#     df = pd.DataFrame([pd.read_json(path, typ="series") for path in pretrain_paths])
#     df = df.join(pd.json_normalize(df["corpus"]))
#     df = df.drop(["corpus"], axis=1)
#     df = df.drop_duplicates()
#     return df["batch_size"][0] * df["cnt_workers"][0]


def read_checkpoints(path, task):
    data = []
    index = []
    for path_checkpoint in gen_checkpoint_paths(path):
        print("reading", path_checkpoint)
        meta_checkpoint = load_json(path_checkpoint / "metadata.json")
        # print(meta_checkpoint["cnt_samples_processed"])
        logs_finetune = read_finetunes(path_checkpoint / "eval" / task)
        index.append(int(meta_checkpoint["cnt_samples_processed"]))
        data.append(logs_finetune)
        # data_checkpoiunt = {
        #     "cnt_samples_processed": int(meta_checkpoint["cnt_samples_processed"]),
        #     "acc": logs_finetune#["val_acc_matched"]
        # }
        # data.append(data_checkpoiunt)
    # paths = list(gen_finetune_paths(path, task))
    # for path_metadata in paths:
    #    with open(path_metadata) as f:
    #        data = json.load(f)
    #        print(data["train_logs"])
    df = pd.DataFrame(data, index=index)
    # df.set_index("cnt_samples_processed", inplace=True)
    df.sort_index(inplace=True)
    return df
    # samples_per_batch = get_pretrain_samples_per_batch(path)

    df = pd.concat(
        [pd.read_json(path) for path in paths],
        ignore_index=True,
    )
    print(df)
    df = df.join(pd.json_normalize(df["train_logs"]))
    df = df.drop(["train_logs"], axis=1)
    df["step"] = df["path_results"].str.extract(r".*step(\d*)/.*")
    df["step"] = df["step"].astype(int)
    # df["cnt_samples"] = df["step"] * samples_per_batch
    return df


def plot_colormesh(df, metric, aggfunc, filename):
    print(df)
    return
    df = pd.pivot_table(
        data=df,
        values=metric,
        index=["epoch", "cnt_samples"],
        aggfunc=aggfunc,
    ).unstack(1)

    fig, ax = plt.subplots()

    X, Y = np.meshgrid(list(map(lambda t: t[1] / 1000000, df.columns)), df.index)
    surf = ax.pcolormesh(X, Y, df, cmap=cm.RdBu, shading="auto")
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.title(f"{aggfunc} of {metric}")
    ax.set_xlabel("Pretraining cnt_samples")
    ax.set_ylabel("Finetuning epochs")
    ax.get_xaxis().set_major_formatter("{x}M")
    if filename:
        plt.savefig(filename, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def plot_2d(df, task, metric, aggfunc, filename):
    epoch_zero = df[0]
    epoch_max = df.max(axis=1)
    plt.fill_between(df.index / 1000000, epoch_zero, epoch_max, alpha=0.9)
    # df[0].plot()
    plt.ylabel("accuracy")
    plt.xlabel("samples processed, million")
    plt.savefig("plot.pdf")
    # exit(0)
    return
    df = pd.pivot_table(
        data=df,
        values=metric,
        index=["path_results", "cnt_samples"],
        aggfunc=aggfunc,
    )
    mean = df[metric].groupby("cnt_samples").mean()
    var = df[metric].groupby("cnt_samples").var().fillna(0)

    ax = mean.plot()
    ax.fill_between(mean.index, mean - var, mean + var, alpha=0.3)

    plt.title(f"mean and var (as area) over different runs of {task}")
    ax.set_xlabel("Num samples")
    ax.set_ylabel(f"{aggfunc} {metric} achieved in a run")
    # ax.get_xaxis().set_major_formatter(lambda t: f"{t/(10**6)}M")
    if filename:
        plt.savefig(filename, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def get_aggfunc(args):
    if args.aggfunc:
        return args.aggfunc
    if "loss" in args.metric:
        return "min"
    elif "acc" in args.metric:
        return "max"
    else:
        raise ValueError(f"Couldn't guess aggfunc from metric {metric}")


def get_plot_paths(args):
    filename = args.filename if args.filename else args.path
    ext = f".{args.ext}"
    colormesh_path = filename
    plot2d_path = filename
    if args.plot2d and args.colormesh:
        colormesh_path += "_colormesh"
        plot2d_path += "_plot2d"
    colormesh_path += ext
    plot2d_path += ext
    return colormesh_path, plot2d_path


def main(args):
    colormesh_path, plot2d_path = get_plot_paths(args)
    aggfunc = get_aggfunc(args)

    df = read_checkpoints(args.path, args.task)

    if args.colormesh:
        plot_colormesh(df, args.metric, aggfunc, colormesh_path)
        print(f"Generated {colormesh_path} (metric: {args.metric}, aggfunc: {aggfunc})")

    if args.plot2d:
        plot_2d(df, args.task, args.metric, aggfunc, plot2d_path)
        print(f"Generated {plot2d_path} (metric: {args.metric}, aggfunc: {aggfunc})")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        help="Path to the pretrain logs",
    )
    parser.add_argument(
        "--task",
        default="NLI",
        help="Finetuning task",
    )
    parser.add_argument(
        "--metric",
        default="val_acc_matched",
        help="Metric of the finetuning task",
    )
    parser.add_argument(
        "--aggfunc",
        help="Aggregation function (guesses 'min' for loss, 'max' for accuracy by default)",
    )
    parser.add_argument(
        "--filename",
        help="Filename of output plot (same as PATH by default)",
    )
    parser.add_argument(
        "--ext",
        default="png",
        help="File extension",
    )
    parser.add_argument(
        "--plot2d",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate 2D plot",
    )
    parser.add_argument(
        "--colormesh",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate colormesh plot",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
