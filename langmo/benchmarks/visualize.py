#!/usr/bin/env python

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm


def gen_pretrain_paths(path):
    for ckpt in Path(path).joinpath("checkpoints").glob("*"):
        yield ckpt


def gen_finetune_paths(path, task):
    for ckpt in gen_pretrain_paths(path):
        eval_runs = ckpt.joinpath("eval").joinpath(task).glob("*")
        for eval_run in eval_runs:
            metadata_path = eval_run / "metadata.json"
            if metadata_path.exists():
                yield metadata_path


def get_pretrain_samples_per_batch(path):
    pretrain_paths = map(
        lambda path: path / "metadata.json",
        gen_pretrain_paths(path),
    )
    df = pd.DataFrame([pd.read_json(path, typ="series") for path in pretrain_paths])
    df = df.join(pd.json_normalize(df["corpus"]))
    df = df.drop(["corpus"], axis=1)
    df = df.drop_duplicates()
    return df["batch_size"][0] * df["cnt_workers"][0]


def get_all_finetunes_as_df(path, task):
    samples_per_batch = get_pretrain_samples_per_batch(path)

    df = pd.concat(
        [pd.read_json(path) for path in gen_finetune_paths(path, task)],
        ignore_index=True,
    )
    df = df.join(pd.json_normalize(df["train_logs"]))
    df = df.drop(["train_logs"], axis=1)
    df["step"] = df["path_results"].str.extract(r".*step(\d*)/.*")
    df["step"] = df["step"].astype(int)
    df["cnt_samples"] = df["step"] * samples_per_batch
    return df


def plot_colormesh(df, metric, aggfunc, filename):
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

    df = get_all_finetunes_as_df(args.path, args.task)

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
