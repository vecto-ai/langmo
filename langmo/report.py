import argparse
from collections import defaultdict
from pathlib import PosixPath as Path

import numpy as np
from protonn.utils import load_json
from tqdm import tqdm

metric_names = ["val_acc_matched", "val_acc_hans"]


def def_value():
    return {name: [] for name in metric_names}


def main():
    parser = argparse.ArgumentParser(description='generate report from snapshots')
    parser.add_argument('path', type=str, help='path to the root folder')
    args = parser.parse_args()
    path = Path(args.path)
    experiments = [x for x in path.iterdir() if x.is_dir()]
    all_stats = defaultdict(def_value)
    for experiment in tqdm(experiments):
        # print(experiment)
        if not (experiment / "metadata.json").is_file():
            continue
        data = load_json(experiment / "metadata.json")
        model_name = data["model_name"]
        if model_name[0] == "/":
            continue
        model_name = model_name.split("/")[-1]
        if data["randomize"]:
            continue
        # TODO: make this a param
        if "siamese" in data:
            if data["siamese"]:
                model_name += "_siam"
                if "encoder_wrapper" in data:
                    model_name += "_" + data["encoder_wrapper"]
                else:
                    print("no wrapper in", experiment)
                if data["freeze_encoder"]:
                    model_name += "_frozen"
        logs = data["train_logs"]
        min_epochs = 4
        if len(logs) < min_epochs:
            continue
        for name in metric_names:
            metric = max([x[name] for x in logs])
            all_stats[model_name][name].append(metric)
            # print(name, metric)

    for model_name in sorted(all_stats):
        metric_all_runs = all_stats[model_name]
        print()
        print(model_name + "   -------------------")
        for name in metric_names:
            print(name)
            print("\truns\tmean\tmax\tstd")
            cnt_runs = len(metric_all_runs[name])
            pct_mean = np.mean(metric_all_runs[name]) * 100
            pct_max = np.max(metric_all_runs[name]) * 100
            pct_std = np.std(metric_all_runs[name]) * 100
            print(f"\t{cnt_runs}\t{pct_mean:.2f}\t{pct_max:.2f}\t{pct_std:.2f}")


if __name__ == "__main__":
    main()
