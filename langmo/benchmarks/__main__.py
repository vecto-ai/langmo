""" CLI for scheduling benchmarks """
import argparse
from pathlib import PosixPath as Path

from langmo.benchmarks import create_files_and_submit


def main():
    print("scheduling benchmark runs")
    parser = argparse.ArgumentParser(description="Automatically schedule benchmarks")
    parser.add_argument("model", type=str, help="model name or path")
    parser.add_argument("task", type=str, help="task name")
    parser.add_argument("--config", type=str, help="use custom config", default="./configs/auto_finetune.yaml")
    parser.add_argument("--out-dir", type=str, help="force output dir")
    parser.add_argument("--tempfiles-dir", type=str, help="force output dir")
    args = parser.parse_args()
    path = Path(args.model)
    name_task = args.task
    if args.tempfiles_dir:
        path_tempfiles = Path(args.tempfiles_dir)
        path_tempfiles.mkdir(exist_ok=True, parents=True)
    else:
        path_tempfiles = None
    if (path / "checkpoints").exists():
        for subdir in (path / "checkpoints").iterdir():
            print("scheduling", subdir)
            create_files_and_submit(subdir, name_task, args.config, args.out_dir, path_tempfiles)
    else:
        create_files_and_submit(path, name_task, args.config, args.out_dir, path_tempfiles)


if __name__ == "__main__":
    main()
