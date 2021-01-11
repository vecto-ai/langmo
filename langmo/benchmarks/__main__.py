""" CLI for scheduling benchmarks """
import sys
from pathlib import PosixPath as Path
from langmo.benchmarks import create_job_files


def schedule_single_snapshot(path):
    print("scheduling", path)
    create_job_files(path)


def main():
    print("scheduling benchmark runs")
    path = Path(sys.argv[1])
    # TODO: support detecting if we are in a single snapshot  - and not search subdirs then
    if not (path / "checkpoints").exists():
        raise RuntimeError("directory does not have checkpoints")
    path = path.resolve() / "checkpoints"
    for subdir in path.iterdir():
        schedule_single_snapshot(subdir)


if __name__ == "__main__":
    main()
