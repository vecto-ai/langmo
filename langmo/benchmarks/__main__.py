""" CLI for scheduling benchmarks """
import sys
from pathlib import PosixPath as Path

from langmo.benchmarks import create_files_and_submit


def main():
    print("scheduling benchmark runs")
    path = Path(sys.argv[1])
    name_task = sys.argv[2]
    # TODO: support detecting if we are in a single snapshot  - and not search subdirs then
    if not (path / "checkpoints").exists():
        raise RuntimeError("directory does not have checkpoints")
    path = path.resolve() / "checkpoints"
    for subdir in path.iterdir():
        print("scheduling", subdir)
        create_files_and_submit(subdir, name_task)


if __name__ == "__main__":
    main()
