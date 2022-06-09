""" CLI for scheduling benchmarks """
import sys
from pathlib import PosixPath as Path

from langmo.benchmarks import create_files_and_submit


def main():
    print("scheduling benchmark runs")
    path = Path(sys.argv[1]).resolve()
    name_task = sys.argv[2]
    if (path / "checkpoints").exists():
        for subdir in (path / "checkpoints").iterdir():
            print("scheduling", subdir)
            create_files_and_submit(subdir, name_task)
    elif (path / "hf").exists():
        print("schdulign single run at", path)
        create_files_and_submit(path, name_task)
    else:
        raise RuntimeError("directory does not have checkpoints")


if __name__ == "__main__":
    main()
