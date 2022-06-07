import shutil
import sys
from pathlib import Path


def main():
    print("copying")
    path_src = Path(sys.argv[1])
    path_dst = Path(sys.argv[2])
    shutil.copy(path_src / "metadata.json", path_dst / "metadata.json")
    for checkpoint in (path_src / "checkpoints").iterdir():
        try:
            shutil.copy(checkpoint / "metadata.json",
                        path_dst / checkpoint.relative_to(path_src) / "metadata.json")
        except:
            print("warning, snapshot meta missing at", checkpoint)
        for eval_task in (checkpoint / "eval").iterdir():
            for eval_run in eval_task.iterdir():
                path_eval_relative = eval_run.relative_to(path_src)
                (path_dst / path_eval_relative).mkdir(parents=True, exist_ok=True)
                shutil.copy(eval_run / "metadata.json",
                            path_dst / path_eval_relative / "metadata.json")


if __name__ == "__main__":
    main()
