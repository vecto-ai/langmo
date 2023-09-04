import random
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 3:
        print("usage: shuffle.py path_in path_out")
        return -1
    print("this is in-memory shuffler, be sure you have enough RAM")
    path_in = Path(sys.argv[1])
    path_out = Path(sys.argv[2])
    path_out.parent.absolute().mkdir(parents=True, exist_ok=True)
    lines = []
    for p in path_in.rglob("*.jsonl"):
        print(p.resolve())
        with open(p) as f:
            for line in f:
                lines.append(line)

    print("total lines:", len(lines))
    random.shuffle(lines)
    print("shuffled")
    with open(path_out, "w") as f:
        for line in lines:
            f.write(line)
            # f.write("\n")


if __name__ == "__main__":
    main()
