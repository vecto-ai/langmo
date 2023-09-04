import sys
from pathlib import Path

from langmo import evaluate

path = Path(sys.argv[1])
for folder in path.iterdir():
    print(folder)
    if not folder.is_dir():
        continue
    evaluate.run(folder / "embs", folder / "eval")
