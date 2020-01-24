from pathlib import Path
from langmo import evaluate


path = Path("/home/blackbird/Cloud/remote/berlin/alex/data/DL_outs/NLP/embed_proto1/20.01.17_13.52.10_kiev1.m.gsic.titech.ac.jp")
for folder in path.iterdir():
    print(folder)
    if not folder.is_dir():
        continue
    evaluate.run(folder / "embs", folder / "eval" / "analogy")
