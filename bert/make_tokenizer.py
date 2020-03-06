from pathlib import Path
import os
import argparse
from tokenizers import ByteLevelBPETokenizer, CharBPETokenizer, BertWordPieceTokenizer
from protonn.utils import save_data_json

# path_corpus = "/home/blackbird/Cloud/remote/berlin/data/NLP/corpora/raw_texts/Eng/BNC"
# path_corpus = "/work/data/NLP/corpora/raw_texts/Eng/BNC"
# path_corpus = "/home/blackbird/Projects_heavy/NLP/langmo/data/sample"


def main():
    parser = argparse.ArgumentParser(description='Benchmark me up, Scotty!')
    parser.add_argument("path_corpus")
    parser.add_argument("--tokenizer")
    parser.add_argument('--min_frequency', type=int, default=5)
    parser.add_argument('--vocab_size', type=int, default=30000)
    parser.add_argument('--path_out', type=str, default="out")

    args = parser.parse_args()

    path_corpus = args.path_corpus
    paths = [str(x) for x in Path(path_corpus).glob("**/*.txt")]

    tokenizers = {"byteBPE": ByteLevelBPETokenizer,
                  "charBPE": CharBPETokenizer,
                  "BertWP": BertWordPieceTokenizer}

    tokenizer = tokenizers[args.tokenizer]()

    # Customize training
    tokenizer.train(files=paths,
                    vocab_size=args.vocab_size,
                    min_frequency=args.min_frequency,
                    special_tokens=["<s>",
                                    "<pad>",
                                    "</s>",
                                    "<unk>",
                                    "<mask>"])

    path_out = Path(args.path_out)
    path_out = path_out / f"{args.tokenizer}_vs_{args.vocab_size}_m{args.min_frequency}"
    os.makedirs(path_out, exist_ok=True)
    tokenizer.save(str(path_out), "t")
    r = tokenizer.encode("Hi, how do you do NoNeXistingTkn?")
    print(r.tokens)
    metadata = {}
    metadata.update(vars(args))
    save_data_json(metadata, path_out / "metadata.json")

if __name__ == "__main__":
    main()
