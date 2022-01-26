import json
import random
import sys
from pathlib import Path

from kapral.corpus import Corpus
from transformers import AutoTokenizer


def main():
    # TODO: use argparse
    if len(sys.argv) < 4:
        print(f"usage: {sys.argv[0]} tokenizer max_length path_corpus path_output_base")
        print(f"output directories corresponding to tokemnizer and sequence length will be created automatically")
        return -1
    name_tokenizer = sys.argv[1]
    max_length = int(sys.argv[2])
    path = sys.argv[3]
    path_out = Path(sys.argv[4]) / name_tokenizer / str(max_length)
    path_out.mkdir(parents=True, exist_ok=True)
    path_out = path_out / "tokenized.jsonl"

    tokenizer = AutoTokenizer.from_pretrained(name_tokenizer)
    corpus = Corpus(path)
    corpus.load_dir_strucute()
    sent_iter = corpus.get_sentence_iterator()
    line_buffer = [tokenizer.cls_token_id]
    cnt_lines_written = 0
    proba_shortening = 0.1
    allowed_underfill = 10
    with open(path_out, "w") as f_out:
        for line in sent_iter:
            tokens = tokenizer(line,
                               add_special_tokens=False,
                               return_attention_mask=False,)["input_ids"]
            line_buffer += tokens
            if len(line_buffer) > max_length - allowed_underfill:
                line_buffer = line_buffer[:max_length]
                min_length = 5
                if random.random() < proba_shortening:
                    line_buffer = line_buffer[: random.randint(min_length, len(line_buffer))]
                line_buffer += [tokenizer.pad_token_id] * (max_length - len(line_buffer))
                serialized = json.dumps(line_buffer)
                f_out.write(serialized)
                f_out.write("\n")
                cnt_lines_written += 1
                if cnt_lines_written % 10000 == 0:
                    print(f"{cnt_lines_written} lines saved, last line len={len(line_buffer)}")
                    print(tokenizer.decode(line_buffer))
                line_buffer = [tokenizer.cls_token_id]
    print("Done!")

if __name__ == "__main__":
    main()

# TODO: add shuffling as a preprocessing step
