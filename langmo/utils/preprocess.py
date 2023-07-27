import argparse
import json
import random
from pathlib import Path
from timeit import default_timer as timer

import humanfriendly
from kapral.corpus import Corpus
from kapral.corpus.iterators import DirIterator
from transformers import AutoTokenizer

from .json_reader import DocFromJSONFileIter  # QualityFilter


class HybridIter:
    def __init__(self, path):
        self.file_iter = DirIterator(path)
        self._gen = self.gen()

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._gen)

    def gen(self):
        for file_name in self.file_iter:
            print("processing", file_name)
            if "json" in str(file_name):
                doc_iter = DocFromJSONFileIter(file_name)
            else:
                doc_iter = Corpus(file_name).get_document_iterator()
            for doc in doc_iter:
                yield doc


def main():
    parser = argparse.ArgumentParser(description='Split texts into training sequences of token ids')
    parser.add_argument('name_tokenizer', type=str, help='tokenizer name')
    parser.add_argument('max_length', type=int, help='sequence length')
    parser.add_argument('path_src', type=str, help='corpus path')
    parser.add_argument('path_dst', type=str, help='output path')
    parser.add_argument('--max_samples', type=int, help='limit number of samples', default=-1)
    args = parser.parse_args()
    name_tokenizer = args.name_tokenizer
    max_length = args.max_length
    path_src = Path(args.path_src)
    path_dst = Path(args.path_dst) / name_tokenizer / str(max_length)
    path_dst.mkdir(parents=True, exist_ok=True)
    time_start = timer()
    tokenizer = AutoTokenizer.from_pretrained(name_tokenizer)
    line_buffer = [tokenizer.cls_token_id]
    cnt_samples_written = 0
    proba_shortening = 0.1
    allowed_underfill = 10
    min_doc_length = 5
    min_truncation_length = 5
    doc_iter = HybridIter(path_src)
    # for f in DirIterator(path):
    #     path_out = path_root / f RALTEIVE TO args.path
    # return
    with open(path_dst / "tokenized.json", "w") as f_out:
        for doc in doc_iter:
            if len(line_buffer) > min_doc_length:
                line_buffer += [tokenizer.sep_token_id]
            else:
                line_buffer = [tokenizer.cls_token_id]
            sent_iter = doc.get_sentence_iterator()
            # TODO: check if this cleans up formatting
            for sentence in sent_iter:
                tokens = tokenizer(sentence,
                                   add_special_tokens=False,
                                   return_attention_mask=False,)["input_ids"]
                line_buffer += tokens
                line_buffer += [tokenizer.sep_token_id]
                if len(line_buffer) > max_length - allowed_underfill:
                    line_buffer = line_buffer[:max_length]
                    if random.random() < proba_shortening:
                        len_shortened = random.randint(min_truncation_length, len(line_buffer))
                        line_buffer = line_buffer[: len_shortened]
                    line_buffer += [tokenizer.pad_token_id] * (max_length - len(line_buffer))
                    serialized = json.dumps(line_buffer)
                    f_out.write(serialized)
                    f_out.write("\n")
                    cnt_samples_written += 1
                    # print(tokenizer.decode(line_buffer))
                    # print()
                    if cnt_samples_written % 10000 == 0:
                        print(f"{cnt_samples_written} lines saved")
                        print(f"\tlast line len={len(line_buffer)}")
                        print(tokenizer.decode(line_buffer))
                        print()
                    line_buffer = [tokenizer.cls_token_id]
                if cnt_samples_written > args.max_samples and args.max_samples > 0:
                    break
            if cnt_samples_written > args.max_samples and args.max_samples > 0:
                break
    time_elapsed = timer() - time_start
    print("Done in", humanfriendly.format_timespan(time_elapsed))


if __name__ == "__main__":
    main()

# TODO: add shuffling as a preprocessing step
