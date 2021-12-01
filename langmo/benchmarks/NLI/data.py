# import numpy as np

import datasets
import torch
from protonn.distributed import dist_adapter as da

from langmo.benchmarks.base_data import BaseCollator, BaseDataModule

dic_heuristics = {"lexical_overlap": 0, "constituent": 1, "subsequence": 2}

labels_heuristics = ["lexical_overlap", "constituent", "subsequence"]
labels_entail = ["entail", "nonentail"]


class Collator(BaseCollator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        empty_tokenized = self.tokenizer(text=[""], text_pair=[""])
        self.has_token_type_ids = "token_type_ids" in empty_tokenized
        cls_sep_end = empty_tokenized["input_ids"][0]
        self.cls, self.mid_seps, self.end_sep = (
            cls_sep_end[:1],
            cls_sep_end[1:-1],
            cls_sep_end[-1:],
        )
        if "sep_cnt" in self.params:
            self.mid_seps = self.params["sep_cnt"] * [self.tokenizer.sep_token_id]

    def truncate(self, seq1, seq2, target_len):
        len1 = len(seq1)
        len2 = len(seq2)
        if self.tokenizer_params["truncation"] and len1 + len2 > target_len:
            new_len1 = 0
            new_len2 = 0
            total_len = 0
            while True:
                if new_len1 < len1:
                    new_len1 += 1
                    total_len += 1
                if total_len == target_len:
                    break
                if new_len2 < len2:
                    new_len2 += 1
                    total_len += 1
                if total_len == target_len:
                    break
            seq1 = seq1[:new_len1]
            seq2 = seq2[:new_len2]
        return seq1, seq2

    def padding(self, results):
        params = self.tokenizer_params
        if "padding" not in params:
            return
        padding = params["padding"]
        if not padding:
            return
        if padding == "max_length":
            target_length = params["max_length"]
        elif padding == "longest":
            target_length = max(map(len, results["input_ids"]))
        for key, data in results.items():
            for ri, row in enumerate(data):
                missing = target_length - len(row)
                pad = self.tokenizer.pad_token_id if key == "input_ids" else 0
                data[ri] += [pad] * missing

    def tokenize(self, text, text_pair):
        sent1 = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        sent2 = self.tokenizer(text_pair, add_special_tokens=False)["input_ids"]
        extra_len = len(self.cls) + len(self.mid_seps) + len(self.end_sep)
        max_length = self.tokenizer_params["max_length"]
        target_len = max_length - extra_len
        results = {"input_ids": [], "attention_mask": []}
        if self.has_token_type_ids:
            results["token_type_ids"] = []
        assert len(sent1) == len(sent2)
        for seq1, seq2 in zip(sent1, sent2):
            seq1, seq2 = self.truncate(seq1, seq2, target_len)
            input_ids = self.cls + seq1 + self.mid_seps + seq2 + self.end_sep
            attention_mask = [1] * len(input_ids)
            if self.has_token_type_ids:
                len1 = len(self.cls) + len(seq1) + len(self.mid_seps)
                len2 = len(seq2) + len(self.end_sep)
                token_type_ids = [0] * len1 + [1] * len2
            results["input_ids"].append(input_ids)
            results["attention_mask"].append(attention_mask)
            if self.has_token_type_ids:
                results["token_type_ids"].append(token_type_ids)
        self.padding(results)
        return {k: torch.LongTensor(v) for k, v in results.items()}

    def __call__(self, x):
        sent1 = [i["premise"] for i in x]
        sent2 = [i["hypothesis"] for i in x]
        labels = [i["label"] for i in x]
        labels = torch.LongTensor(labels)
        if "heuristic" in x[0]:
            heuristic = [dic_heuristics[i["heuristic"]] for i in x]
            heuristic = torch.LongTensor(heuristic)
        else:
            heuristic = None
        if not self.params["siamese"]:
            features = self.tokenize(text=sent1, text_pair=sent2)
        else:
            sent1 = self.tokenizer(text=sent1, **self.tokenizer_params)
            sent2 = self.tokenizer(text=sent2, **self.tokenizer_params)
            features = {"left": sent1, "right": sent2}
        # TODO: get max len from both parts
        return (features, labels, heuristic)


class NLIDataModule(BaseDataModule):
    def __init__(self, tokenizer, params):
        super().__init__(tokenizer, params)
        self.collator = Collator(self.tokenizer, params)

    def setup(self, stage=None):
        # TODO: get the right data loaded (might be something like "to
        # local SSD")

        self.cnt_train_samples = 0
        if da.rank() == 0:
            # TODO: can we download without loading
            ds_hans = datasets.load_dataset("hans")
            print("preload hans", ds_hans)
            ds = datasets.load_dataset("multi_nli")
            self.cnt_train_samples = len(ds["train"])

        num_samples_tensor = torch.LongTensor([self.cnt_train_samples])
        self.cnt_train_samples = da.broadcast(num_samples_tensor, 0).item()

    def train_dataloader(self):
        return [self.get_split_dataloader("multi_nli", "train")]

    def val_dataloader(self):
        dataloaders = [
            self.get_split_dataloader("multi_nli", "validation_matched"),
            self.get_split_dataloader("multi_nli", "validation_mismatched"),
            self.get_split_dataloader("hans", "validation"),
        ]
        return dataloaders


# ---- code to reuse when we pad manually-----------

# def zero_pad_item(sample, max_len):
#     if sample.shape[0] > max_len:
#         return sample[:max_len]
#     else:
#         size_pad = max_len - sample.shape[0]
#         # print("!!!!", sample)
#         assert size_pad >= 0
#         res = np.hstack([np.zeros(size_pad, dtype=np.int64), sample])
#         return res


# def sequences_to_padded_tensor(seqs, max_len):
#     seqs = list(seqs)
#     # max_len = max([len(s) for s in seqs])
#     # print(max_len)
#     # if max_len > 128:
#     #    max_len = 128
#     # print(seqs)
#     padded = [zero_pad_item(s, max_len) for s in seqs]
#     padded = np.array(padded, dtype=np.int64)
#     # LSTM-specific
#     # padded = np.rollaxis(padded, 1, 0)
#     padded = torch.from_numpy(padded)
#     return padded

# class MyDataLoader:
#     def __init__(self, sent1, sent2, labels, batch_size):
#         # optinally sort
#         if da.rank() != 0:
#             tr_logging.set_verbosity_error()
#         tuples = list(zip(sent1, sent2, labels))
#         cnt_batches = len(tuples) / batch_size
#         batches = np.array_split(tuples, cnt_batches)
#         self.batches = [self.zero_pad_batch(b) for b in batches]

#     def zero_pad_batch(self, batch):
#         sent1, sent2, labels = zip(*batch)
#         max_len_sent1 = max(len(s) for s in sent1)
#         max_len_sent2 = max(len(s) for s in sent2)
#         max_len = max(max_len_sent1, max_len_sent2)
#         sent1 = sequences_to_padded_tensor(sent1, max_len)
#         sent2 = sequences_to_padded_tensor(sent2, max_len)
#         labels = torch.LongTensor(labels)
#         return ((sent1, sent2), labels)

#     def __len__(self):
#         return len(self.batches)

#     def __getitem__(self, idx):
#         return self.batches[idx]


# def ds_to_tensors(dataset, tokenizer, batch_size, test, params):
#     sent1 = [i["premise"] for i in dataset]
#     sent2 = [i["hypothesis"] for i in dataset]
#     labels = [i["label"] for i in dataset]
#     if test:
#         # TODO: use bs and da size
#         cnt_testrun_samples = batch_size * 2
#         sent1 = sent1[:cnt_testrun_samples]
#         sent2 = sent2[:cnt_testrun_samples]
#         labels = labels[:cnt_testrun_samples]
#     if params["uncase"]:
#         sent1 = [i.lower() for i in sent1]
#         sent2 = [i.lower() for i in sent2]
#     labels = torch.LongTensor(labels)
#     # texts_or_text_pairs = list(zip(sent1, sent2))
#     features = tokenizer(
#         text=sent1,
#         text_pair=sent2,
#         max_length=128,
#         padding="max_length",
#         truncation=True,
#         return_tensors="pt",
#         return_token_type_ids=True,
#     )
#     ids = torch.split(features["input_ids"], batch_size)
#     masks = torch.split(features["attention_mask"], batch_size)
#     segments = torch.split(features["token_type_ids"], batch_size)
#     labels = torch.split(labels, batch_size)
#     batches_inputs = [
#         {"input_ids": i[0], "attention_mask": i[1], "token_type_ids": i[2]}
#         for i in zip(ids, masks, segments)
#     ]
#     res = list(zip(batches_inputs, labels))
#     return res


#     FOR SIAMESE
#     sent1 = list(map(vocab.tokens_to_ids, sent1))
#     sent2 = list(map(vocab.tokens_to_ids, sent2))
#     # labels = map(lambda x: dic_labels[x], df["gold_label"])
#     return MyDataLoader(sent1, sent2, labels, batch_size)
