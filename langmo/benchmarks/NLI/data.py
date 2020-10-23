# import numpy as np
import pandas
import json
import torch
import pytorch_lightning as pl
import os
import numpy as np
from torch.utils.data import DataLoader
import horovod.torch as hvd


def zero_pad_item(sample, max_len):
    if sample.shape[0] > max_len:
        return sample[:max_len]
    else:
        size_pad = max_len - sample.shape[0]
        # print("!!!!", sample)
        assert size_pad >= 0
        res = np.hstack([np.zeros(size_pad, dtype=np.int64), sample])
        return res


def sequences_to_padded_tensor(seqs, max_len):
    seqs = list(seqs)
    #max_len = max([len(s) for s in seqs])
    # print(max_len)
    #if max_len > 128:
    #    max_len = 128
    # print(seqs)
    padded = [zero_pad_item(s, max_len) for s in seqs]
    padded = np.array(padded, dtype=np.int64)
    padded = np.rollaxis(padded, 1, 0)
    padded = torch.from_numpy(padded)
    return padded


def my_collate(x):
    sent1, sent2, labels = zip(* x)
    # TODO: get max len from both parts
    sent1 = sequences_to_padded_tensor(sent1)
    sent2 = sequences_to_padded_tensor(sent2)
    labels = torch.LongTensor(labels)
    # TODO: rollaxis
    return (sent1, sent2, labels)


class MyDataLoader():
    def __init__(self, sent1, sent2, labels):
        # optinally sort
        batch_size = 32
        tuples = list(zip(sent1, sent2, labels))
        cnt_batches = len(tuples) / batch_size
        batches = np.array_split(tuples, cnt_batches)
        self.batches = [self.zero_pad_batch(b) for b in batches]

    def zero_pad_batch(self, batch):
        sent1, sent2, labels = zip(* batch)
        max_len_sent1 = max(len(s) for s in sent1)
        max_len_sent2 = max(len(s) for s in sent2)
        max_len = max(max_len_sent1, max_len_sent2)
        sent1 = sequences_to_padded_tensor(sent1, max_len)
        sent2 = sequences_to_padded_tensor(sent2, max_len)
        labels = torch.LongTensor(labels)
        return (sent1, sent2, labels)

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        return self.batches[idx]


def read_ds(path, vocab, batch_size, test=False):
    train = []
    cnt = 0
    with open(path) as f:
        for line in f:
            train.append(json.loads(line))
            cnt += 1
            if test and cnt > 2048:
                break
    print(f"{len(train)} samples loaded")
    df = pandas.DataFrame(train)
    chunks = np.array_split(df, hvd.size())
    df = chunks[hvd.rank()]
    dic_labels = {l: i for i, l in enumerate(sorted(df["gold_label"].unique()))}
    df["sentence1"] = df["sentence1"].apply(lambda s: s.lower())
    df["sentence2"] = df["sentence2"].apply(lambda s: s.lower())
#    print(df["sentence1"][:10])
    # TODO: abstract tokenization away
    sent1 = list(map(vocab.tokens_to_ids, df["sentence1"]))
    sent2 = map(vocab.tokens_to_ids, df["sentence2"])
    labels = map(lambda x: dic_labels[x], df["gold_label"])
    # dataset = list(zip(sent1, sent2, labels))
    # apply padding here
    return MyDataLoader(sent1, sent2, labels)
    #return DataLoader(dataset, collatellate_fn=my_collate, batch_size=batch_size, num_workers=1)
    # # tuples = zip(zip(sent1, sent2), labels)


class NLIDataModule(pl.LightningDataModule):
    def __init__(self, path, vocab, batch_size):
        super().__init__()
        self.path = path
        self.batch_size = batch_size
        self.vocab = vocab

    def setup(self, stage=None):
        # print("doing setup")
        # TODO: probably need to scatter indices here by hvd explicitly
        pass

    def train_dataloader(self):
        path = os.path.join(self.path, "multinli_1.0_train.jsonl")
        return read_ds(path, self.vocab, self.batch_size)

    def val_dataloader(self):
        path = os.path.join(self.path, "multinli_1.0_dev_matched.jsonl")
        return read_ds(path, self.vocab, self.batch_size)


# TODO: make it actually an iterator
# class Iterator:
#     def __init__(self, tuples_train, size_batch):
#         self.size_batch = size_batch
#         train = sorted(tuples_train, key=lambda x: max(x[0][0].shape[0], x[0][1].shape[0]))
#         self.cnt_samples = len(train)
#         self.batches = []
#         for i in range(0, len(train), size_batch):
#             batch = train[i:i + size_batch]
#             self.batches.append(self.zero_pad_batch(batch))

#     def zero_pad_item(self, sample, max_len):
#         size_pad = max_len - sample.shape[0]
#         assert size_pad >= 0
#         res = np.hstack([np.zeros(size_pad, dtype=np.int64), sample])
#         return res

#     def zero_pad_batch(self, batch):
#         max_len = max([max(len(i[0][0]), len(i[0][1])) for i in batch])
#         list_s1 = []
#         list_s2 = []
#         list_labels = []
#         for sample in batch:
#             (s1, s2), label = sample
#             s1 = self.zero_pad_item(s1, max_len)
#             s2 = self.zero_pad_item(s2, max_len)
#             list_s1.append(s1)
#             list_s2.append(s2)
#             list_labels.append(label)
#         block_s1 = np.vstack(list_s1)
#         block_s2 = np.vstack(list_s2)
#         block_s1 = np.rollaxis(block_s1, 1, start=0)  # make it sequence-first
#         block_s2 = np.rollaxis(block_s2, 1, start=0)  # make it sequence-first
#         labels = np.array(list_labels)
#         return (block_s1, block_s2), labels
