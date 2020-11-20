# import numpy as np
# import pandas
# import json
import torch
import pytorch_lightning as pl
# import os
import numpy as np
# from torch.utils.data import DataLoader
import horovod.torch as hvd
import datasets


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


def read_ds(dataset, vocab, batch_size, test):
    sent1 = [i["premise"].lower() for i in dataset]
    sent2 = [i["hypothesis"].lower() for i in dataset]
    labels = [i["label"] for i in dataset]
    sent1 = list(map(vocab.tokens_to_ids, sent1))
    sent2 = list(map(vocab.tokens_to_ids, sent2))
    # labels = map(lambda x: dic_labels[x], df["gold_label"])
    return MyDataLoader(sent1, sent2, labels)


class NLIDataModule(pl.LightningDataModule):
    def __init__(self, path, vocab, batch_size, test):
        super().__init__()
        self.path = path
        self.batch_size = batch_size
        self.vocab = vocab
        self.test = test
        self.percent_start = float(hvd.rank()) / float(hvd.size()) * 100
        self.percent_end = float(hvd.rank() + 1) / float(hvd.size()) * 100

    def setup(self, stage=None):
        # print("doing setup")
        # TODO: probably need to scatter indices here by hvd explicitly
        pass

    def train_dataloader(self):
        ri = datasets.ReadInstruction('train',
                                      from_=self.percent_start,
                                      to=self.percent_end, unit='%')

        ds = datasets.load_dataset('multi_nli', split=ri)
        return read_ds(ds, self.vocab, self.batch_size, self.test)

    def val_dataloader(self):
        ri = datasets.ReadInstruction('validation_matched',
                                      from_=self.percent_start,
                                      to=self.percent_end, unit='%')

        ds = datasets.load_dataset('multi_nli', split=ri)
        return read_ds(ds, self.vocab, self.batch_size, self.test)


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
