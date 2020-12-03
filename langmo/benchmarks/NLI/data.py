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
import transformers
from protonn.utils import describe_var


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
    # LSTM-specific
    # padded = np.rollaxis(padded, 1, 0)
    padded = torch.from_numpy(padded)
    return padded


# def my_collate(x):
#     sent1, sent2, labels = zip(* x)
#     # TODO: get max len from both parts
#     sent1 = sequences_to_padded_tensor(sent1)
#     sent2 = sequences_to_padded_tensor(sent2)
#     labels = torch.LongTensor(labels)
#     # TODO: rollaxis
#     return (sent1, sent2, labels)


class MyDataLoader():
    def __init__(self, sent1, sent2, labels, batch_size):
        # optinally sort
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
        return ((sent1, sent2), labels)

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        return self.batches[idx]


def ds_to_tensors(dataset, tokenizer, batch_size, test):
    sent1 = [i["premise"].lower() for i in dataset]
    sent2 = [i["hypothesis"].lower() for i in dataset]
    labels = [i["label"] for i in dataset]
    if test:
        sent1 = sent1[:32]
        sent2 = sent2[:32]
        labels = labels[:32]
    labels = torch.LongTensor(labels)
    texts_or_text_pairs = list(zip(sent1, sent2))
    features = tokenizer(
        texts_or_text_pairs,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    )
    ids = torch.split(features["input_ids"], batch_size)
    masks = torch.split(features["attention_mask"], batch_size)
    segments = torch.split(features["token_type_ids"], batch_size)
    labels = torch.split(labels, batch_size)
    return list(zip(zip(ids, masks, segments), labels))
#     FOR SIAMESE
#     sent1 = list(map(vocab.tokens_to_ids, sent1))
#     sent2 = list(map(vocab.tokens_to_ids, sent2))
#     # labels = map(lambda x: dic_labels[x], df["gold_label"])
#     return MyDataLoader(sent1, sent2, labels, batch_size)


class NLIDataModule(pl.LightningDataModule):
    def __init__(self, vocab, batch_size, test):
        super().__init__()
        self.batch_size = batch_size
        self.vocab = vocab
        self.test = test
        self.percent_start = float(hvd.rank()) / float(hvd.size()) * 100
        self.percent_end = float(hvd.rank() + 1) / float(hvd.size()) * 100

    def setup(self, stage=None):
        # print("doing setup")
        # TODO: do donwload here
        # TODO: probably need to scatter indices here by hvd explicitly

        pass

    def train_dataloader(self):
        ri = datasets.ReadInstruction('train',
                                      from_=self.percent_start,
                                      to=self.percent_end, unit='%')
        ds = datasets.load_dataset('multi_nli', split=ri)
        return ds_to_tensors(ds, self.vocab, self.batch_size, self.test)

    def val_dataloader(self):
        ri = datasets.ReadInstruction('validation_matched',
                                      from_=self.percent_start,
                                      to=self.percent_end, unit='%')
        ds = datasets.load_dataset('multi_nli', split=ri)
        dataloader_matched = ds_to_tensors(ds, self.vocab, self.batch_size, self.test)

        ri = datasets.ReadInstruction('validation_mismatched',
                                      from_=self.percent_start,
                                      to=self.percent_end, unit='%')
        ds = datasets.load_dataset('multi_nli', split=ri)
        dataloader_mismatched = ds_to_tensors(ds, self.vocab, self.batch_size, self.test)

        ri = datasets.ReadInstruction('validation',
                                      from_=self.percent_start,
                                      to=self.percent_end, unit='%')
        ds = datasets.load_dataset('hans', split=ri)
        dataloader_hans = ds_to_tensors(ds, self.vocab, self.batch_size, self.test)

        return [dataloader_matched, dataloader_mismatched, dataloader_hans]
