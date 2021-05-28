# import numpy as np
import datasets
from torch.utils.data import DataLoader
import horovod.torch as hvd
import pytorch_lightning as pl
import torch
# from datasets import logging as tr_logging

# from protonn.utils import describe_var


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


class Collator:
    def __init__(self, tokenizer, params):
        self.tokenizer = tokenizer
        self.params = params

    def __call__(self, x):
        sent1 = [i["premise"] for i in x]
        sent2 = [i["hypothesis"] for i in x]
        labels = [i["label"] for i in x]
        labels = torch.LongTensor(labels)
        tokenizer_params = {
            "padding": "max_length",
            "truncation": True,
            "return_tensors": "pt",
        }
        if not self.params["siamese"]:
            features = self.tokenizer(
                text=sent1,
                text_pair=sent2,
                max_length=128,
                ** tokenizer_params,
            )
            return (features, labels)
        # siamese
        sent1 = self.tokenizer(
            text=sent1,
            max_length=128,
            ** tokenizer_params
        )
        sent2 = self.tokenizer(
            text=sent2,
            max_length=128,
            ** tokenizer_params,
        )
        return ({"left": sent1, "right": sent2}, labels)
        # sent1, sent2, labels = zip(* x)
        # TODO: get max len from both parts
        # sent1 = sequences_to_padded_tensor(sent1)
        # sent2 = sequences_to_padded_tensor(sent2)


# class MyDataLoader:
#     def __init__(self, sent1, sent2, labels, batch_size):
#         # optinally sort
#         if hvd.rank() != 0:
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
#         # TODO: use bs and hvd size
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


class NLIDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, batch_size, params):
        super().__init__()
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.params = params
        self.test = params["test"]
        # TODO: if test, make dataset loading faster by setting small percents here
        self.percent_start = float(hvd.rank()) / float(hvd.size()) * 100
        self.percent_end = float(hvd.rank() + 1) / float(hvd.size()) * 100

    def setup(self, stage=None):
        # TODO: get the right data loaded (might be something like "to
        # local SSD")

        self.cnt_train_samples = 0
        if hvd.rank() == 0:
            # TODO: can we download without loading
            ds_hans = datasets.load_dataset("hans")
            print("preload hans", ds_hans)
            ds = datasets.load_dataset("multi_nli")
            self.cnt_train_samples = len(ds["train"])

        num_samples_tensor = torch.LongTensor([self.cnt_train_samples])
        self.cnt_train_samples = hvd.broadcast(num_samples_tensor, 0).item()

    def get_split_dataloader(self, dataset, split):
        ri = datasets.ReadInstruction(
            split,
            from_=self.percent_start,
            to=self.percent_end,
            unit="%",
        )
        ds = datasets.load_dataset(dataset, split=ri)
        collator = Collator(self.tokenizer, self.params)
        return DataLoader(ds, batch_size=self.params["batch_size"], collate_fn=collator)

    def train_dataloader(self):
        return [self.get_split_dataloader("multi_nli", "train")]

    def val_dataloader(self):
        dataloaders = [
            self.get_split_dataloader("multi_nli", "validation_matched"),
            self.get_split_dataloader("multi_nli", "validation_mismatched"),
            self.get_split_dataloader("hans", "validation"),
        ]
        return dataloaders
