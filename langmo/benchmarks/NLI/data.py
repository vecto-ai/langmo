import numpy as np
import pandas
import json
import pytorch_lightning as pl


def read_ds(path, embs, test=False):
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
    dic_labels = {l: i for i, l in enumerate(sorted(df["gold_label"].unique()))}
    df["sentence1"] = df["sentence1"].apply(lambda s: s.lower())
    df["sentence2"] = df["sentence2"].apply(lambda s: s.lower())
#    print(df["sentence1"][:10])
    sent1 = map(embs.vocabulary.tokens_to_ids, df["sentence1"])
    sent2 = map(embs.vocabulary.tokens_to_ids, df["sentence2"])
    labels = map(lambda x: dic_labels[x], df["gold_label"])
    tuples = zip(zip(sent1, sent2), labels)
    return tuples


# TODO: make it actually an iterator
class Iterator:
    def __init__(self, tuples_train, size_batch):
        self.size_batch = size_batch
        train = sorted(tuples_train, key=lambda x: max(x[0][0].shape[0], x[0][1].shape[0]))
        self.cnt_samples = len(train)
        self.batches = []
        for i in range(0, len(train), size_batch):
            batch = train[i:i + size_batch]
            self.batches.append(self.zero_pad_batch(batch))

    def zero_pad_item(self, sample, max_len):
        size_pad = max_len - sample.shape[0]
        assert size_pad >= 0
        res = np.hstack([np.zeros(size_pad, dtype=np.int64), sample])
        return res

    def zero_pad_batch(self, batch):
        max_len = max([max(len(i[0][0]), len(i[0][1])) for i in batch])
        list_s1 = []
        list_s2 = []
        list_labels = []
        for sample in batch:
            (s1, s2), label = sample
            s1 = self.zero_pad_item(s1, max_len)
            s2 = self.zero_pad_item(s2, max_len)
            list_s1.append(s1)
            list_s2.append(s2)
            list_labels.append(label)
        block_s1 = np.vstack(list_s1)
        block_s2 = np.vstack(list_s2)
        block_s1 = np.rollaxis(block_s1, 1, start=0)  # make it sequence-first
        block_s2 = np.rollaxis(block_s2, 1, start=0)  # make it sequence-first
        labels = np.array(list_labels)
        return (block_s1, block_s2), labels


class NLIDataModule(pl.LightningDataModule):
    def __init__(self, path, batch_size):
        super().__init__()
        self.path = path
        self.batch_size = batch_size

    def setup(self, stage=None):
        # print("doing setup")
        # TODO: probably need to scatter indices here by hvd explicitly
        pass

    def train_dataloader(self):
        return load_ds_from_dir(os.path.join(self.path, "train"), self.batch_size)

    def val_dataloader(self):
        return load_ds_from_dir(os.path.join(self.path, "validation"), self.batch_size)
