import numpy as np


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
