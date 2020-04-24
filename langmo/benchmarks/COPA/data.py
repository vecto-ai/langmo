import numpy as np


# TODO: make it actually an iterator
class Iterator:
    def __init__(self, tuples_train, size_batch):
        self.size_batch = size_batch
        train = sorted(tuples_train, key=lambda x: x[0].shape[0])
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
        max_len = max([len(i[0]) for i in batch])
        padded = [(self.zero_pad_item(item, max_len), label) for item, label in batch]
        samples, labels = zip(*padded)
        samples = np.vstack(samples)
        samples = np.rollaxis(samples, 1, start=0)  # make it sequence-first
        labels = np.array(labels)
        return samples, labels
