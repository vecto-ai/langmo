from collections import namedtuple

import horovod.torch as hvd
import pytorch_lightning as pl
import torch
from vecto.corpus import Corpus, CorpusView
from threading import Thread
from queue import Queue

TBatch = namedtuple(
    "TBatch", ["input_ids", "token_type_ids", "attention_mask", "labels"]
)


def shuffle_tensor(tensor):
    perm_indices = torch.randperm(len(tensor))
    return tensor[perm_indices]


class BatchIter:
    def __init__(self, line_iter, tokenizer, params):
        self.line_iter = line_iter
        self.batch_size = params["batch_size"]
        self.max_length = params["max_length"]
        self.tokenizer = tokenizer
        self._queue = Queue(maxsize=5)
        self._thread = Thread(target=self.thread, args=(), daemon=True)
        self._thread.start()

    def __iter__(self):
        return self

    def __next__(self):
        batch = self._queue.get()
        # print(self._queue.qsize())
        if batch is None:
            self._thread.join()
            raise StopIteration()
        return batch
        # return next(self.__gen__)

    def mask_line(self, line, mask_id):
        # TODO: move to config
        proba_masking = 0.15
        ids_nonzero = line.nonzero(as_tuple=True)[0]
        ids_nonzero = shuffle_tensor(ids_nonzero)
        cnt_masked = int(len(ids_nonzero) * proba_masking)
        ids_nonzero = ids_nonzero[:cnt_masked]
        line[ids_nonzero] = mask_id
        return line

    def encode_batch(self, batch):
        lines = [self.tokenizer.convert_tokens_to_string(line) for line in batch]
        encoded = self.tokenizer(
            lines,
            # is_split_into_words=True,
            max_length=self.max_length,
            # TODO: consider padding to the max length of the batch
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        # TODO: for languages which can be tokenated - add support of word-level masking
        # that is before sequences are converted to IDS

        # consider masking or not masking special tokens like SEP and CLS

        encoded["labels"] = encoded["input_ids"].clone()
        ids = encoded["input_ids"]
        for i in range(len(encoded["input_ids"])):
            ids[i] = self.mask_line(ids[i], self.tokenizer.mask_token_id)
        return TBatch(
            input_ids=ids,
            token_type_ids=encoded["token_type_ids"],
            attention_mask=encoded["attention_mask"],
            labels=encoded["labels"],
        )

    def read_next_batch(self):
        batch = []
        for line in self.line_iter:
            if len(line) > 10:
                batch.append(line)
            if len(batch) == self.batch_size:
                ret = self.encode_batch(batch)
                yield ret
                batch = []
        # discarding last incomplete batch here

    def thread(self):
        for batch in self.read_next_batch():
            self._queue.put(batch)
        self._queue.put(None)


class TextDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, params):  # , vocab, batch_size, params):
        super().__init__()
        self.params = params
        self.tokenizer = tokenizer
        self.corpus = Corpus(self.params["path_corpus"])

    def setup(self, stage=None):
        # TODO: do this in rank 0 and send to the rest
        # Otherwise make sure files are sorted in the same order
        self.corpus.load_dir_strucute()
        print("loaded corpus of size", self.corpus.total_bytes)
        self.corpus_view = CorpusView(self.corpus,
                                      rank=hvd.rank(),
                                      size=hvd.size())

    def train_dataloader(self):
        # sent3 options:
        # - recycle old sentences (ring buffer)
        # - read from other worker (another corpus view~
        # - random place in current view (assuming it is big enough)
        # - something else
        #
        # encoded = { ids, masks, etc } == batch
        #
        # mlm_ids = shitton of text, possibly spanning multiple sentences.
        # sent1_ids and sent2_ids = consecutive sentences (single sentence)
        # sent3_ids = random/remote = sentence (single sentence)
        # return [[mlm_ids, sent1_ids, sent2_ids, sent3_ids]  x for batch_size] y for cnt_batches]

        # TODO: implement some proper logger
        # print("created view corpus")
        # TODO: add an option to skip short lines to line iter
        # print("loaded dir structure")
        line_iter = self.corpus_view.get_sequence_iterator(sequence_length=self.params["max_length"] - 2,
                                                          tokenizer=self.tokenizer.tokenize)
        # print("created line iter")
        batch_iter = BatchIter(line_iter, self.tokenizer, self.params)
        return batch_iter
