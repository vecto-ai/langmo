from collections import namedtuple
from queue import Queue
from threading import Thread

import horovod.torch as hvd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, DistributedSampler
from vecto.corpus import Corpus, DirCorpus

TBatch = namedtuple(
    "TBatch", ["input_ids", "token_type_ids", "attention_mask", "labels"]
)


def shuffle_tensor(tensor, generator):
    perm = torch.randperm(len(tensor), generator=generator)
    return tensor[perm]


def mask_line(line, mask_id, generator=None):
    # TODO: move to config
    proba_masking = 0.15
    ids_nonzero = line.nonzero(as_tuple=True)[0][1:-1]
    ids_nonzero = shuffle_tensor(ids_nonzero, generator=generator)
    cnt_masked = int(len(ids_nonzero) * proba_masking)
    ids_nonzero = ids_nonzero[:cnt_masked]
    line[ids_nonzero] = mask_id
    return line


class BatchIter:
    def __init__(self, line_iter, tokenizer, params):
        print("### CREATING BATCH ITER")
        self.line_iter = line_iter
        self.batch_size = params["batch_size"]
        self.max_length = params["max_length"]
        self.tokenizer = tokenizer
        self._queue = Queue(maxsize=5)
        self._thread = Thread(target=self.thread, args=(), daemon=True)
        self._thread.start()
        cnt_batches_per_epoch = params["cnt_samples_per_epoch"] / params["batch_size"]
        self.batches_per_epoch = cnt_batches_per_epoch / hvd.size()
        self.cnt_batches_produced = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.cnt_batches_produced >= self.batches_per_epoch:
            self.cnt_batches_produced = 0
            raise StopIteration()
        batch = self._queue.get()
        # print(self._queue.qsize())
        # if batch is None:
        #    self._thread.join()
        # raise StopIteration()
        self.cnt_batches_produced += 1
        return batch
        # return next(self.__gen__)

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
            ids[i] = mask_line(ids[i], self.tokenizer.mask_token_id)
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
        self.val_corpus = DirCorpus(self.params["path_val_corpus"])

    def setup(self, stage=None):
        # TODO: do this in rank 0 and send to the rest
        # Otherwise make sure files are sorted in the same order
        self.corpus.load_dir_strucute()
        print("loaded corpus of size", self.corpus.total_bytes)
        # self.corpus_view = CorpusView(self.corpus,
        #                              rank=hvd.rank(),
        #                              size=hvd.size())
        self.val_setup()

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
        line_iter = self.corpus.get_looped_sequence_iterator(
            sequence_length=self.params["max_length"] - 2,
            tokenizer=self.tokenizer.tokenize,
            rank=hvd.rank(),
            size=hvd.size(),
        )
        # print("created line iter")
        batch_iter = BatchIter(line_iter, self.tokenizer, self.params)
        return batch_iter

    def val_setup(self):
        self.val_data = list(
            self.val_corpus.get_sequence_iterator(
                self.params["max_length"] - 2,
                self.tokenizer.tokenize,
            )
        )
        self.val_gen = torch.Generator()
        self.val_rng_reset()

    def val_collator(self, batch, generator=None):
        # THIS IS A BLATANT COPY! of the logic of `BatchIter` -- Emil
        lines = [self.tokenizer.convert_tokens_to_string(line) for line in batch]
        encoded = self.tokenizer(
            lines,
            max_length=self.params["max_length"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        encoded["labels"] = encoded["input_ids"].clone()
        ids = encoded["input_ids"]
        for i in range(len(encoded["input_ids"])):
            ids[i] = mask_line(ids[i], self.tokenizer.mask_token_id, self.val_gen)
        return TBatch(
            input_ids=ids,
            token_type_ids=encoded["token_type_ids"],
            attention_mask=encoded["attention_mask"],
            labels=encoded["labels"],
        )

    def val_rng_reset(self):
        self.val_gen.manual_seed(42)

    def val_dataloader(self):
        self.val_rng_reset()
        sampler = DistributedSampler(self.val_data, hvd.size(), hvd.rank(), False)
        return DataLoader(
            self.val_data,
            batch_size=self.params["batch_size"],
            collate_fn=self.val_collator,
            sampler=sampler,
        )
