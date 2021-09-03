import json
import random
from collections import namedtuple
from queue import Queue
from threading import Thread

import horovod.torch as hvd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, DistributedSampler
from vecto.corpus import Corpus, DirCorpus

IGNORE_TOKEN_ID = -100

TBatch = namedtuple(
    "TBatch", ["input_ids", "token_type_ids", "attention_mask", "labels"]
)


def shuffle_tensor(tensor, generator):
    perm = torch.randperm(len(tensor), generator=generator)
    return tensor[perm]


def mask_line(line, tokenizer, ignore_token_id, generator=None):
    # TODO: move to config
    proba_masking = 0.15
    token_ids = torch.LongTensor(line)
    labels = token_ids.clone()
    rolls = torch.rand(token_ids.shape, generator=generator)
    mask_non_special = torch.tensor([i not in tokenizer.all_special_ids for i in line])
    mask_good_rolls = rolls < proba_masking
    mask_mask = mask_good_rolls & mask_non_special
    # TODO: check if we have too few or too many masks?
    # wasn't constant number if mask positions better?
    # TODO: with 0.1 proba mask not with [MASK] but with original token
    # TODO: with same proba mask with random token
    # for time being let's just mask with mask
    # how usefull is it to combine
    # TODO: why not set attention mask to 0 in these positions??
    token_ids[mask_mask] = tokenizer.mask_token_id
    labels[~mask_mask] = ignore_token_id
    # print(mask_mask)
    # the way with fixed count of masked tokens each time
    # ids_nonzero = line.nonzero(as_tuple=True)[0][1:-1]
    # ids_nonzero = shuffle_tensor(ids_nonzero, generator=generator)
    # cnt_masked = int(len(ids_nonzero) * proba_masking)
    # ids_nonzero = ids_nonzero[:cnt_masked]
    # line[ids_nonzero] = mask_id
    return token_ids, labels


class BatchIter:
    def __init__(self, line_iter, tokenizer, params):
        print("### CREATING BATCH ITER")
        self.line_iter = line_iter
        self.batch_size = params["batch_size"]
        self.max_length = params["max_length"]
        self.tokenizer = tokenizer
        cnt_batches_per_epoch = params["cnt_samples_per_epoch"] / params["batch_size"]
        self.batches_per_epoch = cnt_batches_per_epoch / params["cnt_workers"]
        self.cnt_batches_produced = 0
        self.dummy_batch = TBatch(
            input_ids=torch.zeros((self.batch_size, self.max_length), dtype=torch.int64),
            token_type_ids=None,
            attention_mask=torch.ones((self.batch_size, self.max_length), dtype=torch.int64),
            labels=torch.zeros((self.batch_size, self.max_length), dtype=torch.int64),
        )
        self.ignore_token_id = IGNORE_TOKEN_ID
        self._queue = Queue(maxsize=5)
        self._thread = Thread(target=self.thread, args=(), daemon=True)
        self._thread.start()
        # this is not well documented but seems to apply to all HF models

    def __iter__(self):
        return self

    def __next__(self):
        if self.cnt_batches_produced >= self.batches_per_epoch:
            self.cnt_batches_produced = 0
            raise StopIteration()
        batch = self._queue.get()
        # print(self._queue.qsize())
        if batch is None:
            self._thread.join()
            raise StopIteration()
        self.cnt_batches_produced += 1
        return batch

    def encode_batch(self, batch):
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        for line in batch:
            # input_ids = self.tokenizer.convert_tokens_to_ids(line)
            input_ids = json.loads(line)
            assert len(input_ids) <= self.max_length
            # input_ids = [self.tokenizer.cls_token_id] + input_ids
            masked_ids, labels = mask_line(input_ids, self.tokenizer, self.ignore_token_id)
            attention_mask = torch.ones_like(masked_ids)
            if len(masked_ids) < self.max_length:
                pad = torch.ones(self.max_length - len(masked_ids), dtype=torch.int64)
                pad_ids = pad * self.tokenizer.pad_token_id
                pad_label = pad * self.ignore_token_id
                masked_ids = torch.hstack([masked_ids, pad_ids])
                labels = torch.hstack([labels, pad_label])
                attention_mask = torch.hstack([attention_mask, torch.zeros_like(pad)])
            batch_input_ids.append(masked_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)
            # ROBERTA does not seam to shorten lines
            # but if the input is read by sentence, we bight want to trim by nearest sentence end
            # input_ids = self.randomly_shoren_line(input_ids)
            # at this stage we do not expect any special characters, so masking should not care
            # masking
        # encoded = self.tokenizer(
        #     lines,
        #     # is_split_into_words=True,
        #     max_length=self.max_length,
        #     # TODO: consider padding to the max length of the batch
        #     padding="max_length",
        #     truncation=True,
        #     return_tensors="pt",
        # )
        # TODO: for languages which can be tokenated - add support of word-level masking
        # that is before sequences are converted to IDS
        # # labels = encoded["input_ids"].clone()
        # ids = encoded["input_ids"]
        # for i in range(len(encoded["input_ids"])):
        #     ids[i], mask #mask_line(ids[i], self.tokenizer)
        #     # TODO: check if this number is model-specific
        #     labels[i][~mask] = -100
        return TBatch(
            input_ids=torch.stack(batch_input_ids),
            token_type_ids=None,
            attention_mask=torch.stack(batch_attention_mask),
            labels=torch.stack(batch_labels),
        )

    def randomly_shoren_line(self, line):
        proba_shortening = 0.1
        min_length = 5
        if random.random() < proba_shortening:
            line = line[: random.randint(min_length, len(line))]
        return line

    def read_next_batch(self):
        batch = []
        for line in self.line_iter:
            # line = self.randomly_shoren_line(line)
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

    @property
    def cnt_restarts(self):
        return self.line_iter.cnt_restarts


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
        # TODO: add an option to skip short lines to line iter
        # print("loaded dir structure")
        # line_iter = self.corpus.get_looped_sequence_iterator(
        #     # -1 is there to append CLS later
        #     sequence_length=self.params["max_length"] - 1,
        #     tokenizer=self.tokenizer.tokenize,
        #     rank=hvd.rank(),
        #     size=hvd.size(),
        #     min_length=10,
        #     reset_on_new_line=False
        # )
        line_iter = self.corpus.get_looped_line_iterator(rank=hvd.rank(), size=hvd.size())
        # print("created line iter")
        batch_iter = BatchIter(line_iter, self.tokenizer, self.params)
        return batch_iter

    def val_setup(self):
        self.val_data = list(
            self.val_corpus.get_sequence_iterator(
                self.params["max_length"] - 1,
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
            ids[i], _ = mask_line(ids[i],
                                  tokenizer=self.tokenizer,
                                  ignore_token_id=IGNORE_TOKEN_ID,
                                  generator=self.val_gen)
        return TBatch(
            input_ids=ids,
            token_type_ids=encoded["token_type_ids"] if "token_type_ids" in encoded else None,
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
