import torch
import pytorch_lightning as pl
from vecto.corpus import ViewCorpus
import horovod.torch as hvd
from collections import namedtuple


TBatch = namedtuple("TBatch", ["input_ids", "token_type_ids", "attention_mask", "labels"])


def shuffle_tensor(tensor):
    perm_indices = torch.randperm(len(tensor))
    return tensor[perm_indices]


class BatchIter:
    def __init__(self, line_iter, tokenizer, params):
        self.__gen__ = self._generate_samples()
        self.line_iter = line_iter
        self.batch_size = params["batch_size"]
        self.max_length = params["max_length"]
        self.tokenizer = tokenizer

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.__gen__)

    def mask_line(self, line, mask_id):
        # TODO: move to config
        proba_masking = 0.15
        ids_nonzero = line.nonzero(as_tuple=True)[0]
        ids_nonzero = shuffle_tensor(ids_nonzero)
        cnt_masked = int(len(ids_nonzero) * proba_masking)
        ids_nonzero = ids_nonzero[:cnt_masked]
        line[ids_nonzero] = mask_id
        return line

    def encode_batch(self, lines):
        encoded = self.tokenizer(
            lines,
            max_length=self.max_length,
            # TODO: consider padding to the max length of the batch
            padding="max_length",
            truncation="only_first",
            return_tensors="pt",
        )
        # TODO: for languages which can be tokenated - add support of word-level masking
        # that is before sequences are converted to IDS

        # consider masking or not masking special tokens like SEP and CLS

        encoded["labels"] = encoded["input_ids"].clone()
        ids = encoded["input_ids"]
        for i in range(len(encoded["input_ids"])):
            ids[i] = self.mask_line(ids[i], self.tokenizer.mask_token_id)
            ids[i][0] = 42
        return TBatch(input_ids=ids,
                      token_type_ids=encoded["token_type_ids"],
                      attention_mask=encoded["attention_mask"],
                      labels=encoded["labels"])

    def _generate_samples(self):
        batch = []
        for line in self.line_iter:
            # TODO: should be a parameter
            if len(line) > 10:
                batch.append(line)
            if len(batch) == self.batch_size:
                ret = self.encode_batch(batch)
                yield ret
                batch = []


class TextDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, params):  # , vocab, batch_size, params):
        super().__init__()
        self.params = params
        self.tokenizer = tokenizer

    def setup(self, stage=None):
        # print("doing setup")
        # TODO: do donwload here
        pass

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

        corpus = ViewCorpus(self.params["path_corpus"])
        # TODO: do this in rank 0 and send to the rest
        # Otherwise make sure files are sorted in the same order
        # TODO: implement some proper logger
        # print("created view corpus")
        corpus.load_dir_strucute()
        # TODO: add an option to skip short lines to line iter
        # print("loaded dir structure")
        line_iter = corpus.get_line_iterator(rank=hvd.rank(), size=hvd.size())
        # print("created line iter")
        batch_iter = BatchIter(line_iter, self.tokenizer, self.params)
        return batch_iter
