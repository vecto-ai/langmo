import torch
import pytorch_lightning as pl
from vecto.corpus import ViewCorpus
from torch.utils.data import DataLoader


def shuffle_tensor(tensor):
    perm_indices = torch.randperm(len(tensor))
    return tensor[perm_indices]


class Collate:
    def __init__(self, tokenizer, params):
        self.tokenizer = tokenizer
        self.params = params

    def mask_line(self, line, mask_id):
        # TODO: move to config
        proba_masking = 0.15
        ids_nonzero = line.nonzero(as_tuple=True)[0]
        ids_nonzero = shuffle_tensor(ids_nonzero)
        cnt_masked = int(len(ids_nonzero) * proba_masking)
        ids_nonzero = ids_nonzero[:cnt_masked]
        line[ids_nonzero] = mask_id
        return line

    def __call__(self, items):
        encoded = self.tokenizer(
            items,
            max_length=self.params["max_length"],
            # TODO: consider padding to the max length of the batch
            padding="max_length",
            truncation="only_first",
            return_tensors="pt",
        )
        # TODO: for languages which can be tokenated - add support of word-level masking
        # that is before sequences are converted to IDS

        # consider masking or not masking special tokens like SEP and CLS
        encoded["labels"] = encoded["input_ids"]
        # mask here
        ids = encoded["input_ids"]
        # should we do w/o padding and pad later?
        for i in range(len(encoded["input_ids"])):
            ids[i] = self.mask_line(ids[i], self.tokenizer.mask_token_id)
        # TODO: shouldn't we compute loss only from masked parts?
        return encoded



class TextDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, params):  # , vocab, batch_size, params):
        super().__init__()
        # self.batch_size = batch_size
        # self.vocab = vocab
        self.params = params
        # self.test = params["test"]
        # self.percent_start = float(hvd.rank()) / float(hvd.size()) * 100
        # self.percent_end = float(hvd.rank() + 1) / float(hvd.size()) * 100
        self.tokenizer = tokenizer

    def setup(self, stage=None):
        # print("doing setup")
        # TODO: do donwload here
        pass


    def train_dataloader(self):
        # ri = datasets.ReadInstruction(
        #     "train", from_=self.percent_start, to=self.percent_end, unit="%"
        # )
        # ds = datasets.load_dataset("multi_nli", split=ri)
        # return ds_to_tensors(ds, self.vocab, self.batch_size, self.test, self.params)

        # WHAT WE WANT
        # from vecto.corpus import Corpus
        # corpus = Corpus("/path", worker_id, size)
        # iter = corpus.get_sentence_iterator()
        #
        # TODO:
        # - `worker_id`/rank
        # - sent3???? TRICKY
        #
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
        corpus.load_dir_strucute()
        # TODO: use actual rank and size
        line_iter = corpus.get_line_iterator(rank=0, size=1)
        # my_sequence = ["cows produce beer.", "I like cold milk test test test test test test test."]
        dataset = [line for line in line_iter if len(line) > 10]
        dataloader = DataLoader(
            dataset, batch_size=self.params["batch_size"], collate_fn=Collate(self.tokenizer, self.params))
        # TODO: batch using dataloader
        # TODO: mask random 15% of tokens

        # Example:
        # here `encoded` is one batch
        # batch_size = 2
        # cnt_batches = 3

        return dataloader
