from collections import namedtuple

import datasets
import lightning as pl
from kapral.corpus import Corpus
from torch.utils.data import DataLoader, DistributedSampler

IGNORE_TOKEN_ID = -100

TBatch = namedtuple("TBatch", ["input_ids", "token_type_ids", "attention_mask", "labels"])


class BaseDataModule(pl.LightningDataModule):
    def __init__(self, cluster_env, tokenizer, params):
        super().__init__()
        self.cluster_env = cluster_env
        self.batch_size = params["batch_size"]
        self.tokenizer = tokenizer
        self.params = params
        self.shuffle = params["shuffle"]
        self.test = params["test"]

    def get_split_dataloader(self, dataset_name, split):
        self.percent_start = float(self.cluster_env.global_rank()) / float(self.cluster_env.world_size()) * 100
        self.percent_end = float(self.cluster_env.global_rank() + 1) / float(self.cluster_env.world_size()) * 100
        shuffle = (split == "train") and (self.shuffle)
        if self.test:
            ds_size = self.batch_size * 2
            start = self.cluster_env.global_rank() * ds_size
            split = f"{split}[{start}:{start + ds_size}]"
            dataset = self._init_dataset((dataset_name), split=split)
            sampler = None
        elif shuffle:
            dataset = self._init_dataset((dataset_name), split=split)
            sampler = DistributedSampler(dataset, self.cluster_env.world_size(), self.cluster_env.global_rank(), shuffle)
        else:
            split = datasets.ReadInstruction(
                split,
                from_=self.percent_start,
                to=self.percent_end,
                unit="%",
            )
            dataset = self._init_dataset((dataset_name), split=split)
            sampler = None
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=self.collator,
            sampler=sampler,
        )

    def _init_dataset(self, dataset_name, split):
        # TODO: consider making it more explicit, e.g. **
        return datasets.load_dataset(*dataset_name, split=split)


class BaseCollator:
    def __init__(self, tokenizer, params):
        self.tokenizer = tokenizer
        self.params = params
        self.tokenizer_params = {
            "padding": params["padding"],
            "truncation": True,
            "return_tensors": "pt",
            "max_length": params["max_length"],
        }


class TextDataModule(pl.LightningDataModule):
    def __init__(self, batch_iterator_cls, cluster_env, tokenizer, params):  # , vocab, batch_size, params):
        super().__init__()
        self.cluster_env = cluster_env
        self.params = params
        self.tokenizer = tokenizer
        self.corpus = Corpus(self.params["path_corpus"])
        self.batch_iterator_cls = batch_iterator_cls  # Batch Iterator Class (NOT object)
        # self.val_corpus = Corpus(self.params["path_val_corpus"])

    def setup(self, stage=None):
        # TODO: do this in rank 0 and send to the rest
        # Otherwise make sure files are sorted in the same order
        self.corpus.load_dir_strucute()
        print("loaded corpus of size", self.corpus.total_bytes)
        # self.corpus_view = CorpusView(self.corpus,
        #                              rank=da.rank(),
        #                              size=da.size())
        # self.val_setup()

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
        #     rank=da.rank(),
        #     size=da.size(),
        #     min_length=10,
        #     reset_on_new_line=False
        # )
        line_iter = self.corpus.get_looped_line_iterator(
            rank=self.trainer.global_rank,
            size=self.trainer.world_size,
        )
        # print("created line iter")
        batch_iter = self.batch_iterator_cls(line_iter, self.tokenizer, self.params)
        return batch_iter

    # def val_setup(self):
    #     self.val_data = list(
    #         self.val_corpus.get_sequence_iterator(
    #             self.params["max_length"] - 1,
    #             self.tokenizer.tokenize,
    #         )
    #     )
    #     self.val_gen = torch.Generator()
    #     self.val_rng_reset()

    def val_rng_reset(self):
        self.val_gen.manual_seed(42)

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
            ids[i], _ = self.mask_line(
                ids[i],
                tokenizer=self.tokenizer,
                ignore_token_id=IGNORE_TOKEN_ID,
                generator=self.val_gen,
            )
        return TBatch(
            input_ids=ids,
            token_type_ids=encoded["token_type_ids"] if "token_type_ids" in encoded else None,
            attention_mask=encoded["attention_mask"],
            labels=encoded["labels"],
        )

    # def val_dataloader(self):
    #     # self.val_rng_reset()
    #     sampler = DistributedSampler(
    #         dataset=self.val_data,
    #         num_replicas=da.world_size(),
    #         rank=da.rank(),
    #         shuffle=False,
    #         seed=42,
    #     )
    #     return DataLoader(
    #         self.val_data,
    #         batch_size=self.params["batch_size"],
    #         collate_fn=self.val_collator,
    #         sampler=sampler,
    #         num_workers=0,
    #         shuffle=False,
    #     )
