import json
import random
from queue import Queue
from threading import Thread

import torch

from ..base_data import IGNORE_TOKEN_ID, TBatch

# from torch.utils.data import DataLoader, DistributedSampler


def shuffle_tensor(tensor, generator):
    perm = torch.randperm(len(tensor), generator=generator)
    return tensor[perm]


class BatchIter:
    def __init__(self, line_iter, tokenizer, params):
        print("### CREATING BATCH ITER")
        self.line_iter = line_iter
        self.params = params
        self.batch_size = params["batch_size"]
        self.max_length = params["max_length"]
        self.tokenizer = tokenizer
        self.batches_per_epoch = params["cnt_samples_per_epoch"] / (params["batch_size"] * params["cnt_workers"])
        print("required samples per epoch", params["cnt_samples_per_epoch"])
        print("batch size", params["batch_size"])
        print("cnt_workers", params["cnt_workers"])
        print("batches per epoch", self.batches_per_epoch)
        self.adjust_processed_batches_on_resume()
        self.prepare_dummy_batch()
        self.ignore_token_id = IGNORE_TOKEN_ID
        self._queue = Queue(maxsize=5)
        self._thread = Thread(target=self.thread, args=(), daemon=True)
        self._thread.start()

    def adjust_processed_batches_on_resume(self):
        if "train_logs" in self.params and len(self.params["train_logs"]) > 1:
            # this is resume
            cnt_samples_seen_in_last_epoch = (
                self.params["cnt_samples_processed"] - self.params["train_logs"][-2]["cnt_samples_processed"]
            )
            # TODO: this is gonna break now that accum batch size is per epoch
            self.cnt_batches_produced = cnt_samples_seen_in_last_epoch / (
                self.params["batch_size"] * self.params["cnt_workers"]
            )
        else:
            self.cnt_batches_produced = 0

    def prepare_dummy_batch(self):
        # this is for perf measuremenet w/o IO bottleneck
        self.dummy_batch = TBatch(
            input_ids=torch.zeros((self.batch_size, self.max_length), dtype=torch.int64),
            token_type_ids=None,
            attention_mask=torch.ones((self.batch_size, self.max_length), dtype=torch.int64),
            labels=torch.zeros((self.batch_size, self.max_length), dtype=torch.int64),
        )

    def __iter__(self):
        return self

    def __next__(self):
        if self.cnt_batches_produced >= self.batches_per_epoch:
            self.cnt_batches_produced = 0
            raise StopIteration()
        batch = self._queue.get(block=True, timeout=400)
        # print(self._queue.qsize())
        if batch is None:
            # self._thread.join()
            raise StopIteration()
        self.cnt_batches_produced += 1
        return batch

    def mask_line(self, line, tokenizer, ignore_token_id, generator=None):
        proba_masking = self.params["proba_masking"]
        proba_random = self.params["proba_random"]
        proba_shortening = self.params["proba_shortening"]
        proba_original = proba_random
        min_truncation_length = 5
        pad_token_id = tokenizer.pad_token_id
        # TODO: move to params
        if random.random() < proba_shortening:
            len_shortened = random.randint(min_truncation_length, len(line))
            for i in range(len_shortened, len(line)):
                line[i] = pad_token_id
        token_ids = torch.LongTensor(line)
        labels = token_ids.clone()
        rolls = torch.rand(token_ids.shape, generator=generator)
        ## NOTE this line makes ratio of masked tokens slightly lower
        ## than each proba_ is set to, however it is a small amount
        if self.params["mask_special_tokens"]:
            mask_non_special = line != pad_token_id
        else:
            mask_non_special = torch.tensor([i not in tokenizer.all_special_ids for i in line])

        mask_with_mask = (rolls < proba_masking) & mask_non_special
        mask_with_random = (rolls < proba_masking + proba_random) & (rolls > proba_masking) & mask_non_special
        mask_with_original = (
            (rolls < proba_masking + proba_random + proba_original)
            & (rolls > proba_masking + proba_random)
            & mask_non_special
        )

        random_tokens = torch.randint(len(tokenizer), token_ids.shape, dtype=torch.long)

        token_ids[mask_with_mask] = tokenizer.mask_token_id
        token_ids[mask_with_random] = random_tokens[mask_with_random]
        labels[~(mask_with_mask | mask_with_random | mask_with_original)] = ignore_token_id

        # TODO: check if we have too few or too many masks?
        # wasn't constant number if mask positions better?
        # TODO: why not set attention mask to 0 in these positions??

        # print(mask_mask)
        # the way with fixed count of masked tokens each time
        # ids_nonzero = line.nonzero(as_tuple=True)[0][1:-1]
        # ids_nonzero = shuffle_tensor(ids_nonzero, generator=generator)
        # cnt_masked = int(len(ids_nonzero) * proba_masking)
        # ids_nonzero = ids_nonzero[:cnt_masked]
        # line[ids_nonzero] = mask_id
        return token_ids, labels

    def encode_batch(self, batch):
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        for line in batch:
            # input_ids = self.tokenizer.convert_tokens_to_ids(line)
            input_ids = json.loads(line)
            assert len(input_ids) <= self.max_length, f"got seq of len {len(input_ids)}"
            # input_ids = [self.tokenizer.cls_token_id] + input_ids
            masked_ids, labels = self.mask_line(input_ids, self.tokenizer, self.ignore_token_id)
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

    # def randomly_shorten_line(self, line):
    #     proba_shortening = 0.1
    #     min_length = 5
    #     if random.random() < proba_shortening:
    #         line = line[: random.randint(min_length, len(line))]
    #     return line

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
