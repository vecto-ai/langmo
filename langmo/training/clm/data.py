import json
import random
from collections import namedtuple
from queue import Queue
from threading import Thread

import torch

from ..base_data import IGNORE_TOKEN_ID, TBatch
from ..mlm.data import BatchIter as BaseBatchIter


def shuffle_tensor(tensor, generator):
    perm = torch.randperm(len(tensor), generator=generator)
    return tensor[perm]


class BatchIter(BaseBatchIter):
    def convert_to_tensor(self, line):
        token_ids = torch.LongTensor(line)
        labels = token_ids.clone()
        return token_ids, labels

    def random_shortening(self, line):
        proba_shortening = self.params["proba_shortening"]
        min_truncation_length = 5
        pad_token_id = self.tokenizer.pad_token_id
        if random.random() < proba_shortening:
            len_shortened = random.randint(min_truncation_length, len(line))
            for i in range(len_shortened, len(line)):
                line[i] = pad_token_id
        return line

    def encode_batch(self, batch):
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        for line in batch:
            # input_ids = self.tokenizer.convert_tokens_to_ids(line)
            input_ids = json.loads(line)
            assert len(input_ids) <= self.max_length, f"got seq of len {len(input_ids)}"
            input_ids = self.random_shortening(input_ids)
            input_ids, labels = self.convert_to_tensor(input_ids)
            attention_mask = torch.ones_like(input_ids)
            if len(input_ids) < self.max_length:
                pad = torch.ones(self.max_length - len(input_ids), dtype=torch.int64)
                pad_ids = pad * self.tokenizer.pad_token_id
                pad_label = pad * self.ignore_token_id
                input_ids = torch.hstack([input_ids, pad_ids])
                labels = torch.hstack([labels, pad_label])
                attention_mask = torch.hstack([attention_mask, torch.zeros_like(pad)])
            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)
        return TBatch(
            input_ids=torch.stack(batch_input_ids),
            token_type_ids=None,
            attention_mask=torch.stack(batch_attention_mask),
            labels=torch.stack(batch_labels),
        )
