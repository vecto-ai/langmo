import datasets
import horovod.torch as hvd
import torch
from langmo.benchmarks.base_data import BaseCollator, BaseDataModule


class Collator(BaseCollator):
    def __call__(self, batch):
        contexts = [i["context"] for i in batch]
        questions = [i["question"] for i in batch]
        # labels = torch.round(torch.tensor(labels)).long()
        features = self.tokenizer(
            text=contexts, text_pair=questions, **self.tokenizer_params
        )
        # print()
        # print("features")
        # print(features)
        # print()
        pos_token_start = []
        pos_token_end = []
        for id_item, item in enumerate(batch):
            # print(id_item)
            # print(item["context"][:20], len(item["context"]))
            # print(item["answers"]["answer_start"], len(item["answers"]["answer_start"]))
            # print(item["answers"]["answer_start"][0])
            # print(features.char_to_token(id_item, item["answers"]["answer_start"][0]))
            if len(item["answers"]["answer_start"]) > 0:
                # print(features["input_ids"][id_item].shape)
                start = features.char_to_token(id_item, item["answers"]["answer_start"][0])
                end = features.char_to_token(
                    id_item,
                    item["answers"]["answer_start"][0]
                    + len(item["answers"]["text"]))
                if start is None:
                    start = 0
                if end is None:
                    end = 0
                    start = 0
                pos_token_start.append(start)
                pos_token_end.append(end)
            else:
                pos_token_start.append(0)
                pos_token_end.append(0)
        # print(pos_token_start)
        # print(pos_token_end)
        pos_token_start = torch.LongTensor(pos_token_start)
        pos_token_end = torch.LongTensor(pos_token_end)
        features["start_positions"] = pos_token_start
        features["end_positions"] = pos_token_end
        answers = [i["answers"]["text"] for i in batch]
        return features, answers


class QADataModule(BaseDataModule):
    def __init__(self, tokenizer, params):
        super().__init__(tokenizer, params)
        self.collator = Collator(self.tokenizer, params)

    def setup(self, stage=None):
        self.cnt_train_samples = 0
        if hvd.rank() == 0:
            ds = datasets.load_dataset("squad_v2")
            self.cnt_train_samples = len(ds["train"])

        num_samples_tensor = torch.LongTensor([self.cnt_train_samples])
        self.cnt_train_samples = hvd.broadcast(num_samples_tensor, 0).item()

    def train_dataloader(self):
        return [self.get_split_dataloader("squad_v2", "train")]

    def val_dataloader(self):
        return [self.get_split_dataloader("squad_v2", "validation")]
