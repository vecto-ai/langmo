import datasets
import torch
import torch.distributed as dist
from langmo.benchmarks.base_data import BaseCollator, BaseDataModule


class GLUECollator(BaseCollator):
    def __init__(self, tokenizer, params):
        super().__init__(tokenizer, params)
        self.sent1 = self.params["sent1"]
        self.sent2 = self.params["sent2"]

    def __call__(self, x):
        sents1 = [i[self.sent1] for i in x]
        sents2 = [i[self.sent2] for i in x] if self.sent2 is not None else None
        labels = [i["label"] for i in x]
        labels = torch.tensor(labels)
        if self.params["num_labels"] > 1:
            labels = labels.long()
        elif self.params["num_labels"] == 1:
            labels = labels.unsqueeze(1)
        # TODO: refactor into BaseCollator as
        # it is the same here and in NLI
        if not self.params["siamese"]:
            features = self.tokenizer(text=sents1, text_pair=sents2, **self.tokenizer_params)
        else:
            sent1 = self.tokenizer(text=sents1, **self.tokenizer_params)
            sent2 = self.tokenizer(text=sents2, **self.tokenizer_params)
            features = {"left": sent1, "right": sent2}
        return (features, labels)


class GLUEDataModule(BaseDataModule):
    def __init__(self, tokenizer, params):
        super().__init__(tokenizer, params)
        self.collator = GLUECollator(self.tokenizer, params)
        self.params = params

    def setup(self, stage=None):
        self.cnt_train_samples = 0
        if self.trainer.global_rank == 0:
            ds = datasets.load_dataset(*(self.params["glue_type"], self.params["name_task"]))
            self.cnt_train_samples = len(ds["train"])
        num_samples_tensor = torch.LongTensor([self.cnt_train_samples])
        if self.params["cnt_gpus_per_node"] > 0:
            num_samples_tensor = num_samples_tensor.cuda()
        dist.broadcast(num_samples_tensor, 0)
        self.cnt_train_samples = num_samples_tensor.item()

    def train_dataloader(self):
        return [self.get_split_dataloader((self.params["glue_type"], self.params["name_task"]), "train")]

    def val_dataloader(self):
        val_splits = self.params["validation_split"]
        # TODO: allow extra validation to be a parameter
        datasets = [self.get_split_dataloader(split["dataset"], split["split"]) for split in val_splits]
        return datasets
