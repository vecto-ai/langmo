import torch
from langmo.benchmarks.base_data import BaseDataModule, BaseCollator
import horovod.torch as hvd
import datasets


class Collator(BaseCollator):
    def __call__(self, x):
        sents = [i["sentence"] for i in x]
        labels = [i["label"] for i in x]
        labels = torch.round(torch.tensor(labels)).long()
        features = self.tokenizer(text=sents, **self.tokenizer_params)
        return (features, labels)


class SSTDataModule(BaseDataModule):
    def __init__(self, tokenizer, params):
        super().__init__(tokenizer, params)
        self.collator = Collator(self.tokenizer, params)

    def setup(self, stage=None):
        self.cnt_train_samples = 0
        if hvd.rank() == 0:
            ds = datasets.load_dataset("sst")
            self.cnt_train_samples = len(ds["train"])

        num_samples_tensor = torch.LongTensor([self.cnt_train_samples])
        self.cnt_train_samples = hvd.broadcast(num_samples_tensor, 0).item()

    def train_dataloader(self):
        return [self.get_split_dataloader("sst", "train")]

    def val_dataloader(self):
        return [self.get_split_dataloader("sst", "validation")]
