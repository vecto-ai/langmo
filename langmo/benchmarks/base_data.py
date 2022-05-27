import datasets
import pytorch_lightning as pl
from protonn.distributed import dist_adapter as da
from torch.utils.data import DataLoader, DistributedSampler


class BaseDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, params):
        super().__init__()
        self.batch_size = params["batch_size"]
        self.tokenizer = tokenizer
        self.params = params
        self.shuffle = params["shuffle"]
        self.test = params["test"]

    def get_split_dataloader(self, dataset_name, split):
        self.percent_start = float(self.trainer.global_rank) / float(self.trainer.world_size) * 100
        self.percent_end = float(self.trainer.global_rank + 1) / float(self.trainer.world_size) * 100
        shuffle = (split == "train") and (self.shuffle)
        if self.test:
            ds_size = self.batch_size * 2
            start = da.rank() * ds_size
            split = f"{split}[{start}:{start+ds_size}]"
            dataset = datasets.load_dataset(dataset_name, split=split)
            sampler = None
        elif shuffle:
            dataset = datasets.load_dataset(dataset_name, split=split)
            sampler = DistributedSampler(dataset, self.trainer.world_size, self.trainer.global_rank, shuffle)
        else:
            split = datasets.ReadInstruction(
                split,
                from_=self.percent_start,
                to=self.percent_end,
                unit="%",
            )
            dataset = datasets.load_dataset(dataset_name, split=split)
            sampler = None
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=self.collator,
            sampler=sampler,
        )


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
