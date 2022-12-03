import datasets
import pytorch_lightning as pl
from torch.utils.data import DataLoader, DistributedSampler


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
