from collections import OrderedDict

from transformers import DataCollatorForLanguageModeling

from langmo.benchmarks.GLUE.data import GLUECollator, GLUEDataModule
from langmo.pretraining.data import TextDataModule

from .config import TaskConfigs


class MultitaskDataModule(TextDataModule):
    def __init__(self, cluster_env, tokenizer, params):
        super().__init__(cluster_env, tokenizer, params)
        self.params = TaskConfigs(params)
        self.tasks_params = self.params["tasks"]
        self.tasks_datamodules = OrderedDict()  # useless for python >= 3.8
        for task_name, task_params in self.tasks_params.items():
            if task_name != "mlm":
                self.tasks_datamodules[task_name] = GLUEDataModule(self.cluster_env, self.tokenizer, task_params)

    def setup(self, stage=None):
        super().setup(stage=stage)

        for taskdatamodule in self.tasks_datamodules.values():
            taskdatamodule.trainer = self.trainer
            taskdatamodule.setup(stage=stage)
            if "mlm" in self.tasks_params:
                taskdatamodule.collator = MultitaskGLUECollator(
                    taskdatamodule.tokenizer,
                    taskdatamodule.params,
                )

        # TODO: this is a bit against the logic in configure
        # optimizers about how to pick cnt_samples_per_epoch
        self.cnt_train_samples = max(
            [self.params.get("cnt_samples_per_epoch", 0)]
            + [datamodule.cnt_train_samples for datamodule in self.tasks_datamodules.values()]
        )

    def train_dataloader(self):
        dataloaders = OrderedDict()
        if "mlm" in self.tasks_params:
            dataloaders["mlm"] = super().train_dataloader()
        for name_task, taskdatamodule in self.tasks_datamodules.items():
            dataloaders[name_task] = taskdatamodule.train_dataloader()[0]

        if self.params["multitask_strategy"] == "sequential":
            return SequentialMultitaskDataLoader(self.params, dataloaders)

        return ParallelMultitaskDataLoader(self.params, dataloaders)

    def val_dataloader(self):
        dataloaders = OrderedDict()
        for name_task, taskdatamodule in self.tasks_datamodules.items():
            dataloaders[name_task] = taskdatamodule.val_dataloader()[0]
        return SequentialMultitaskDataLoader(self.params, dataloaders, is_validation=True)


class MultitaskDataLoader:
    def __init__(self, params, dataloaders, is_validation=False):
        self.params = params
        self.dataloaders = dataloaders
        self.dataloaders_epochs = {i: 0 for i in self.dataloaders}
        self.current_step = 0
        self.batches_per_epoch = params["cnt_samples_per_epoch"] / (params["batch_size"] * params["cnt_workers"])
        self.is_validation = is_validation
        self._batch_generator = self._init_batch_generator()

    def __next__(self):
        if (self.current_step > self.batches_per_epoch) and not self.is_validation:
            self.current_step = 0
            raise StopIteration()

        out = next(self._batch_generator)
        if out is None:
            raise StopIteration()
        self.current_step += 1
        return out

    def __iter__(self):
        if (not self.params["continuous_finetune"]) or self.is_validation:
            self._batch_generator = self._init_batch_generator()
            self.current_step = 0
        return self


class ParallelMultitaskDataLoader(MultitaskDataLoader):
    def _init_batch_generator(self):
        self.running_dataloaders = {task_name: iter(dataloader) for task_name, dataloader in self.dataloaders.items()}
        while len(self.running_dataloaders) > 0:
            out = OrderedDict()
            for task_name in list(self.running_dataloaders):
                dataiter = self.running_dataloaders[task_name]
                try:
                    out[task_name] = next(dataiter)
                except StopIteration:
                    if (not self.params["continuous_finetune"]) or self.is_validation:
                        del self.running_dataloaders[task_name]
                    else:
                        self.running_dataloaders[task_name] = iter(self.dataloaders[task_name])

            if len(out) == 0:
                if not self.params["continuous_finetune"]:
                    return
                else:
                    continue

            yield out


class SequentialMultitaskDataLoader(MultitaskDataLoader):
    def _init_batch_generator(self):
        self.running_dataloaders = {task_name: iter(dataloader) for task_name, dataloader in self.dataloaders.items()}
        self.mlm_dataloader = self.running_dataloaders.pop("mlm", None)
        self.mlm_done = False
        while True:
            for task_name, dataiter in self.running_dataloaders.items():
                for batch in dataiter:
                    out = OrderedDict()
                    if (not self.mlm_done) and (self.mlm_dataloader is not None):
                        try:
                            out["mlm"] = next(self.mlm_dataloader)
                        except StopIteration:
                            self.mlm_done = True
                    out[task_name] = batch
                    yield out
            if (not self.params["continuous_finetune"]) or self.is_validation:
                break

        if (not self.mlm_done) and (self.mlm_dataloader is not None):
            while True:
                try:
                    yield {"mlm": next(self.mlm_dataloader)}
                except StopIteration:
                    return


class MultitaskGLUECollator(GLUECollator):
    def __init__(self, tokenizer, params):
        super().__init__(tokenizer, params)
        self.mlm_collator = DataCollatorForLanguageModeling(self.tokenizer)
        self.tokenizer_params["return_special_tokens_mask"] = True

    def __call__(self, x):
        features, labels = super().__call__(x)
        (
            features["input_ids"],
            features["labels"],
        ) = self.mlm_collator.torch_mask_tokens(features["input_ids"], features.pop("special_tokens_mask", None))
        return features, labels
