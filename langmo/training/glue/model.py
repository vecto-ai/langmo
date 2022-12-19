import json
import math

import torch
import torch.nn.functional as F
import torchmetrics
from langmo.training.base import (BaseClassificationModel,
                                    ClassificationFinetuner)
from langmo.config import GLUEConfig
from langmo.utils.distributed import allreduce
from torchmetrics.functional import accuracy, pearson_corrcoef


class GLUEModel(BaseClassificationModel):
    def __init__(self, net=None, tokenizer=None, params=None):
        super().__init__(net, tokenizer, params)
        num_labels = self.hparams["num_labels"]
        self.val_epoch_metrics = []
        for _ in params["validation_split_names"]:
            if num_labels == 1:
                self.val_epoch_metrics.append({
                    "pearson_corr": torchmetrics.PearsonCorrCoef(),
                    "spearman_corr": torchmetrics.SpearmanCorrCoef(),
                    # TODO: if accuracy now supports binary, why don't we use for 1 label as well
                })
            elif num_labels > 1:
                self.val_epoch_metrics.append({
                    "accuracy": torchmetrics.Accuracy(num_classes=num_labels, task="multiclass"),
                    "f1": torchmetrics.F1Score(num_classes=num_labels, task="multiclass"),
                    "matthews_corr": torchmetrics.MatthewsCorrCoef(num_classes=num_labels, task="multiclass"),
                })
        self.validation_split_names = params["validation_split_names"]

    def training_step(self, batch, batch_idx):
        inputs, targets = batch[0]
        # 0 is there seince PL returns tuple of batched from all dataloaders
        # not sure if this will be persisten behavior
        logits = self(inputs)
        loss = self._compute_loss(logits, targets)
        if self.hparams["num_labels"] == 1:
            logits, targets = logits.reshape(-1), targets.reshape(-1)
        acc = self._compute_metric(logits, targets)
        step_metrics = {
            "train_loss": loss,
            "train_acc": acc,
        }
        self.log_dict(step_metrics, on_step=True, on_epoch=True)
        return loss

    def _compute_loss(self, logits, targets):
        if self.hparams["num_labels"] > 1:
            return F.cross_entropy(logits, targets)
        else:
            return F.mse_loss(logits, targets)

    def _compute_metric(self, logits, targets):
        if self.hparams["num_labels"] == 1:
            return pearson_corrcoef(logits, targets)
        elif self.hparams["num_labels"] > 1:
            return accuracy(preds=torch.argmax(logits, -1),
                            target=targets,
                            task="multiclass",
                            num_classes=self.hparams["num_labels"])

    # TODO: reuse logic in  test step if we do that
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        inputs, targets = batch
        logits = self(inputs)
        if self.hparams["save_predictions"]:
            for i in range(len(logits)):
                sample_details = {}
                sample_details["logits"] = logits[i].tolist()
                sample_details["input_ids"] = inputs["input_ids"][i].tolist()
                sample_details["targets"] = targets[i].item()
                self.files_predictions[dataloader_idx].write(json.dumps(sample_details))
                self.files_predictions[dataloader_idx].write("\n")
        for _, metric in self.val_epoch_metrics[dataloader_idx].items():
            # TODO: This could probably be done in a better place
            metric.to(self.device)
            if self.hparams["num_labels"] == 1:
                logits, targets = logits.reshape(-1), targets.reshape(-1)
            metric.update(logits, targets)
        split_name = self.validation_split_names[dataloader_idx]
        loss = self._compute_loss(logits, targets)
        metrics_step = {f"{split_name}_loss": loss}
        return metrics_step

    def validation_epoch_end(self, outputs):
        print("### validation epoch end")
        metrics = self.hparams["train_logs"][-1]
        # self.add_epoch_id_to_metrics(metrics)
        if not isinstance(outputs[0], list):
            outputs = [outputs]
        for id_dataloader, outputs_per_split in enumerate(outputs):
            name_split = self.validation_split_names[id_dataloader]
            loss = torch.stack([x[f"{name_split}_loss"] for x in outputs_per_split]).mean()
            loss = allreduce(loss)
            metrics[f"{name_split}_loss"] = loss.item()
            for key, metric in self.val_epoch_metrics[id_dataloader].items():
                computed = metric.compute().item()
                metrics[f"{name_split}_" + key] = computed if math.isfinite(computed) else 0
                metric.reset()

        self.log_dict(metrics)
        # TODO: make sure metrics are logged
        # self.save_metrics_and_model(metrics)


class GLUEFinetuner(ClassificationFinetuner):
    def __init__(self, name_task, class_data_module, class_model):
        super().__init__(name_task, class_data_module, class_model, config_type=GLUEConfig)
