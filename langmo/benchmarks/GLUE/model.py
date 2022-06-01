import torch.nn.functional as F
import torch
import math
import torchmetrics
from torchmetrics.functional import accuracy, pearson_corrcoef

from langmo.config import GLUEConfig
from langmo.base import PLBase
from langmo.benchmarks.base import BaseFinetuner, allreduce
from transformers import AutoModelForSequenceClassification


class GLUEModel(PLBase):
    def __init__(self, net=None, tokenizer=None, params=None):
        super().__init__(net, tokenizer, params)
        num_labels = self.hparams["num_labels"]
        if num_labels == 1:
            self.metrics = {
                "pearson_corr": torchmetrics.PearsonCorrCoef(),
                "spearman_corr": torchmetrics.SpearmanCorrCoef(),
            }
        elif num_labels > 1:
            self.metrics = {
                "accuracy": torchmetrics.Accuracy(num_classes=num_labels),
                "f1": torchmetrics.F1Score(num_classes=num_labels),
                "matthews_corr": torchmetrics.MatthewsCorrCoef(num_classes=num_labels),
            }

    def forward(self, inputs):
        return self.net(**inputs)["logits"]

    def training_step(self, batch, batch_idx):
        inputs, targets = batch[0]
        # 0 is there seince PL returns tuple of batched from all dataloaders
        # not sure if this will be persisten behavior
        logits = self(inputs)
        loss = self._compute_loss(logits, targets)
        acc = self._compute_metric(logits, targets)
        metrics = {
            "train_loss": loss,
            "train_acc": acc,
        }
        self.log_dict(metrics, on_step=True, on_epoch=True)
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
            return accuracy(torch.argmax(logits, 1), targets)

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        logits = self(inputs)
        loss = self._compute_loss(logits, targets)
        metrics = {"val_loss": loss}
        for _, metric in self.metrics.items():

            # TODO: This could probably be done in a better place
            metric.to(self.device)
            metric.update(logits, targets)

        return metrics

    def validation_epoch_end(self, outputs):
        print("### validation epoch end")
        metrics = self.hparams["train_logs"][-1]
        # self.add_epoch_id_to_metrics(metrics)
        loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        loss = allreduce(loss)
        metrics["val_loss"] = loss.item()
        for key, metric in self.metrics.items():
            computed = metric.compute().item()
            metrics["val_" + key] = computed if math.isfinite(computed) else 0
            metric.reset()

        self.log_dict(metrics)
        # TODO: make sure metrics are logged
        # self.save_metrics_and_model(metrics)

class GLUEFineTuner(BaseFinetuner):
    def _init_params(self, name_task):
        self.params = GLUEConfig(name_task)

    def create_net(self):
        name_model = self.params["model_name"]
        net = AutoModelForSequenceClassification.from_pretrained(
            name_model, num_labels=self.params["num_labels"]
        )
        name_run = name_model.split("pretrain")[-1]
        # TODO: this should be done in pretraining!!!!!
        # net.bert.embeddings.token_type_embeddings.weight.data = torch.zeros_like(net.bert.embeddings.token_type_embeddings.weight)

        return net, name_run
