import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import ModuleDict
import torchmetrics
from transformers import AutoModelForMaskedLM, AutoModel

from langmo.base import PLBase
from langmo.utils.distributed import allreduce
from langmo.benchmarks.base import BaseFinetuner

from .config import TaskConfigs


class MultitaskFinetuner(BaseFinetuner):
    def create_net(self):
        if "mlm" in self.params["tasks"]:
            return (
                AutoModelForMaskedLM.from_pretrained(self.params["model_name"]),
                "with_mlm",
            )
        else:
            return (
                AutoModel.from_pretrained(self.params["model_name"]),
                "downstream_only",
            )


class MultitaskModule(PLBase):
    def __init__(self, net, tokenizer, params):
        super().__init__(net, tokenizer, params)
        self.params = TaskConfigs(params)
        self.tasks_params = self.params["tasks"]
        self.heads = ModuleDict(
            {
                task_name: ClassificationHead(
                    task_spec["head_config"],
                    output_size=self.net.config.hidden_size,
                )
                for task_name, task_spec in self.tasks_params.items()
                if task_name != "mlm"
            }
        )
        self.val_epoch_metrics = {}
        for task_name, task_spec in self.tasks_params.items():
            if task_name == "mlm":
                continue

            num_labels = task_spec["num_labels"]
            if task_spec["num_labels"] == 1:
                self.val_epoch_metrics[task_name] = {
                    "pearson_corr": torchmetrics.PearsonCorrCoef(),
                    "spearman_corr": torchmetrics.SpearmanCorrCoef(),
                }
            elif task_spec["num_labels"] > 1:
                self.val_epoch_metrics[task_name] = {
                    "accuracy": torchmetrics.Accuracy(num_classes=num_labels),
                    "f1": torchmetrics.F1Score(num_classes=num_labels),
                    "matthews_corr": torchmetrics.MatthewsCorrCoef(
                        num_classes=num_labels
                    ),
                }
        self.there_is_mlm = "mlm" in self.tasks_params

    def forward(self, task_name, x, output_logits=False):
        if self.there_is_mlm:
            if task_name == "mlm":
                return self.net(**x._asdict()).loss, None

            output = self.net(**x[0], output_hidden_states=True)
            mlm_loss = output.loss

            last_hidden_state = self.net(**x[0], output_hidden_states=True).hidden_states[-1]
            fine_tuning_loss = self.heads[task_name](
                last_hidden_state[:, 0, :],
                x[1].to(self.device),
                output_logits=output_logits,
            )
            return mlm_loss, fine_tuning_loss
        else:
            return None, self.heads[task_name](
                self.net(**x[0].to(self.device)).last_hidden_state[:, 0, :],
                x[1].to(self.device),
                output_logits=output_logits,
            )

    def training_step(self, batch_dict, batch_dict_idx):
        losses = {}
        mlm_losses = {}
        for task_name, batch in batch_dict.items():
            mlm_losses[task_name], losses[task_name] = self(task_name, batch)
            with torch.no_grad():
                if losses[task_name] is not None:
                    self.log(
                        f"{task_name}_loss",
                        losses[task_name].clone().detach(),
                        sync_dist=True,
                    )
                if mlm_losses[task_name] is not None:
                    self.log(
                        f"{task_name}_mlm_loss".replace("mlm_mlm", "mlm"),
                        mlm_losses[task_name].clone().detach(),
                        sync_dist=True,
                    )

        loss = sum(
            self.tasks_params[task_name]["loss_coef"] * losses[task_name]
            for task_name in losses
            if task_name != "mlm"
        )

        with torch.no_grad():
            self.log("cumulative_loss", loss.clone().detach(), sync_dist=True)

        if self.there_is_mlm:
            all_mlm_loss = sum(i for i in mlm_losses.values() if i is not None)
            with torch.no_grad():
                self.log(
                    "cumulative_mlm_loss", all_mlm_loss.clone().detach(), sync_dist=True
                )
            loss += all_mlm_loss

        return loss

    # TODO: reuse logic in  test step if we do that
    def validation_step(self, batch_dict, batch_idx):
        step_metrics = {}
        for task_name in self.tasks_params:
            if task_name == "mlm" or task_name not in batch_dict:
                continue

            batch = batch_dict[task_name]
            _, logits = self(task_name, batch, output_logits=True)

            if self.tasks_params[task_name]["num_labels"] == 1:
                logits, targets = logits.reshape(-1), batch[1].reshape(-1)
            else:
                targets = batch[1]
            loss = self.heads[task_name].loss(logits, targets)
            step_metrics[task_name] = {f"{task_name}_loss": loss}

            for _, compute_metric in self.val_epoch_metrics[task_name].items():
                compute_metric.to(self.device)
                compute_metric.update(logits, targets)
                # if self.hparams["save_predictions"]:
                #     for i in range(len(logits)):
                #         sample_details = {}
                #         sample_details["logits"] = logits[i].tolist()
                #         sample_details["input_ids"] = inputs["input_ids"][i].tolist()
                #         sample_details["targets"] = targets[i].item()
                #         self.files_predictions[dataloader_idx].write(json.dumps(sample_details))
                #         self.files_predictions[dataloader_idx].write("\n")

        return step_metrics

    def validation_epoch_end(self, outputs):
        print("### validation epoch end")
        metrics = self.hparams["train_logs"][-1]
        for task_name in self.tasks_params:
            loss = [
                step_metrics[task_name][f"{task_name}_loss"].clone().detach()
                for step_metrics in outputs
                if task_name in step_metrics
            ]
            if len(loss) == 0:
                continue
            loss = torch.stack(loss).mean()
            loss = allreduce(loss)

            metrics[f"{task_name}_loss"] = loss.item()
            for metric_name, compute_metric in self.val_epoch_metrics[
                task_name
            ].items():
                computed = compute_metric.compute().item()
                metrics[f"{task_name}_{metric_name}"] = (
                    computed if math.isfinite(computed) else 0
                )
                compute_metric.reset()

        self.log_dict(metrics, sync_dist=True)


class ClassificationHead(nn.Module):
    def __init__(self, head_config, output_size):
        super().__init__()
        head_params = head_config
        module_list = [
            nn.Linear(output_size, head_params.get_key("hidden_size")),
            head_params.get_key("activation")(),
            nn.Dropout(head_params.get_key("dropout_p")),
        ]
        for _ in range(head_params.get_key("n_layers") - 1):
            module_list += [
                nn.Linear(
                    head_params.get_key("hidden_size"),
                    head_params.get_key("hidden_size"),
                ),
                head_params.get_key("activation")(),
                nn.Dropout(head_params.get_key("dropout_p")),
            ]
        module_list.append(
            nn.Linear(
                head_params.get_key("hidden_size"), head_params.get_key("num_labels")
            )
        )
        self.head = nn.ModuleList(module_list)
        self.loss = head_params.get_key("loss")()

    def forward(self, x, y, output_logits=False):
        # TODO: can output logits if needed
        for layer in self.head:
            x = layer(x)
        if output_logits:
            return x
        return self.loss(x, y)


# TODO: this is a copy of roberta MLMHead from transformers
# leave it here because ideally I think it would be nicer and more
# solid to have it implemented

# class MLMHead(nn.Module):
#     """Roberta Head for masked language modeling."""

#     def __init__(self, config):
#         super().__init__()
#         self.dense = nn.Linear(config.hidden_size, config.hidden_size)
#         self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

#         self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
#         self.bias = nn.Parameter(torch.zeros(config.vocab_size))
#         self.decoder.bias = self.bias

#     def forward(self, features, **kwargs):
#         x = self.dense(features)
#         x = gelu(x)
#         x = self.layer_norm(x)

#         # project back to size of vocabulary with bias
#         x = self.decoder(x)

#         return x

#     def _tie_weights(self):
#         # To tie those two weights if they get disconnected (on TPU or when the bias is resized)
#         # For accelerate compatibility and to not break backward compatibility
#         if self.decoder.bias.device.type == "meta":
#             self.decoder.bias = self.bias
#         else:
#             self.bias = self.decoder.bias
