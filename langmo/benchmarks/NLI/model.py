import torch
import torch.nn.functional as F

from torchmetrics.functional import accuracy

from langmo.benchmarks.base import BaseClassificationModel, aggregate_batch_stats, allreduce
from .data import labels_entail, labels_heuristics


class NLIModel(BaseClassificationModel):
    def __init__(self, net, tokenizer, params):
        super().__init__(net, tokenizer, params)
        # self.example_input_array = ((
        #     torch.zeros((128, params["batch_size"]), dtype=torch.int64),
        #     torch.zeros((128, params["batch_size"]), dtype=torch.int64),
        # ))
        self.ds_prefixes = {0: "matched", 1: "mismatched", 2: "hans"}

    def training_step(self, batch, batch_idx):
        inputs, targets, heuristic = batch[0]
        # this is to fix PL 1.2+ thinking that top level list is multiple iterators
        # should be address by returning proper dataloader
        logits = self(inputs)
        loss = F.cross_entropy(logits, targets)
        acc = accuracy(F.softmax(logits, dim=1), targets)
        # acc = accuracy(logits, targets)
        metrics = {
            "train_loss": loss,
            "train_acc": acc,
        }
        self.log_dict(metrics, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        # print("got val batch\n" + describe_var(batch))
        inputs, targets, heuristic = batch
        logits = self(inputs)
        if dataloader_idx == 2:
            entail = logits[:, :1]
            non_entail = logits[:, 1:]
            non_entail = non_entail.max(axis=1).values
            logits = torch.cat((entail, non_entail.unsqueeze(1)), 1)
        loss = F.cross_entropy(logits, targets)
        # acc = accuracy(torch.nn.functional.softmax(logits, dim=1), targets)
        mask_correct = torch.argmax(logits, axis=1) == targets
        cnt_correct = mask_correct.sum()
        # if self.hparams["test"] and dataloader_idx == 2:
        #     print(
        #         f"worker {da.rank()} of {da.world_size()}\n"
        #         f"\tval batch {batch_idx} ({logits.size()}) of dloader {dataloader_idx}\n"
        #         f"\ttargets: {targets.sum()}, acc is {acc}"
        #     )
        metrics = {
            f"val_loss": loss,
            # f"val_acc": acc,
            f"cnt_correct": cnt_correct,
            f"cnt_questions": torch.tensor(targets.shape[0]),
        }
        if dataloader_idx == 2:
            for entail in [0, 1]:
                for id_heuristic in [0, 1, 2]:
                    # fmt: off
                    split_name = f"{labels_heuristics[id_heuristic]}_{labels_entail[entail]}"
                    mask_split = torch.logical_and((targets == entail), (heuristic == id_heuristic))
                    metrics[f"cnt_correct_{split_name}"] = torch.logical_and(mask_correct, mask_split).sum()
                    metrics[f"cnt_questions_{split_name}"] = mask_split.sum()
                    # fmt: on
        return metrics

    def validation_epoch_end(self, outputs):
        metrics = self.hparams["train_logs"][-1]
        # self.add_epoch_id_to_metrics(metrics)
        for id_dataloader, lst_split in enumerate(outputs):
            name_dataset = self.ds_prefixes[id_dataloader]
            loss = torch.stack([x["val_loss"] for x in lst_split]).mean()  # .item()
            # TODO: refactor this reduction an logging in one helper function
            metrics[f"val_loss_{name_dataset}"] = allreduce(loss, None).item()
            cnt_correct = aggregate_batch_stats(lst_split, "cnt_correct")
            cnt_questions = aggregate_batch_stats(lst_split, "cnt_questions")
            metrics[f"val_acc_{name_dataset}"] = cnt_correct / cnt_questions

            for entail in [0, 1]:
                cnt_correct_label = 0
                cnt_questions_label = 0
                for id_heuristic in [0, 1, 2]:
                    split_name = f"{labels_heuristics[id_heuristic]}_{labels_entail[entail]}"
                    cnt_correct = aggregate_batch_stats(lst_split, f"cnt_correct_{split_name}")
                    cnt_questions = aggregate_batch_stats(lst_split, f"cnt_questions_{split_name}")
                    if cnt_questions > 0:
                        metrics[f"cnt_correct_{name_dataset}_{split_name}"] = cnt_correct
                        metrics[f"val_acc_{name_dataset}_{split_name}"] = (
                            cnt_correct / cnt_questions
                        )
                    cnt_correct_label += cnt_correct
                    cnt_questions_label += cnt_questions
                if cnt_questions_label > 0:
                    metric_name = f"val_acc_{name_dataset}_{labels_entail[entail]}"
                    metric_val = cnt_correct_label / cnt_questions_label
                    metrics[metric_name] = metric_val
                    # self.log(metric_name, metric_val)
            self.log_dict(metrics)
            # cnt_correct = aggregate_batch_stats(lst_split, "cnt_correct")
        # self.save_metrics_and_model(metrics)
