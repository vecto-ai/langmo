from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from langmo.base import PLBase
from langmo.benchmarks.NLI.model import (BertWithCLS, BertWithLSTM, Siamese,
                                         TopMLP2)
from langmo.cluster_mpi import MPIClusterEnvironment
from langmo.config import ConfigFinetune as Config
from langmo.nn.utils import reinit_model, reinit_tensor
from langmo.trainer import get_trainer
from protonn.utils import get_time_str
from torchmetrics.functional import accuracy
from transformers import (AutoModel, AutoModelForQuestionAnswering,
                          AutoModelForSequenceClassification, AutoTokenizer)
from transformers import logging as tr_logging


class BaseClassificationModel(PLBase):
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

    def _compute_metric(self, logits, targets):
        return accuracy(F.softmax(logits, dim=1), targets)

    def _compute_loss(self, logits, targets):
        return F.cross_entropy(logits, targets)


class BaseFinetuner:
    def __init__(self, name_task, class_data_module, class_model):
        # TODO: refactor this into sub-methods
        # TODO: and this da is over-complicated
        cluster_env = MPIClusterEnvironment()
        # da.init("horovod")
        if cluster_env.global_rank() != 0:
            tr_logging.set_verbosity_error()  # to reduce warning of unused weights
        self._init_params(name_task)
        if cluster_env.global_rank() == 0:
            path_wandb = Path(self.params["path_results"]) / "wandb"
            path_wandb.mkdir(parents=True, exist_ok=True)
        cluster_env.barrier()
        timestamp = get_time_str()
        self.tokenizer = AutoTokenizer.from_pretrained(self.params["tokenizer_name"])

        # wandb_logger.watch(net, log='gradients', log_freq=100)
        # embs = vecto.embeddings.load_from_dir(params["path_embeddings"])
        # bottom = AutoModel.from_pretrained(name_model)
        # net = Siamese(bottom, TopMLP2())
        self.net, name_run = self.create_net()
        if self.params["randomize"]:
            reinit_model(self.hparamsnet)
            name_run += "_RND"
        name_run += f"_{'↓' if self.params['uncase'] else '◯'}_{timestamp[:-3]}"
        if "suffix" in self.params:
            self.params["name_project"] += f"_{self.params['suffix']}"
        self.params["name_run"] = name_run
        self.model = class_model(self.net, self.tokenizer, self.params)
        self.maybe_randomize_special_tokens()

        if self.params["test"]:
            self.params["cnt_epochs"] = 3
        self.data_module = class_data_module(
            # embs.vocabulary,
            self.tokenizer,
            params=self.params,
        )
        self.trainer = get_trainer(self.params, cluster_env)
        # TODO: Please use the DeviceStatsMonitor callback directly instead.
        # TODO: sync_batchnorm: bool = False, to params

    def _init_params(self, name_task):
        self.params = Config(name_task=name_task)

    def maybe_randomize_special_tokens(self):
        if "rand_tok" in self.params:
            rand_tok = self.params["rand_tok"]
            id_dict = {
                "cls": self.tokenizer.cls_token_id,
                "sep": self.tokenizer.sep_token_id,
            }
            for tok in rand_tok:
                tok_id = id_dict[tok]
                # with torch.no_grad:
                tok_emb = self.net.get_input_embeddings().weight[tok_id]
                reinit_tensor(tok_emb)

    def run(self):
        self.trainer.fit(self.model, self.data_module)


class ClassificationFinetuner(BaseFinetuner):
    def create_net(self):
        name_model = self.params["model_name"]
        if self.params["siamese"]:
            name_run = "siam_" + self.params["encoder_wrapper"] + "_"
            if self.params["freeze_encoder"]:
                name_run += "fr_"
            name_run += name_model
            encoder = AutoModel.from_pretrained(name_model, num_labels=3)
            encoder = BertWithCLS(encoder, freeze=self.params["freeze_encoder"])
            net = Siamese(encoder, TopMLP2(in_size=encoder.get_output_size() * 4))
        else:
            net = AutoModelForSequenceClassification.from_pretrained(
                name_model, num_labels=3
            )
            name_run = name_model.split("pretrain")[-1]
            # TODO: this should be done in pretraining!!!!!
            # net.bert.embeddings.token_type_embeddings.weight.data = torch.zeros_like(net.bert.embeddings.token_type_embeddings.weight)

        return net, name_run


class QAFinetuner(BaseFinetuner):
    def create_net(self):
        name_model = self.params["model_name"]
        net = AutoModelForQuestionAnswering.from_pretrained(name_model)
        return net, name_model


def allreduce(tensor: torch.Tensor, op: Optional[int] = None) -> torch.Tensor:
    if op is None:
        dist.all_reduce(tensor)
        tensor /= dist.get_world_size()
    else:
        # print(tensor)
        dist.all_reduce(tensor, op=op)
    return tensor


def aggregate_batch_stats(batch_stats, key):
    if key in batch_stats[0]:
        value = torch.stack([x[key] for x in batch_stats]).sum()
    else:
        value = torch.tensor(0)
    # print("reducing", key, value)
    if torch.cuda.is_available():
        value = value.cuda()
    value = allreduce(value, op=dist.ReduceOp.SUM)
    return value.item()
