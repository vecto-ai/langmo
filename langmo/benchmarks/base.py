from typing import Optional

import torch
import torch.distributed as dist
from langmo.base import PLBase
from langmo.benchmarks.NLI.model import (BertWithCLS, BertWithLSTM, Siamese,
                                         TopMLP2)
from langmo.callbacks.model_snapshots_schedule import FinetuneMonitor
from langmo.cluster_mpi import MPIClusterEnvironment
from langmo.config import ConfigFinetune
from langmo.nn.utils import reinit_model, reinit_tensor
from langmo.trainer import get_trainer
from protonn.utils import get_time_str
from transformers import (AutoModel, AutoModelForQuestionAnswering,
                          AutoModelForSequenceClassification, AutoTokenizer)
from transformers import logging as tr_logging


class BaseClassificationModel(PLBase):
    def forward(self, inputs):
        current_batch_size = inputs["input_ids"].shape[0]
        self.hparams["cnt_samples_processed"] += (
            current_batch_size * self.hparams["cnt_workers"]
        )
        return self.net(**inputs)["logits"]

    # TODO: this seems to be wrong and also not used
    # def _compute_metric(self, logits, targets):
    #     return accuracy(F.softmax(logits, dim=1), targets)

    # def _compute_loss(self, logits, targets):
    #     return F.cross_entropy(logits, targets)


class BaseFinetuner:
    def __init__(self, name_task, class_data_module, class_model, config_type=ConfigFinetune):
        # TODO: refactor this into sub-methods
        cluster_env = MPIClusterEnvironment()
        self.params = config_type(name_task, cluster_env)
        if cluster_env.global_rank() != 0:
            tr_logging.set_verbosity_error()  # to reduce warning of unused weights
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
        self.trainer = get_trainer(self.params, cluster_env, extra_callbacks=[FinetuneMonitor()])
        # TODO: Please use the DeviceStatsMonitor callback directly instead.
        # TODO: sync_batchnorm: bool = False, to params

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
