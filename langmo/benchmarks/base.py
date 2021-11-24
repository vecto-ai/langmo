import horovod.torch as hvd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from langmo.base import PLBase
from langmo.benchmarks.NLI.model import (BertWithCLS, BertWithLSTM, Siamese,
                                         TopMLP2)
from langmo.config import ConfigFinetune as Config
from langmo.nn.utils import reinit_model, reinit_tensor
from protonn.utils import get_time_str
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
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
        loss = F.cross_entropy(logits, targets)
        acc = accuracy(F.softmax(logits, dim=1), targets)
        metrics = {
            "train_loss": loss,
            "train_acc": acc,
        }
        self.log_dict(metrics, on_step=True, on_epoch=True)
        return loss


class BaseFinetuner:
    def __init__(self, name_task, class_data_module, class_model):
        # TODO: refactor this into sub-methods
        hvd.init()
        if hvd.rank() != 0:
            tr_logging.set_verbosity_error()  # to reduce warning of unused weights
        self.params = Config(name_task=name_task, is_master=(hvd.rank() == 0))
        timestamp = get_time_str()
        self.tokenizer = AutoTokenizer.from_pretrained(self.params["model_name"])

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
            name_wandb_project = (
                self.params["name_project"] + f"_{self.params['suffix']}"
            )
        else:
            name_wandb_project = self.params["name_project"]
        self.model = class_model(self.net, self.tokenizer, self.params)
        self.maybe_randomize_special_tokens()

        if self.params["test"]:
            self.params["cnt_epochs"] = 3
        self.data_module = class_data_module(
            # embs.vocabulary,
            self.tokenizer,
            params=self.params,
        )
        lr_monitor = LearningRateMonitor(logging_interval="step")
        self.wandb_logger = WandbLogger(
            project=name_wandb_project,
            name=name_run,
            save_dir=self.params["path_results"],
        )
        self.trainer = pl.Trainer(
            default_root_dir=self.params["path_results"],
            weights_save_path=self.params["path_results"],
            gpus=1,
            num_sanity_val_steps=-1,
            # num_sanity_val_steps=0,
            max_epochs=self.params["cnt_epochs"],
            strategy="horovod",
            precision=self.params["precision"],
            replace_sampler_ddp=False,
            # early_stop_callback=early_stop_callback,
            # we probably don't need to checkpoint eval - but can make this optional
            callbacks=[lr_monitor],  # on_n_step_callback
            checkpoint_callback=False,
            logger=self.wandb_logger,
            progress_bar_refresh_rate=0,
            gradient_clip_val=1.0,
            track_grad_norm=2,
            terminate_on_nan=True,
        )
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
        return net, name_run


class QAFinetuner(BaseFinetuner):
    def create_net(self):
        name_model = self.params["model_name"]
        net = AutoModelForQuestionAnswering.from_pretrained(name_model)
        return net, name_model


def aggregate_batch_stats(batch_stats, key):
    if key in batch_stats[0]:
        value = torch.stack([x[key] for x in batch_stats]).sum()
    else:
        value = torch.tensor(0)
    # print("reducing", key, value)
    value = hvd.allreduce(value, average=False)
    return value
