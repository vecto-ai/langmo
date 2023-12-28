from protonn.pl.cluster_mpi import MPIClusterEnvironment
from protonn.utils import get_time_str
from transformers import AutoTokenizer
from transformers import logging as tr_logging

from langmo.base import PLBase
from langmo.callbacks.model_snapshots_schedule import FinetuneMonitor
from langmo.config import ConfigFinetune
from langmo.nn import create_net

# from langmo.nn.heads import get_downstream_head
from langmo.nn.utils import reinit_model, reinit_tensor
from langmo.trainer import get_trainer


class BaseClassificationModel(PLBase):
    def forward(self, inputs):
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
        self.params = config_type(name_task, is_master=cluster_env.is_master)
        if cluster_env.global_rank() != 0:
            tr_logging.set_verbosity_error()  # to reduce warning of unused weights
        cluster_env.barrier()
        timestamp = get_time_str()
        self.tokenizer = AutoTokenizer.from_pretrained(self.params["tokenizer_name"])

        # wandb_logger.watch(net, log='gradients', log_freq=100)
        # embs = vecto.embeddings.load_from_dir(params["path_embeddings"])
        # bottom = AutoModel.from_pretrained(name_model)
        # net = Siamese(bottom, TopMLP2())

        # NOTE: create_net can't be called before
        # self.tokenizer is created
        self.net, name_run = self.create_net()
        if self.params["randomize"]:
            reinit_model(self.net)
            name_run += "_RND"
        name_run += f"_{'↓' if self.params['uncase'] else '◯'}_{timestamp[:-3]}"
        self.params["name_run"] = name_run
        self.model = class_model(self.net, self.tokenizer, self.params)
        self.maybe_randomize_special_tokens()

        self.data_module = class_data_module(
            cluster_env,
            self.tokenizer,
            params=self.params,
        )
        # TODO: fix finetune monitor
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
        return create_net(self.params)
