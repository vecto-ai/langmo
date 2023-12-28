import sys

from .data import GLUEDataModule
from .model import GLUEModel
from protonn.experiment import Experiment
from .config import GLUEConfig
from transformers import AutoTokenizer
from langmo.nn import create_net
from langmo.trainer import get_trainer
from langmo.callbacks.model_snapshots_schedule import FinetuneMonitor


class GLUEExperiment(Experiment):
    def __init__(self):
        super().__init__(name_task=sys.argv[2])
        # def __init__(self, name_task, class_data_module, class_model, config_type=ConfigFinetune):
        # TODO: refactor this into sub-methods
        self.params = GLUEConfig(
            name_task=self.name_task,
            is_master=self.is_master,
        )
        # config_type(name_task, is_master=cluster_env.is_master)
        self.cluster_env.barrier()
        # timestamp = get_time_str()
        self.tokenizer = AutoTokenizer.from_pretrained(self.params["tokenizer_name"])
        class_model = GLUEModel
        class_data_module = GLUEDataModule
        # wandb_logger.watch(net, log='gradients', log_freq=100)
        # embs = vecto.embeddings.load_from_dir(params["path_embeddings"])
        # bottom = AutoModel.from_pretrained(name_model)
        # net = Siamese(bottom, TopMLP2())
        # TODO

        # NOTE: create_net can't be called before
        # self.tokenizer is created
        self.net, name_run = create_net(self.params)
        # TODO: implement this by creating model from config vs pretrained
        if self.params["randomize"]:
            reinit_model(self.net)
            name_run += "_RND"
        name_run += f"_{'↓' if self.params['uncase'] else '◯'}_{self.params['timestamp'][:-3]}"
        self.params["name_run"] = name_run
        self.trainer = get_trainer(self.params, self.cluster_env, extra_callbacks=[FinetuneMonitor()])
        self.params["cnt_workers"] = self.trainer.world_size
        self.model = class_model(self.net, self.tokenizer, self.params)
        self.maybe_randomize_special_tokens()

        self.data_module = class_data_module(
            self.cluster_env,
            self.tokenizer,
            params=self.params,
        )
        # TODO: fix finetune monitor
        # TODO: Please use the DeviceStatsMonitor callback directly instead.
        # TODO: sync_batchnorm: bool = False, to params

    # TODO: move it into some model helpers place
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
        # TODO: name_task should be renamed to glue_task
        # finetuner = GLUEFinetuner(name_task, GLUEDataModule, GLUEModel)
        # finetuner.run()
        self.trainer.fit(self.model, self.data_module)


if __name__ == "__main__":
    experiment = GLUEExperiment()
    experiment.run()
