from protonn.experiment import Experiment
from transformers import AutoTokenizer
from langmo.nn import create_net
from langmo.trainer import get_trainer
from langmo.callbacks.model_snapshots_schedule import FinetuneMonitor


class BaseExperiment(Experiment):
    def __init__(self, name_task, class_model, class_data_module, class_config):
        super().__init__(name_task)
        self.params = class_config(
            name_task=self.name_task,
            is_master=self.is_master,
        )
        self.maybe_create_unique_path()
        self.net, self.name_run = self.create_net()
        self.params["name_run"] = self.name_run
        if self.params["randomize"]:
            reinit_model(self.net)
            self.params["name_run"] += "_RND"
        self.params["name_run"] += f"_{'↓' if self.params['uncase'] else '◯'}_{self.params['timestamp'][:-3]}"
        self.tokenizer = AutoTokenizer.from_pretrained(self.params["tokenizer_name"])
        self.trainer = get_trainer(self.params, self.cluster_env, extra_callbacks=[FinetuneMonitor()])
        self.params["cnt_workers"] = self.trainer.world_size
        self.model = class_model(self.net, self.tokenizer, self.params)
        self.data_module = class_data_module(
            self.cluster_env,
            self.tokenizer,
            params=self.params,
        )

    def run(self):
        self.trainer.fit(self.model, self.data_module)


class BaseFinetuneExperiment(BaseExperiment):
    def create_net(self):
        return create_net(self.params)
