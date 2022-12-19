from langmo.config.base import ConfigFinetune

QATASKS = ["squad", "squad_v2"]


class QAConfig(ConfigFinetune):
    def set_defaults(self):
        super().set_defaults()
        self.defaults["max_answer_length"] = 30
        self.defaults["n_best"] = 20
        self.defaults["stride"] = 50
        self.defaults["num_labels"] = 2

    def _validate(self):
        if self["name_task"] not in QATASKS:
            raise Exception(
                f"Question answering does not support task: {self.name_task}.\n"
                f"Supported tasks are: {', '.join(QATASKS)}"
            )
