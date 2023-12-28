"""
Fine-tuning a model on question aswering datasets such as squad
SQuAD2.0 combines the 100,000 questions in SQuAD1.1 with over 50,000
unanswerable questions written adversarially by crowdworkers to look
similar to answerable ones.

Reported metrics are

EM - The Exact Match metric measures the percentage of predictions that match
any one of the ground truth answers exactly.

F1 - = precision * recall / (rpecision + recall), where precision is the ratio of the number of
shared words to the total number of words in the prediction, and recall is the ratio of the
number of shared words to the total number of words in the ground truth
"""
import sys

from langmo.training.base_experiment import BaseExperiment

from .config import QAConfig
from .data import QADataModule as DataModule
from .model import QAModel, QANet


class QAExperiment(BaseExperiment):
    def __init__(self, name_task, class_data_module, class_model, config_type=QAConfig):
        super().__init__(name_task, class_data_module, class_model, config_type)

    def create_net(self):
        name_model = self.params["model_name"]
        # net = AutoModelForQuestionAnswering.from_pretrained(name_model)
        net = QANet(name_model)
        return net, name_model


def main():
    name_task = sys.argv[2]
    finetuner = QAExperiment(name_task, DataModule, QAModel)
    finetuner.run()


if __name__ == "__main__":
    main()
