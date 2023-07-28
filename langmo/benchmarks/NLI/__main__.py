from langmo.training.base import ClassificationFinetuner

from .data import NLIDataModule
from .model import NLIModel


def main():
    name_task = "NLI"
    finetuner = ClassificationFinetuner(name_task, NLIDataModule, NLIModel)
    finetuner.run()


if __name__ == "__main__":
    raise UserWarning(f"{__file__} shouldn't be executed directly (it's deprecated)")
