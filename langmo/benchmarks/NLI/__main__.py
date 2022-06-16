from langmo.benchmarks.base import ClassificationFinetuner
from .data import NLIDataModule
from .model import NLIModel


def main():
    name_task = "NLI"
    finetuner = ClassificationFinetuner(name_task, NLIDataModule, NLIModel)
    finetuner.run()


if __name__ == "__main__":
    main()
