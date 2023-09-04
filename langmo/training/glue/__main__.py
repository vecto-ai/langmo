import sys

from .data import GLUEDataModule
from .model import GLUEFinetuner, GLUEModel

name_task = sys.argv[2]


def main():
    finetuner = GLUEFinetuner(name_task, GLUEDataModule, GLUEModel)
    finetuner.run()


if __name__ == "__main__":
    main()
