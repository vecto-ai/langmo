import sys

from .data import GLUEDataModule
from .model import GLUEModel, GLUEFineTuner

name_task = sys.argv[2]

def main():
    finetuner = GLUEFineTuner(name_task, GLUEDataModule, GLUEModel)
    finetuner.run()

if __name__ == "__main__":
    main()
