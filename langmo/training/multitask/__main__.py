from .config import ConfigMultitask
from .data import MultitaskDataModule
from .model import MultitaskModule, MultitaskFinetuner


def main():

    finetuner = MultitaskFinetuner(
        name_task="multitask",
        class_data_module=MultitaskDataModule,
        class_model=MultitaskModule,
        config_type=ConfigMultitask,
    )
    finetuner.run()


if __name__ == "__main__":
    main()
