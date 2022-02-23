import sys
from pathlib import Path

from protonn.distributed import dist_adapter as da
from protonn.utils import load_json
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

from .data import TextDataModule
from .plmodel import PLModel
from .trainer import get_trainer
from langmo.config import ConfigResume as Config


def load_model_from_checkpoint(path, params):
    # TODO: consider checkpoint files being moved
    tokenizer = AutoTokenizer.from_pretrained(
        path / "checkpoints" / "ep_-1_smpl_000" / "hf"
    )
    config = AutoConfig.from_pretrained(path / "checkpoints" / "ep_-1_smpl_000" / "hf")
    net = AutoModelForMaskedLM.from_config(config)
    net.train()
    model = PLModel.load_from_checkpoint(
        path / "PL_model.ckpt",
        net=net,
        tokenizer=tokenizer,
        params=params,
    )
    return model


def main():
    print("RESUMING")
    # TODO: this shoul be gone when we move to DDP
    da.init("horovod")
    path = Path(sys.argv[1])
    params = load_json(path / "metadata.json")
    if len(sys.argv) > 2:
        replaced_param_path = Path(sys.argv[2])
        replaced_params = Config(name_task="pretrain", old_params=params, param_path=replaced_param_path)
        params.update(replaced_params)
    model = load_model_from_checkpoint(path, params)
    trainer = get_trainer(params)
    data_module = TextDataModule(
        tokenizer=model.tokenizer,
        params=params,
    )
    trainer.fit(model, data_module, ckpt_path=path / "PL_model.ckpt")
    print("Training done")
    # TODO: support overwriting params
    # TODO: nake sure we are not running out of memory when deserializing models to same GPU
    # TODO: any way to resume on WANDB?


if __name__ == "__main__":
    main()
