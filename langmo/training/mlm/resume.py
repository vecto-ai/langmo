import sys
from pathlib import Path

from langmo.callbacks.model_snapshots_schedule import Monitor
from langmo.cluster_mpi import MPIClusterEnvironment
# from langmo.config import ConfigResume as Config
from langmo.trainer import get_trainer
from protonn.utils import load_json
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

from .data import TextDataModule
from .plmodel import PLModel


def load_model_from_checkpoint(path, params):
    # TODO: consider checkpoint files being moved
    path_hf = path / "checkpoints" / "ep_-01_smpl_0" / "hf"
    tokenizer = AutoTokenizer.from_pretrained(path_hf)
    config = AutoConfig.from_pretrained(path_hf)
    # TODO: assert batch size didn't change or think how to deal with it
    net = AutoModelForMaskedLM.from_config(config)
    net.train()
    model = PLModel.load_from_checkpoint(
        path / "resume" / "PL_model.ckpt",
        net=net,
        tokenizer=tokenizer,
        params=params,
    )
    return model


def main():
    cluster_env = MPIClusterEnvironment()
    if cluster_env.local_rank() == 0:
        print("RESUMING")
    path = Path(sys.argv[1])
    params = load_json(path / "resume" / "metadata.json")
    # TODO: this is semi borken and probably not needed
    # if we need train an alternative version - can just use snapshot as a new model,
    # just force train to not re-init
    # if len(sys.argv) > 2:
    #     replaced_param_path = Path(sys.argv[2])
    #     replaced_params = Config(
    #         name_task="pretrain", old_params=params, param_path=replaced_param_path
    #     )
    #     params.update(replaced_params)
    model = load_model_from_checkpoint(path, params)
    trainer = get_trainer(params, cluster_env, [Monitor()])
    model.hparams["train_logs"] = model.hparams["train_logs"][:-1]
    data_module = TextDataModule(
        cluster_env,
        tokenizer=model.tokenizer,
        params=params,
    )
    trainer.fit(model, data_module, ckpt_path=path / "resume" / "PL_model.ckpt")
    print("Training done")
    # TODO: what if e.g. cnt_workers changed
    # TODO: check if cnt_gpus_per_node is same
    # TODO: nake sure we are not running out of memory when deserializing models to same GPU
    # TODO: any way to resume on WANDB?


if __name__ == "__main__":
    main()
