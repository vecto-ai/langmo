import sys
from pathlib import Path

from protonn.pl.cluster_mpi import MPIClusterEnvironment
from protonn.utils import load_json
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

from langmo.callbacks.model_snapshots_schedule import Monitor

# from langmo.config import ConfigResume as Config
from langmo.trainer import get_trainer

from ..base_data import TextDataModule
from .data import BatchIter
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
    # This will be overwritten on actual loading state upon trainer.fit
    # but we need to let monitor know that we are in resume
    print("loaded params")
    print(params)
    model.hparams.update(params)
    model.hparams["is_resume"] = True  # this entry is removed right away in monitor setup
    print(model.hparams)
    return model


def fix_batch_accumulation_schedule(schedule):
    # because integer keys in json became strings 🤦
    keys = list(schedule)
    for k in keys:
        v = schedule.pop(k)
        schedule[int(k)] = v


def main():
    cluster_env = MPIClusterEnvironment()
    if cluster_env.local_rank() == 0:
        print("RESUMING")
    path = Path(sys.argv[1])
    params = load_json(path / "resume" / "metadata.json")
    fix_batch_accumulation_schedule(params["accumulate_batches"])
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
    print("model loaded")
    # TODO: this doesn't load custom callbacks
    # but ideally this code should somehow not be duplicated
    trainer = get_trainer(params, cluster_env, [Monitor()])
    print("trainer created")
    # ok we remove this because train epoch starts adds a new one... but we shouldn't
    # model.hparams["train_logs"] = model.hparams["train_logs"][:-1]
    data_module = TextDataModule(
        batch_iterator_cls=BatchIter,
        cluster_env=cluster_env,
        tokenizer=model.tokenizer,
        params=params,
    )
    # actual loading of model state inc hparams happens here
    trainer.fit(model, data_module, ckpt_path=path / "resume" / "PL_model.ckpt")
    print("Training done")
    # TODO: what if e.g. cnt_workers changed
    # TODO: check if cnt_gpus_per_node is same
    # TODO: nake sure we are not running out of memory when deserializing models to same GPU
    # TODO: any way to resume on WANDB?


if __name__ == "__main__":
    main()
