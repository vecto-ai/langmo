import pytorch_lightning as pl
import torch
from langmo.callbacks.layernorm import LayerNormCallback
from langmo.callbacks.monitor import Monitor
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger


def get_trainer(params):
    # use 1 GPU with horovod and -1 with DDP
    # if (da.rank() != 0):
    #     params["path_results"] = "/tmp"
    # cluster_env = MPIClusterEnvironment()
    # gpus = [cluster_env.local_rank()] if params["use_gpu"] else 0
    if params["use_gpu"]:
        assert torch.cuda.device_count() > 0, "Asked for `use_gpu` but no gpu detected"
    lr_monitor = LearningRateMonitor(logging_interval="step")
    gpus = 1 if params["use_gpu"] else 0
    trainer = pl.Trainer(
        default_root_dir=params["path_results"],
        weights_save_path=params["path_results"],
        gpus=gpus,
        # num_nodes=cluster_env.cnt_nodes(),
        num_sanity_val_steps=0 if "resume" in params else -1,
        max_epochs=params["cnt_epochs"],
        strategy="horovod",
        precision=params["precision"],
        replace_sampler_ddp=False,
        logger=WandbLogger(
            project=params["name_project"],
            name=params["name_run"],
            save_dir=params["path_results"],
        ),
        log_every_n_steps=params["log_every_n_steps"],
        reload_dataloaders_every_n_epochs=0,
        # TODO: is this ok?
        # theirs samples do like you did
        # but there is special checkpoint_callback param too....
        callbacks=[lr_monitor, LayerNormCallback(), Monitor()],
        gradient_clip_val=params["gradient_clip_val"],
        enable_progress_bar=False,
        enable_checkpointing=False,
        # TODO: figure out what is this
        track_grad_norm=1,
        # detect_anomaly=True, # This is very slow!
        # profiler="simple",
        # plugins="deepspeed_stage_2",
        # plugins=[cluster_env],
        accumulate_grad_batches=params["accumulate_batches"],
    )
    return trainer
