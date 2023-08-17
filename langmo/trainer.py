import os

import lightning as pl
import lightning_utilities
import torch
from langmo.logger_dummy import DummyLogger
from lightning.pytorch.callbacks import GradientAccumulationScheduler
# from pytorch_lightning.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy

# from langmo.callbacks.layernorm import LayerNormCallback
# from langmo.callbacks.monitor import Monitor


def get_trainer(params, cluster_env, extra_callbacks):
    # use 1 GPU with horovod and -1 with DDP
    # if (da.rank() != 0):
    #     params["path_results"] = "/tmp"
    # gpus = [cluster_env.local_rank()] if params["use_gpu"] else 0
    # print(f"### trying to use gpus: {gpus} ")
    extra_callbacks.append(GradientAccumulationScheduler(params["accumulate_batches"]))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if params["cnt_gpus_per_node"] > 0:
        assert torch.cuda.device_count() > 0, "Asked for GPUs but none detected"
    # lr_monitor = LearningRateMonitor(logging_interval="step")
    # TODO: check if making only local GPU visible makes init faster
    # gpus = [int(os.environ["RANK"])] if params["use_gpu"] else 0
    # gpus = -1 if (params["cnt_gpus_per_node"] > 0) else 0
    lightning_utilities.core.rank_zero.rank_zero_only.rank = cluster_env.global_rank()
    if "WANDB_MODE" not in os.environ or os.environ["WANDB_MODE"].lower() != "disabled":
        logger = WandbLogger(
            project=params["name_project"],
            name=params["name_run"],
            save_dir=params["path_results"],
        )
    else:
        logger = DummyLogger()
    strategy = DDPStrategy(**params["ddp_strategy_params"])
    trainer = pl.Trainer(
        strategy=strategy,
        plugins=[cluster_env],
        default_root_dir=params["path_results"],
        devices=params["devices"],
        accelerator=params["accelerator"],
        # num_nodes=int(os.environ["CNT_NODES"]),  # cluster_env.cnt_nodes(),
        num_nodes=cluster_env.cnt_nodes(),
        num_sanity_val_steps=0 if "resume" in params else params["num_sanity_val_steps"],
        max_epochs=params["cnt_epochs"],
        precision=params["precision"],
        use_distributed_sampler=False,
        logger=logger,
        log_every_n_steps=params["log_every_n_steps"],
        reload_dataloaders_every_n_epochs=0,
        # TODO: is this ok?
        # theirs samples do like you did
        # but there is special checkpoint_callback param too....
        # callbacks=[lr_monitor, LayerNormCallback(), Monitor()],
        # TODO: layernorm doesn't work with albert
        callbacks=extra_callbacks,
        gradient_clip_val=params["gradient_clip_val"],
        enable_progress_bar=False,
        enable_checkpointing=False,
        # TODO: figure out what is this
        # track_grad_norm=1, # TODO: https://lightning.ai/pages/releases/2.0.0/
        # detect_anomaly=True, # This is very slow!
        # profiler="simple",
        # plugins="deepspeed_stage_2",
        # plugins=[cluster_env],
        # accumulate_grad_batches=,
    )
    return trainer
