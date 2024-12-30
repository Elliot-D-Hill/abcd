import os

from lightning import Trainer
from lightning.fabric.plugins.environments.slurm import SLURMEnvironment
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    RichProgressBar,
)

# StochasticWeightAveraging,
from lightning.pytorch.loggers import TensorBoardLogger

from abcd.config import Config


def make_callbacks(cfg: Config, checkpoint: bool, callbacks: list | None = None):
    if callbacks is None:
        callbacks = []
    if cfg.trainer.enable_progress_bar:
        callbacks.append(RichProgressBar())
    if checkpoint:
        checkpoint_callback = ModelCheckpoint(
            dirpath=cfg.filepaths.data.results.checkpoints,
            filename="{epoch}_{step}_{val_loss:.4f}",
            save_last=True,
            verbose=cfg.verbose,
            save_top_k=0,
        )
        callbacks.append(checkpoint_callback)
    # swa_callback = StochasticWeightAveraging(swa_lrs=cfg.trainer.swa_lrs)
    # callbacks.append(swa_callback)
    return callbacks


def make_trainer(
    cfg: Config, checkpoint: bool, callbacks: list | None = None
) -> Trainer:
    callbacks = make_callbacks(cfg, checkpoint, callbacks=callbacks)
    logger = TensorBoardLogger(save_dir=cfg.filepaths.data.results.logs)
    logger = logger if cfg.log else False
    num_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", 1))
    plugins = []
    if "SLURM_JOB_ID" in os.environ:
        print(f"Running in SLURM environment with Job ID: {os.environ['SLURM_JOB_ID']}")
        plugins.append(SLURMEnvironment())
    trainer = Trainer(
        logger=logger,
        callbacks=callbacks,
        accelerator="auto",
        devices="auto",
        num_nodes=num_nodes,
        strategy="auto",
        log_every_n_steps=cfg.logging.log_every_n_steps,
        fast_dev_run=cfg.fast_dev_run,
        enable_checkpointing=checkpoint,
        precision="bf16-mixed",
        plugins=plugins,
        **cfg.trainer.model_dump(exclude={"swa_lrs"}),
    )
    return trainer
