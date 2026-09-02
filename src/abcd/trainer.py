from pathlib import Path

import lightning
from lightning.pytorch.callbacks import ModelCheckpoint

from .config import Trainer


def make_trainer(
    cfg: Trainer, checkpoints: Path
) -> tuple[lightning.Trainer, ModelCheckpoint]:
    """Trainer that keeps only the checkpoint with the lowest validation loss."""
    checkpoint = ModelCheckpoint(
        dirpath=checkpoints,
        filename="best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        enable_version_counter=False,
    )
    trainer = lightning.Trainer(
        logger=False, callbacks=[checkpoint], devices=1, **cfg.model_dump()
    )
    return trainer, checkpoint
