import lightning
import polars as pl
import torch
from torchmetrics.classification import MulticlassAUROC

from .config import Index
from .dataset import ABCDDataModule
from .model import Network


def predict(
    trainer: lightning.Trainer,
    model: Network,
    data_module: ABCDDataModule,
    checkpoint_path: str,
    ignore_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Test-set logits (N, C) and labels (N,) of observed steps, in test-set row order."""
    batches = trainer.predict(
        model, dataloaders=data_module.test_dataloader(), ckpt_path=checkpoint_path
    )
    if batches is None:
        raise ValueError("trainer.predict returned no batches.")
    outputs = torch.cat([outputs.flatten(0, 1) for outputs, _ in batches])
    labels = torch.cat([labels.flatten() for _, labels in batches])
    observed = labels.ne(ignore_index)
    return outputs[observed], labels[observed]


def make_predictions(
    test: pl.DataFrame, index: Index, outputs: torch.Tensor, labels: torch.Tensor
) -> pl.DataFrame:
    """Test-set identifiers and labels with the class logits of each row as `output`."""
    frame = test.select(index.sample_id, index.event, index.previous_label, index.label)
    if not torch.equal(frame[index.label].to_torch(), labels):
        raise ValueError("Predictions are out of order relative to the test-set rows.")
    return frame.with_columns(pl.Series("output", outputs.tolist()))


def auroc(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """One-vs-rest AUROC per class."""
    metric = MulticlassAUROC(num_classes=outputs.shape[-1], average="none")
    return metric(outputs, labels)
