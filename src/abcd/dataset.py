from functools import partial

import polars as pl
import torch
from lightning import LightningDataModule
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from .config import Dataloader, Index

type Sample = tuple[torch.Tensor, torch.Tensor]


def make_sequences(frame: pl.DataFrame, index: Index) -> list[Sample]:
    """One (inputs, labels) pair per subject; the event index stays in as a feature."""
    sequences = []
    for subject in frame.partition_by(index.sample_id, maintain_order=True):
        features = subject.drop(index.sample_id, index.previous_label, index.label)
        inputs = features.to_torch(dtype=pl.Float32)  # (T, F)
        labels = subject[index.label].to_torch()  # (T,)
        sequences.append((inputs, labels))
    return sequences


def collate(batch: list[Sample], ignore_index: int) -> Sample:
    inputs, labels = zip(*batch)
    padded_inputs = pad_sequence(list(inputs), batch_first=True)  # (B, T, F)
    padded_labels = pad_sequence(
        list(labels), batch_first=True, padding_value=ignore_index
    )  # (B, T)
    return padded_inputs, padded_labels


class ABCDDataModule(LightningDataModule):
    def __init__(
        self, splits: dict[str, pl.DataFrame], index: Index, dataloader: Dataloader
    ) -> None:
        super().__init__()
        self.train = make_sequences(splits["train"], index=index)
        self.val = make_sequences(splits["val"], index=index)
        self.test = make_sequences(splits["test"], index=index)
        self.loader = partial(
            DataLoader,
            batch_size=dataloader.batch_size,
            num_workers=dataloader.num_workers,
            collate_fn=partial(collate, ignore_index=dataloader.ignore_index),
        )

    @property
    def input_dim(self) -> int:
        return self.train[0][0].shape[-1]

    def train_dataloader(self) -> DataLoader:
        return self.loader(self.train, shuffle=True, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        return self.loader(self.val)

    def test_dataloader(self) -> DataLoader:
        return self.loader(self.test)
