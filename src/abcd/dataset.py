from functools import partial

import numpy as np
import polars as pl
import torch
from lightning import LightningDataModule
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from abcd.config import Config


def make_tensor_dataset(cfg: Config, dataset: pl.DataFrame):
    data = []
    for df in dataset.partition_by(
        cfg.index.sample_id, maintain_order=True, include_key=False
    ):
        labels = df.select(cfg.index.label).to_torch(dtype=pl.Float32)
        exclude = pl.exclude(cfg.index.split, cfg.index.label)
        subject = df.select(exclude)
        features = subject.to_torch(dtype=pl.Float32)
        data.append((features, labels))
    return data


class TimeSeriesDataset(Dataset):
    def __init__(self, cfg: Config, data: pl.DataFrame) -> None:
        self.dataset = make_tensor_dataset(cfg=cfg, dataset=data)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        features, labels = self.dataset[index]
        return features, labels


class FileDataset(Dataset):
    def __init__(self, cfg: Config, data: list[str]) -> None:
        self.files = data
        self.cfg = cfg

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        filepath = self.files[index]
        data = np.load(filepath)
        features = torch.tensor(data["features"], dtype=torch.float32)
        labels = torch.tensor(data["label"], dtype=torch.float32)
        return features, labels


def collate_fn(batch):
    features, labels = zip(*batch)
    padded_features = pad_sequence(features, batch_first=True, padding_value=0)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=float("nan"))
    return padded_features, padded_labels.squeeze(-1)


def init_datasets(splits: dict[str, pl.DataFrame | list[str]], cfg: Config):
    match splits["train"], splits["val"], splits["test"]:
        case list(), list(), list():
            train = FileDataset(cfg=cfg, data=splits["train"])
            val = FileDataset(cfg=cfg, data=splits["val"])
            test = FileDataset(cfg=cfg, data=splits["test"])
        case pl.DataFrame(), pl.DataFrame(), pl.DataFrame():
            train = TimeSeriesDataset(cfg=cfg, data=splits["train"])
            val = TimeSeriesDataset(cfg=cfg, data=splits["val"])
            test = TimeSeriesDataset(cfg=cfg, data=splits["test"])
    return train, val, test


class ABCDDataModule(LightningDataModule):
    def __init__(
        self, splits: dict[str, pl.DataFrame | list[str]], cfg: Config
    ) -> None:
        super().__init__()
        train, val, test = init_datasets(splits=splits, cfg=cfg)
        self.train = train
        self.val = val
        self.test = test
        self.loader = partial(
            DataLoader, **cfg.dataloader.model_dump(), collate_fn=collate_fn
        )

    def train_dataloader(self):
        return self.loader(self.train, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return self.loader(self.val, shuffle=False)

    def test_dataloader(self):
        return self.loader(self.test, shuffle=False)
