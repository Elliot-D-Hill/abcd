from functools import partial

import polars as pl
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
        features = df.select(pl.exclude(cfg.index.label)).to_torch(dtype=pl.Float32)
        data.append((features, labels))
    return data


class RNNDataset(Dataset):
    def __init__(self, cfg: Config, dataset: pl.DataFrame) -> None:
        self.dataset = make_tensor_dataset(cfg=cfg, dataset=dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        features, labels = self.dataset[index]
        return features, labels


def collate_fn(batch):
    features, labels = zip(*batch)
    padded_features = pad_sequence(features, batch_first=True, padding_value=0)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=float("nan"))
    return padded_features, padded_labels


class ABCDDataModule(LightningDataModule):
    def __init__(self, splits: dict[str, pl.DataFrame], cfg: Config) -> None:
        super().__init__()
        self.train_dataset = RNNDataset(cfg=cfg, dataset=splits["train"])
        self.val_dataset = RNNDataset(cfg=cfg, dataset=splits["val"])
        self.test_dataset = RNNDataset(cfg=cfg, dataset=splits["test"])
        self.feature_names = (
            splits["train"].drop([cfg.index.sample_id, cfg.index.label]).columns
        )
        self.loader = partial(
            DataLoader,
            **cfg.dataloader.model_dump(),
            collate_fn=collate_fn,
        )

    def train_dataloader(self):
        return self.loader(self.train_dataset, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, shuffle=False)
