import random
from functools import partial

import numpy as np
import polars as pl
import torch
from lightning import LightningDataModule
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import BigBirdTokenizer

from abcd.config import Config


def make_tensor_dataset(cfg: Config, dataset: pl.DataFrame):
    data = []
    features = pl.exclude(cfg.index.split, cfg.index.label, cfg.index.propensity)
    for df in dataset.partition_by(
        cfg.index.sample_id, maintain_order=True, include_key=False
    ):
        labels = df[cfg.index.label].to_torch().float()
        inputs = df.select(features).to_torch(dtype=pl.Float32)
        if cfg.experiment.analysis == "propensity":
            propensity = df.select(cfg.index.propensity).to_torch(dtype=pl.Float32)
        else:
            propensity = torch.tensor([1.0] * labels.size(0))
        sample = (inputs, labels, propensity)
        data.append(sample)
    return data, df.select(features).columns


class TimeSeriesDataset(Dataset):
    def __init__(self, cfg: Config, data: pl.DataFrame) -> None:
        self.dataset, self.columns = make_tensor_dataset(cfg=cfg, dataset=data)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index) -> tuple[torch.Tensor, ...]:
        features, labels, propensity = self.dataset[index]
        return features, labels, propensity


class FileDataset(Dataset):
    def __init__(self, cfg: Config, data: list[str]) -> None:
        self.files = data
        self.cfg = cfg

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index) -> tuple[torch.Tensor, ...]:
        filepath = self.files[index]
        data = np.load(filepath)
        features = torch.tensor(data["features"], dtype=torch.float32)
        labels = torch.tensor(data["label"], dtype=torch.float32)
        dummy_propensity = torch.tensor([1.0] * labels.size(0))
        return features, labels, dummy_propensity


# df = df.select(
#     pl.col("sentence")
#     .list.sample(fraction=1, shuffle=True)
#     .list.join(separator="</s> <s>")
# )


def add_bos_eos(sentences: list[str]) -> str:
    return "</s> " + " </s> <s> ".join(sentences) + " <s>"


def make_llm_dataset(df: pl.DataFrame):
    print("Making LLM dataset")
    data = []
    for i in range(df.height):
        sentences = df[i]["sentence"].to_list()[0]
        sentences = add_bos_eos(sentences)
        labels = df[i]["y_{t+1}"].to_torch().float()
        dummy_propensity = torch.tensor([1.0] * labels.size(0))
        sample = (sentences, labels, dummy_propensity)
        data.append(sample)
    return data


class LLMDataset(Dataset):
    def __init__(self, data):
        self.data = make_llm_dataset(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentences, labels, propensity = self.data[idx]
        sentences = sentences.split(" </s> <s> ")
        random.shuffle(sentences)
        sentences = add_bos_eos(sentences)
        return sentences, labels, propensity


def collate_fn(batch):
    features, labels, propensity = zip(*batch)
    features = pad_sequence(features, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=torch.nan).squeeze(-1)
    propensity = pad_sequence(propensity, batch_first=True, padding_value=torch.nan)
    return features, labels, propensity


def llm_collate_fn(batch, tokenizer):
    sentences, labels, propensity = zip(*batch)
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
    labels = torch.concat(labels)
    propensity = torch.concat(propensity)
    return inputs, labels, propensity


def init_datasets(splits: dict, cfg: Config):
    if cfg.experiment.analysis in {"mri_all", "questions_mri_all"}:
        train = FileDataset(cfg=cfg, data=splits["train"])
        val = FileDataset(cfg=cfg, data=splits["val"])
        test = FileDataset(cfg=cfg, data=splits["test"])
    elif cfg.experiment.analysis == "llm":
        train = LLMDataset(data=splits["train"])
        val = LLMDataset(data=splits["val"])
        test = LLMDataset(data=splits["test"])
    else:
        train = TimeSeriesDataset(cfg=cfg, data=splits["train"])
        val = TimeSeriesDataset(cfg=cfg, data=splits["val"])
        test = TimeSeriesDataset(cfg=cfg, data=splits["test"])
    return train, val, test


def get_tokenizer(cfg: Config):
    if (cfg.filepaths.data.models.model / "tokenizer_config.json").exists():
        tokenizer = BigBirdTokenizer.from_pretrained(
            cfg.filepaths.data.models.model, clean_up_tokenization_spaces=True
        )
    else:
        tokenizer = BigBirdTokenizer.from_pretrained(
            cfg.model.llm_name, clean_up_tokenization_spaces=True
        )
        tokenizer.save_pretrained(cfg.filepaths.data.models.model)
    return tokenizer


class ABCDDataModule(LightningDataModule):
    def __init__(
        self, splits: dict[str, pl.DataFrame | list[str]], cfg: Config
    ) -> None:
        super().__init__()
        train, val, test = init_datasets(splits=splits, cfg=cfg)
        self.train = train
        self.val = val
        self.test = test
        if cfg.experiment.analysis == "llm":
            tokenizer = get_tokenizer(cfg=cfg)
            collate = partial(llm_collate_fn, tokenizer=tokenizer)
        else:
            collate = collate_fn
        self.loader = partial(DataLoader, collate_fn=collate, **cfg.dataloader.dict())
        if isinstance(train, TimeSeriesDataset):
            self.columns = train.columns
        else:
            self.columns = []

    def train_dataloader(self):
        return self.loader(self.train, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return self.loader(self.val, shuffle=False)

    def test_dataloader(self):
        return self.loader(self.test, shuffle=False)
