import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class Index(Frozen):
    sample_id: str
    event: str
    previous_label: str
    label: str
    split: str


class Preprocess(Frozen):
    feature_events: tuple[int, ...]
    splits: dict[str, float]


class Columns(Frozen):
    cbcl: tuple[str, ...]
    mri: tuple[str, ...]
    excluded: tuple[str, ...]


class Dataloader(Frozen):
    batch_size: int
    num_workers: int
    ignore_index: int


class Trainer(Frozen):
    accelerator: str
    precision: str
    max_epochs: int
    gradient_clip_val: float
    enable_progress_bar: bool
    enable_model_summary: bool


class Optimizer(Frozen):
    lr: float
    momentum: float
    nesterov: bool
    weight_decay: float


class Scheduler(Frozen):
    factor: float
    patience: int


class Model(Frozen):
    hidden_dim: int
    num_layers: int
    dropout: float
    output_dim: int


class Filepaths(Frozen):
    dataset: Path
    train: Path
    val: Path
    test: Path
    checkpoints: Path
    predictions: Path


class Config(Frozen):
    random_seed: int
    index: Index
    preprocess: Preprocess
    columns: Columns
    dataloader: Dataloader
    trainer: Trainer
    optimizer: Optimizer
    scheduler: Scheduler
    model: Model
    filepaths: Filepaths


def get_config(path: Path) -> Config:
    with path.open("rb") as file:
        return Config(**tomllib.load(file))
