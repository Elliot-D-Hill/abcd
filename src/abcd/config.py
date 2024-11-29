from copy import deepcopy
from multiprocessing import cpu_count
from pathlib import Path

from pydantic import BaseModel
from tomllib import load


class Splits(BaseModel):
    train: Path
    val: Path
    test: Path


class Raw(BaseModel):
    labels: Path
    dataset: Path
    metadata: Path
    features: Path
    mri: Path
    splits: Path


class Processed(BaseModel):
    labels: Path
    features: Path
    dataset: Path


class Results(BaseModel):
    metrics: Path
    checkpoints: Path
    study: Path
    logs: Path
    predictions: Path


class Data(BaseModel):
    raw: Raw
    processed: Processed
    analytic: Splits
    results: Results


class Filepaths(BaseModel):
    tables: Path
    plots: Path
    data: Data


class Preprocess(BaseModel):
    n_neighbors: int
    n_quantiles: int
    train_size: float


class Dataloader(BaseModel):
    batch_size: int
    num_workers: int | str
    pin_memory: bool
    persistent_workers: bool
    multiprocessing_context: str

    def __post_init__(self):
        if self.num_workers == "auto":
            self.num_workers = cpu_count()


class Trainer(BaseModel):
    gradient_clip: float
    max_epochs: dict


class Tuner(BaseModel):
    n_trials: int
    sampler: str


class Logging(BaseModel):
    checkpoint_every_n_steps: int
    log_every_n_steps: int


class Optimizer(BaseModel):
    lr: dict
    momentum: dict
    weight_decay: dict
    lambda_l1: dict
    nesterov: bool


class Model(BaseModel):
    method: dict
    hidden_dim: dict
    num_layers: dict
    dropout: dict


class Evaluate(BaseModel):
    n_bootstraps: int


class Dataset(BaseModel):
    name: str
    respondent: str
    columns: list[str]


class Features(BaseModel):
    mh_p_cbcl: Dataset
    abcd_p_demo: Dataset
    led_l_adi: Dataset
    abcd_y_lt: Dataset
    abcd_aces: Dataset
    ce_p_fes: Dataset
    ce_y_fes: Dataset
    ce_p_nsc: Dataset
    ce_y_nsc: Dataset
    ce_y_pm: Dataset
    ce_p_psb: Dataset
    ce_y_psb: Dataset
    ce_y_srpf: Dataset
    nt_p_stq: Dataset
    nt_y_st: Dataset
    ph_p_sds: Dataset
    su_p_pr: Dataset
    mh_p_fhx: Dataset
    mri_y_dti_fa_fs_at: Dataset
    mri_y_rsfmr_cor_gp_gp: Dataset
    mri_y_tfmr_sst_csvcg_dsk: Dataset
    mri_y_tfmr_mid_alrvn_dsk: Dataset
    mri_y_tfmr_nback_2b_dsk: Dataset


class Index(BaseModel):
    join_on: list[str]
    sample_id: str
    event: str
    label: str


class Analyses(BaseModel):
    analyses: list[str]
    analysis: str
    factor_models: list[str]
    factor_model: str


class Config(BaseModel):
    random_seed: int
    device: str
    fast_dev_run: bool
    regenerate: bool
    tune: bool
    log: bool
    predict: bool
    evaluate: bool
    importance: bool
    plot: bool
    tables: bool
    verbose: bool
    index: Index
    analyses: Analyses
    filepaths: Filepaths
    preprocess: Preprocess
    dataloader: Dataloader
    features: Features
    trainer: Trainer
    tuner: Tuner
    optimizer: Optimizer
    model: Model
    evaluation: Evaluate
    logging: Logging


def update_paths(new_path: Path, cfg: Config) -> Config:
    analytic = deepcopy(cfg.filepaths.data.analytic.model_dump())
    for name, path in analytic.items():
        new_filepath = new_path / path
        new_filepath.parent.mkdir(parents=True, exist_ok=True)
        analytic[name] = new_filepath
    cfg.filepaths.data.analytic = Splits(**analytic)
    results = deepcopy(cfg.filepaths.data.results.model_dump())
    for name, path in results.items():
        new_filepath = new_path / path
        new_filepath.parent.mkdir(parents=True, exist_ok=True)
        results[name] = new_filepath
    cfg.filepaths.data.results = Results(**results)
    return cfg


def get_config(factor_model: str, analysis: str | None = None) -> Config:
    with open("cfg.toml", "rb") as f:
        cfg = Config(**load(f))
    if analysis is None:
        return cfg
    elif analysis == "metadata":
        new_path = Path("data/analyses/metadata")
    else:
        new_path = Path(f"data/analyses/{factor_model}/{analysis}")
    cfg.analysis = analysis
    cfg.factor_model = factor_model
    return update_paths(new_path=new_path, cfg=cfg)
