from copy import deepcopy
from pathlib import Path

from pydantic import BaseModel
from tomllib import load


class Experiment(BaseModel):
    analyses: list[str]
    analysis: str
    factor_models: list[str]
    factor_model: str
    split_on: str


class Index(BaseModel):
    join_on: list[str]
    sample_id: str
    event: str
    label: str
    split: str
    site: str


class Preprocess(BaseModel):
    n_neighbors: int
    null_cutoff: float
    columns_to_drop: list[str]
    splits: dict[str, float]


class Dataloader(BaseModel):
    batch_size: int
    num_workers: int | str
    pin_memory: bool
    persistent_workers: bool
    multiprocessing_context: str | None = None


class Trainer(BaseModel):
    gradient_clip_val: float
    max_epochs: int


class Tuner(BaseModel):
    n_trials: int
    sampler: str


class Logging(BaseModel):
    checkpoint_every_n_steps: int
    log_every_n_steps: int


class Optimizer(BaseModel):
    lr: float
    momentum: float
    weight_decay: float
    lambda_l1: float
    nesterov: bool


class Model(BaseModel):
    method: str
    hidden_dim: int
    num_layers: int
    dropout: float
    input_dim: int
    output_dim: int


class ModelHParams(BaseModel):
    hidden_dim: dict
    num_layers: dict
    dropout: dict
    method: dict


class OptimizerHParams(BaseModel):
    lr: dict
    momentum: dict
    weight_decay: dict
    lambda_l1: dict


class TrainerHParams(BaseModel):
    max_epochs: dict


class Hyperparameters(BaseModel):
    model: ModelHParams
    optimizer: OptimizerHParams
    trainer: TrainerHParams


class Evaluate(BaseModel):
    n_bootstraps: int


class Dataset(BaseModel):
    name: str
    respondent: str
    columns: list[str]


class Splits(BaseModel):
    path: Path
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
    metadata: Path
    subject_metadata: Path


class Results(BaseModel):
    metrics: Path
    checkpoints: Path
    best_model: Path
    study: Path
    logs: Path
    predictions: Path
    shap_values: Path
    shap_coef: Path
    group_shap_coef: Path


class Data(BaseModel):
    raw: Raw
    processed: Processed
    analytic: Splits
    results: Results


class Filepaths(BaseModel):
    tables: Path
    plots: Path
    data: Data


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
    experiment: Experiment
    filepaths: Filepaths
    preprocess: Preprocess
    dataloader: Dataloader
    features: Features
    trainer: Trainer
    tuner: Tuner
    hyperparameters: Hyperparameters
    optimizer: Optimizer
    model: Model
    evaluation: Evaluate
    logging: Logging


def update_paths(new_path: Path, cfg: Config) -> Config:
    analytic = deepcopy(cfg.filepaths.data.analytic.model_dump())
    for name, path in analytic.items():
        new_filepath = new_path / "analytic" / path
        new_filepath.parent.mkdir(parents=True, exist_ok=True)
        analytic[name] = new_filepath
    cfg.filepaths.data.analytic = Splits(**analytic)
    results = deepcopy(cfg.filepaths.data.results.model_dump())
    for name, path in results.items():
        new_filepath = new_path / "results" / path
        new_filepath.parent.mkdir(parents=True, exist_ok=True)
        results[name] = new_filepath
    cfg.filepaths.data.results = Results(**results)
    return cfg


def get_config(path: str, factor_model: str, analysis: str | None = None) -> Config:
    with open(path, "rb") as f:
        cfg = Config(**load(f))
    if analysis is None:
        return cfg
    elif analysis == "metadata":
        new_path = Path("data/analyses/metadata")
    else:
        new_path = Path(f"data/analyses/{factor_model}/{analysis}")
    cfg.experiment.analysis = analysis
    cfg.experiment.factor_model = factor_model
    return update_paths(new_path=new_path, cfg=cfg)
