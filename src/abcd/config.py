from copy import deepcopy
from pathlib import Path

import tomllib
from pydantic import BaseModel


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
    previous_label: str
    split: str
    site: str
    propensity: str


class Preprocess(BaseModel):
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
    enable_model_summary: bool
    enable_progress_bar: bool


class Tuner(BaseModel):
    n_trials: int
    methods: dict[int, str]


class Logging(BaseModel):
    checkpoint_every_n_steps: int
    log_every_n_steps: int


class Optimizer(BaseModel):
    lr: float
    momentum: float
    weight_decay: float
    nesterov: bool


class Model(BaseModel):
    method: str
    hidden_dim: int
    num_layers: int
    dropout: float
    output_dim: int
    input_dim: int = -1  # set at runtime


class ModelHParams(BaseModel):
    hidden_dim: dict
    num_layers: dict
    dropout: dict
    method: dict


class OptimizerHParams(BaseModel):
    lr: dict
    momentum: dict
    weight_decay: dict


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
    metadata: Path


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


class EvalMetrics(BaseModel):
    metrics: Path
    curves: Path
    sens_spec: Path


class Shap(BaseModel):
    shap_values: Path
    group_shap_values: Path
    shap_coef: Path
    group_shap_coef: Path


class Results(BaseModel):
    checkpoints: Path
    best_model: Path
    study: Path
    logs: Path
    predictions: Path
    eval: EvalMetrics
    shap: Shap


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
    analytic = deepcopy(cfg.filepaths.data.analytic.dict())
    for name, path in analytic.items():
        new_filepath = new_path / "analytic" / path
        new_filepath.parent.mkdir(parents=True, exist_ok=True)
        analytic[name] = new_filepath
    cfg.filepaths.data.analytic = Splits(**analytic)
    results = deepcopy(cfg.filepaths.data.results.dict())
    for name, path in results.copy().items():
        if isinstance(path, Path):
            new_filepath = new_path / "results" / path
            new_filepath.parent.mkdir(parents=True, exist_ok=True)
            results[name] = new_filepath
        if (name in ["eval", "shap"]) and isinstance(path, dict):
            for subkey, subval in list(path.items()):
                new_filepath = new_path / "results" / name / subval
                new_filepath.parent.mkdir(parents=True, exist_ok=True)
                results[name][subkey] = new_filepath
    cfg.filepaths.data.results = Results(**results)
    return cfg


def get_config(path: str, factor_model: str, analysis: str | None = None) -> Config:
    with open(path, "rb") as f:
        data = tomllib.load(f)
        cfg = Config(**data)
    if analysis is None:
        return cfg
    elif analysis == "metadata":
        new_path = Path("data/analyses/metadata")
    else:
        new_path = Path(f"data/analyses/{factor_model}/{analysis}")
    cfg.experiment.analysis = analysis
    cfg.experiment.factor_model = factor_model
    return update_paths(new_path=new_path, cfg=cfg)
