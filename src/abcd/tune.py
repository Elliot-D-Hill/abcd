from functools import partial

import optuna
import polars as pl
from optuna.samplers import QMCSampler, TPESampler

from abcd.config import Config
from abcd.model import Network, make_trainer


def make_params(trial: optuna.Trial, cfg: Config):
    hparams = cfg.hyperparameters
    cfg.model.method = trial.suggest_categorical(**hparams.model.method)
    cfg.model.hidden_dim = trial.suggest_int(**hparams.model.hidden_dim)
    cfg.model.num_layers = trial.suggest_int(**hparams.model.num_layers)
    cfg.model.dropout = trial.suggest_float(**hparams.model.dropout)
    cfg.optimizer.lr = trial.suggest_float(**hparams.optimizer.lr)
    cfg.optimizer.weight_decay = trial.suggest_float(**hparams.optimizer.weight_decay)
    cfg.optimizer.lambda_l1 = trial.suggest_float(**hparams.optimizer.lambda_l1)
    cfg.trainer.max_epochs = trial.suggest_int(**hparams.trainer.max_epochs)
    return cfg


def get_model(cfg: Config, best: bool = False):
    if best:
        filepath = cfg.filepaths.data.results.best_model
        return Network.load_from_checkpoint(checkpoint_path=filepath)
    return Network(cfg=cfg)


def objective(trial: optuna.Trial, cfg: Config, data_module):
    cfg = make_params(trial, cfg=cfg)
    model = get_model(cfg=cfg)
    trainer = make_trainer(cfg=cfg, checkpoint=True)
    trainer.fit(model, datamodule=data_module)
    metrics = trainer.validate(model, datamodule=data_module, ckpt_path="last")
    return metrics[-1]["val_loss"]


def get_sampler(cfg: Config):
    half_trials = cfg.tuner.n_trials // 2
    if cfg.tuner.sampler == "tpe":
        sampler = TPESampler(
            multivariate=True, n_startup_trials=half_trials, seed=cfg.random_seed
        )
    elif cfg.tuner.sampler == "qmc":
        sampler = QMCSampler(seed=cfg.random_seed)
    else:
        raise ValueError(
            f"Invalid sampler '{cfg.tuner.sampler}'. Choose from: 'tpe', 'qmc'"
        )
    return sampler


def tune(cfg: Config, data_module):
    sampler = get_sampler(cfg=cfg)
    study = optuna.create_study(
        sampler=sampler, direction="minimize", study_name="ABCD"
    )
    objective_function = partial(objective, cfg=cfg, data_module=data_module)
    study.optimize(func=objective_function, n_trials=cfg.tuner.n_trials)
    df = pl.DataFrame(study.trials_dataframe())
    df.write_parquet("data/study.parquet")
