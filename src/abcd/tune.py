from pathlib import Path

import optuna
import polars as pl
from optuna.samplers import QMCSampler, TPESampler

from abcd.config import Config
from abcd.dataset import ABCDDataModule
from abcd.model import Network, make_trainer

METHODS = {0: "mlp", 1: "rnn", 2: "lstm"}


def make_params(trial: optuna.Trial, cfg: Config):
    hparams = cfg.hyperparameters
    method_index = trial.suggest_int(**hparams.model.method)
    cfg.model.method = METHODS[method_index]
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


class Objective:
    def __init__(self, cfg: Config, data_module: ABCDDataModule):
        self.cfg = cfg
        self.data_module = data_module
        self.best_val_loss = float("inf")

    def __call__(self, trial: optuna.Trial):
        cfg = make_params(trial, cfg=self.cfg)
        model = get_model(cfg=cfg)
        trainer = make_trainer(cfg=cfg, checkpoint=True)
        path = Path(cfg.filepaths.data.results.checkpoints / "last.ckpt")
        if path.exists():
            path.unlink()
        trainer.fit(model, datamodule=self.data_module)
        metrics = trainer.validate(model, datamodule=self.data_module, ckpt_path="last")
        val_loss = metrics[-1]["val_loss"]
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            trainer.save_checkpoint(path.with_name("best.ckpt"))
        return val_loss


def tune_model(cfg: Config, data_module):
    sampler = QMCSampler(seed=cfg.random_seed)
    study = optuna.create_study(
        sampler=sampler, direction="minimize", study_name="ABCD"
    )
    objective = Objective(cfg=cfg, data_module=data_module)
    half_trials = cfg.tuner.n_trials // 2
    study.optimize(func=objective, n_trials=half_trials)
    sampler = TPESampler(multivariate=True, seed=cfg.random_seed)
    study.sampler = sampler
    study.optimize(func=objective, n_trials=half_trials)
    df = pl.DataFrame(study.trials_dataframe())
    df.write_parquet("data/study.parquet")
