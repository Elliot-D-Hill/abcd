from typing import cast

import optuna
import polars as pl
from optuna.integration import PyTorchLightningPruningCallback
from optuna.pruners import HyperbandPruner
from optuna.samplers import QMCSampler, TPESampler
from torch.utils.checkpoint import checkpoint

from abcd.config import Config
from abcd.dataset import ABCDDataModule
from abcd.model import AutoEncoderClassifer, Network
from abcd.trainer import make_trainer


def make_params(trial: optuna.Trial, cfg: Config):
    cfg = cfg.model_copy(deep=True)
    hparams = cfg.hyperparameters
    method_index = trial.suggest_int(**hparams.model.method)
    cfg.model.method = cfg.tuner.methods[method_index]
    cfg.model.hidden_dim = trial.suggest_int(**hparams.model.hidden_dim)
    cfg.model.num_layers = trial.suggest_int(**hparams.model.num_layers)
    cfg.model.dropout = trial.suggest_float(**hparams.model.dropout)
    cfg.optimizer.lr = trial.suggest_float(**hparams.optimizer.lr)
    cfg.optimizer.momentum = trial.suggest_float(**hparams.optimizer.momentum)
    cfg.optimizer.weight_decay = trial.suggest_float(**hparams.optimizer.weight_decay)
    cfg.trainer.swa_lrs = trial.suggest_float(**hparams.trainer.swa_lrs)
    return cfg


def get_model(cfg: Config, best: bool) -> Network | AutoEncoderClassifer:
    model_class = (
        AutoEncoderClassifer
        if cfg.experiment.analysis in {"mri_all", "questions_mri_all"}
        else Network
    )
    if best:
        filepath = cfg.filepaths.data.results.best_model
        model = model_class.load_from_checkpoint(checkpoint_path=filepath)
    else:
        model = model_class(cfg=cfg)
        if model_class == AutoEncoderClassifer:
            model = cast(AutoEncoderClassifer, checkpoint(model))
    return model


class Objective:
    def __init__(self, cfg: Config, data_module: ABCDDataModule):
        self.cfg = cfg
        self.data_module = data_module
        self.minimize = cfg.tuner.direction == "minimize"
        self.best_value = float("inf") if self.minimize else float("-inf")

    def __call__(self, trial: optuna.Trial):
        cfg = make_params(trial, cfg=self.cfg)
        model = get_model(cfg=cfg, best=False)
        pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val_loss")
        trainer = make_trainer(cfg=cfg, checkpoint=False, callbacks=[pruning_callback])
        trainer.fit(model, datamodule=self.data_module)
        val_loss = trainer.callback_metrics["val_loss"].item()
        val_auroc = trainer.callback_metrics["val_auroc"].item()
        if self.minimize:
            save_checkpoint = val_loss < self.best_value
            current_value = val_loss
        else:
            save_checkpoint = val_auroc > self.best_value
            current_value = val_auroc
        text = ""
        if save_checkpoint:
            text += "Best value yet: "
            self.best_value = current_value
            path = cfg.filepaths.data.results.checkpoints / "best.ckpt"
            trainer.save_checkpoint(path)
        if trainer.is_global_zero:
            text += f"Trial: {trial.number}, Loss: {val_loss:.2f}, Mean AUROC: {val_auroc:.2f}"
            print(text)
        return current_value


def tune_model(cfg: Config, data_module):
    sampler = QMCSampler(seed=cfg.random_seed)
    pruner = HyperbandPruner(
        min_resource=cfg.tuner.min_resource,
        max_resource=cfg.trainer.max_epochs,
    )
    storage = optuna.storages.RDBStorage(
        url="sqlite:///:memory:",
        engine_kwargs={"pool_size": 10, "connect_args": {"timeout": 10}},
    )
    study = optuna.create_study(
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction=cfg.tuner.direction,
        study_name="ABCD",
    )
    objective = Objective(cfg=cfg, data_module=data_module)
    half_trials = cfg.tuner.n_trials // 2
    study.optimize(func=objective, n_trials=half_trials)
    study.sampler = TPESampler(multivariate=True, seed=cfg.random_seed)
    study.optimize(func=objective, n_trials=half_trials)
    df = pl.DataFrame(study.trials_dataframe())
    df.write_parquet(cfg.filepaths.data.results.study)
