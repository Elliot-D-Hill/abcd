import optuna
import polars as pl
import torch
from optuna.samplers import QMCSampler, TPESampler
from torchmetrics.functional.classification import multiclass_auroc

from abcd.config import Config
from abcd.dataset import ABCDDataModule
from abcd.model import AutoEncoderClassifer, Network, make_trainer


def make_params(trial: optuna.Trial, cfg: Config):
    hparams = cfg.hyperparameters
    method_index = trial.suggest_int(**hparams.model.method)
    cfg.model.method = cfg.tuner.methods[method_index]
    cfg.model.hidden_dim = trial.suggest_int(**hparams.model.hidden_dim)
    cfg.model.num_layers = trial.suggest_int(**hparams.model.num_layers)
    cfg.model.dropout = trial.suggest_float(**hparams.model.dropout)
    cfg.optimizer.lr = trial.suggest_float(**hparams.optimizer.lr)
    cfg.optimizer.momentum = trial.suggest_float(**hparams.optimizer.momentum)
    cfg.optimizer.weight_decay = trial.suggest_float(**hparams.optimizer.weight_decay)
    cfg.trainer.max_epochs = trial.suggest_int(**hparams.trainer.max_epochs)
    return cfg


def get_model(cfg: Config, best: bool):
    if cfg.experiment.analysis in {"mri_all", "questions_mri_all"}:
        model_class = AutoEncoderClassifer
    else:
        model_class = Network
    if best:
        filepath = cfg.filepaths.data.results.best_model
        return model_class.load_from_checkpoint(checkpoint_path=filepath)
    return model_class(cfg=cfg)


def auc(trainer, model, data_module):
    predictions = trainer.predict(model, dataloaders=data_module.val_dataloader())
    if predictions is None:
        return float("inf")
    outputs = torch.cat([x[0] for x in predictions])
    labels = torch.cat([x[1] for x in predictions])
    return multiclass_auroc(
        preds=outputs,
        target=labels.int(),
        num_classes=outputs.shape[-1],
        average="none",
    )


class Objective:
    def __init__(self, cfg: Config, data_module: ABCDDataModule):
        self.cfg = cfg
        self.data_module = data_module
        self.best_val_loss = float("inf")

    def __call__(self, trial: optuna.Trial):
        cfg = make_params(trial, cfg=self.cfg)
        model = get_model(cfg=cfg, best=False)
        trainer = make_trainer(cfg=cfg, checkpoint=False)
        trainer.fit(model, datamodule=self.data_module)
        val_loss = trainer.callback_metrics["val_loss"].item()
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            path = cfg.filepaths.data.results.checkpoints / "best.ckpt"
            trainer.save_checkpoint(path)
        auroc_score = auc(trainer, model=model, data_module=self.data_module)
        print("AUC:", auroc_score)
        return val_loss


def tune_model(cfg: Config, data_module):
    sampler = QMCSampler(seed=cfg.random_seed)
    study = optuna.create_study(
        sampler=sampler, direction="minimize", study_name="ABCD"
    )
    objective = Objective(cfg=cfg, data_module=data_module)
    half_trials = cfg.tuner.n_trials // 2
    study.optimize(func=objective, n_trials=half_trials)
    study.sampler = TPESampler(multivariate=True, seed=cfg.random_seed)
    study.optimize(func=objective, n_trials=half_trials)
    df = pl.DataFrame(study.trials_dataframe())
    df.write_parquet(cfg.filepaths.data.results.study)
