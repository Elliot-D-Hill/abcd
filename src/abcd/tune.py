import optuna
import polars as pl
import torch
from optuna.integration import PyTorchLightningPruningCallback
from optuna.pruners import HyperbandPruner
from optuna.samplers import QMCSampler, TPESampler
from torchmetrics.functional.classification import multiclass_auroc

from abcd.config import Config
from abcd.dataset import ABCDDataModule
from abcd.model import AutoEncoderClassifer, Network, make_trainer


def make_params(trial: optuna.Trial, cfg: Config):
    cfg = cfg.model_copy(deep=True)
    hparams = cfg.hyperparameters
    method_index = trial.suggest_int(**hparams.model.method)
    cfg.model.method = cfg.tuner.methods[method_index]
    cfg.model.autoencoder = True  # trial.suggest_int(**hparams.model.autoencoder)
    cfg.model.hidden_dim = trial.suggest_int(**hparams.model.hidden_dim)
    cfg.model.num_layers = trial.suggest_int(**hparams.model.num_layers)
    cfg.model.dropout = trial.suggest_float(**hparams.model.dropout)
    cfg.optimizer.lr = trial.suggest_float(**hparams.optimizer.lr)
    cfg.optimizer.momentum = trial.suggest_float(**hparams.optimizer.momentum)
    cfg.optimizer.weight_decay = trial.suggest_float(**hparams.optimizer.weight_decay)
    cfg.trainer.max_epochs = trial.suggest_int(**hparams.trainer.max_epochs)
    cfg.trainer.swa_lrs = trial.suggest_float(**hparams.trainer.swa_lrs)
    return cfg


def get_model(cfg: Config, best: bool):
    # cfg.experiment.analysis in {"mri_all", "questions_mri_all"}:
    if cfg.model.autoencoder:
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
        target=labels,
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
        pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val_loss")
        trainer = make_trainer(cfg=cfg, checkpoint=False, callbacks=[pruning_callback])
        trainer.fit(model, datamodule=self.data_module)
        val_loss = trainer.callback_metrics["val_loss"].item()
        val_auroc = trainer.callback_metrics["val_auroc"].item()
        text = ""
        if val_loss < self.best_val_loss:
            text += "Lowest loss yet: "
            self.best_val_loss = val_loss
            path = cfg.filepaths.data.results.checkpoints / "best.ckpt"
            trainer.save_checkpoint(path)
        if trainer.is_global_zero:
            text += f"Trial: {trial.number}, Loss: {val_loss}, Mean AUROC: {val_auroc}"
            auroc = auc(trainer, model, self.data_module)
            text += f", AUROC: {auroc}"
            print(text)
        return val_loss


def tune_model(cfg: Config, data_module):
    sampler = QMCSampler(seed=cfg.random_seed)
    pruner = HyperbandPruner(
        min_resource=cfg.hyperparameters.trainer.max_epochs["low"],
        max_resource=cfg.hyperparameters.trainer.max_epochs["high"],
    )
    storage = optuna.storages.RDBStorage(
        url="sqlite:///:memory:",
        engine_kwargs={"pool_size": 20, "connect_args": {"timeout": 10}},
    )
    study = optuna.create_study(
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction="minimize",
        study_name="ABCD",
    )
    objective = Objective(cfg=cfg, data_module=data_module)
    half_trials = cfg.tuner.n_trials // 2
    study.optimize(func=objective, n_trials=half_trials)
    study.sampler = TPESampler(multivariate=True, seed=cfg.random_seed)
    study.optimize(func=objective, n_trials=half_trials)
    df = pl.DataFrame(study.trials_dataframe())
    df.write_parquet(cfg.filepaths.data.results.study)
