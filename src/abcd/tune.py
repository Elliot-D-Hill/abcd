from functools import partial

import optuna
from optuna.samplers import MOTPESampler, QMCSampler, TPESampler
from tomllib import load

from abcd.config import Config
from abcd.model import make_model, make_trainer
from abcd.utils import cleanup_checkpoints


def make_params(trial: optuna.Trial, cfg: Config):
    with open("config.toml", "rb") as f:
        data = load(f)
    data["model"]["hidden_dim"] = trial.suggest_int(**cfg.model.hidden_dim)
    data["model"]["num_layers"] = trial.suggest_int(**cfg.model.num_layers)
    data["model"]["dropout"] = trial.suggest_float(**cfg.model.dropout)
    data["model"]["method"] = trial.suggest_categorical(**cfg.model.method)
    data["optimizer"]["lr"] = trial.suggest_float(**cfg.optimizer.lr)
    data["optimizer"]["weight_decay"] = trial.suggest_float(
        **cfg.optimizer.weight_decay
    )
    data["optimizer"]["lambda_l1"] = trial.suggest_float(**cfg.optimizer.lambda_l1)
    data["training"]["max_epochs"] = trial.suggest_int(**cfg.trainer.max_epochs)
    return Config(**data)


def objective(
    trial: optuna.Trial, cfg: Config, data_module, input_dim: int, output_dim: int
):
    cfg = make_params(trial, cfg)
    model = make_model(input_dim=input_dim, output_dim=output_dim, cfg=cfg)
    trainer = make_trainer(cfg=cfg, checkpoint=True)
    trainer.fit(model, datamodule=data_module)
    cleanup_checkpoints(cfg.filepaths.data.results.checkpoints, mode="min")
    loss = trainer.checkpoint_callbacks[0].best_model_score.item()  # type: ignore
    return loss


def get_sampler(cfg: Config):
    half_trials = cfg.tuner.n_trials // 2
    if cfg.tuner.sampler == "tpe":
        sampler = TPESampler(
            multivariate=True,
            n_startup_trials=half_trials,
            seed=cfg.random_seed,
        )
        direction = "minimize"
    elif cfg.tuner.sampler == "motpe":
        sampler = MOTPESampler(n_startup_trials=half_trials, seed=cfg.random_seed)
        direction = ["minimize", "maximize"]
    elif cfg.tuner.sampler == "qmc":
        sampler = QMCSampler(seed=cfg.random_seed)
        direction = "minimize"
    else:
        raise ValueError(
            f"Invalid sampler '{cfg.tuner.sampler}'. Choose from: 'tpe', 'motpe', 'qmc'"
        )
    return sampler, direction


def tune(cfg: Config, data_module, input_dim: int, output_dim: int):
    sampler, direction = get_sampler(cfg=cfg)
    study = optuna.create_study(
        sampler=sampler, directions=direction, study_name="ABCD"
    )
    objective_function = partial(
        objective,
        cfg=cfg,
        data_module=data_module,
        input_dim=input_dim,
        output_dim=output_dim,
    )
    study.optimize(func=objective_function, n_trials=cfg.tuner.n_trials)
    df = study.trials_dataframe()
    df.to_csv("data/study.csv", index=False)
