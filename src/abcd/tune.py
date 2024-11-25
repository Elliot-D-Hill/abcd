from functools import partial

import optuna
from optuna.samplers import MOTPESampler, QMCSampler, TPESampler

from abcd.cfg import Config
from abcd.model import make_model, make_trainer
from abcd.utils import cleanup_checkpoints


def objective(
    trial: optuna.Trial,
    cfg: Config,
    data_module,
    input_dim: int,
    output_dim: int,
):
    model_params = {
        "hidden_dim": trial.suggest_int(**cfg.model.hidden_dim),
        "num_layers": trial.suggest_int(**cfg.model.num_layers),
        "dropout": trial.suggest_float(**cfg.model.dropout),
        "method": trial.suggest_categorical(**cfg.model.method),
    }
    optimizer_params = {
        "lr": trial.suggest_float(**cfg.optimizer.lr),
        "weight_decay": trial.suggest_float(**cfg.optimizer.weight_decay),
        "lambda_l1": trial.suggest_float(**cfg.optimizer.lambda_l1),
    }
    lambda_l1 = trial.suggest_float(**cfg.optimizer.lambda_l1)
    model = make_model(
        input_dim=input_dim,
        output_dim=output_dim,
        nesterov=cfg.optimizer.nesterov,
        method="rnn",
        lambda_l1=lambda_l1,
        **optimizer_params,
        **model_params,
    )
    max_epochs = trial.suggest_int(**cfg.training.max_epochs)
    trainer = make_trainer(max_epochs=max_epochs, cfg=cfg, checkpoint=True)
    trainer.fit(model, datamodule=data_module)
    cleanup_checkpoints(cfg.filepaths.data.results.checkpoints, mode="min")
    loss = trainer.checkpoint_callbacks[0].best_model_score.item()  # type: ignore
    return loss


def get_sampler(cfg: Config):
    half_trials = cfg.n_trials // 2
    if cfg.sampler == "tpe":
        sampler = TPESampler(
            multivariate=True,
            n_startup_trials=half_trials,
            seed=cfg.random_seed,
        )
        direction = "minimize"
    elif cfg.sampler == "motpe":
        sampler = MOTPESampler(n_startup_trials=half_trials, seed=cfg.random_seed)
        direction = ["minimize", "maximize"]
    elif cfg.sampler == "qmc":
        sampler = QMCSampler(seed=cfg.random_seed)
        direction = "minimize"
    else:
        raise ValueError(
            f"Invalid sampler '{cfg.sampler}'. Choose from: 'tpe', 'motpe', 'qmc'"
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
    study.optimize(func=objective_function, n_trials=cfg.n_trials)
    df = study.trials_dataframe()
    df.to_csv("data/study.csv", index=False)
