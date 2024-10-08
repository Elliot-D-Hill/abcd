from functools import partial
from optuna import Trial, create_study
from optuna.samplers import TPESampler

from abcd.config import Config
from abcd.model import make_model, make_trainer
from abcd.utils import cleanup_checkpoints


def objective(
    trial: Trial,
    config: Config,
    data_module,
    input_dim: int,
    output_dim: int,
):
    model_params = {
        "hidden_dim": trial.suggest_categorical(
            name="hidden_dim", choices=config.model.hidden_dim
        ),
        "num_layers": trial.suggest_int(
            name="num_layers",
            low=config.model.num_layers["low"],
            high=config.model.num_layers["high"],
        ),
        "dropout": trial.suggest_float(
            name="dropout",
            low=config.model.dropout["low"],
            high=config.model.dropout["high"],
        ),
        "method": trial.suggest_categorical(name="method", choices=["rnn", "mlp"]),
    }
    optimizer_params = {
        "lr": trial.suggest_float(
            name="lr",
            low=config.optimizer.lr["low"],
            high=config.optimizer.lr["high"],
        ),
        "weight_decay": trial.suggest_float(
            name="weight_decay",
            low=config.optimizer.weight_decay["low"],
            high=config.optimizer.weight_decay["high"],
        ),
    }
    model = make_model(
        input_dim=input_dim,
        output_dim=output_dim,
        momentum=config.optimizer.momentum,
        nesterov=config.optimizer.nesterov,
        **optimizer_params,
        **model_params,
    )
    trainer = make_trainer(config, checkpoint=True)
    trainer.fit(model, datamodule=data_module)
    cleanup_checkpoints(config.filepaths.data.results.checkpoints, mode="min")
    return trainer.checkpoint_callbacks[0].best_model_score.item()  # type: ignore


def tune(config: Config, data_module, input_dim: int, output_dim: int):
    sampler = TPESampler(
        seed=config.random_seed,
        multivariate=True,
        n_startup_trials=config.n_trials // 2,
    )
    study = create_study(
        sampler=sampler,
        direction="minimize",
        study_name="ABCD",
    )
    objective_function = partial(
        objective,
        config=config,
        data_module=data_module,
        input_dim=input_dim,
        output_dim=output_dim,
    )
    study.optimize(func=objective_function, n_trials=config.n_trials)
