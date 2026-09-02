from pathlib import Path

from lightning import seed_everything

from .config import get_config
from .dataset import ABCDDataModule
from .evaluate import auroc, make_predictions, predict
from .model import Network
from .process import make_dataset
from .trainer import make_trainer


def main() -> None:
    cfg = get_config(Path("config.toml"))
    seed_everything(cfg.random_seed, workers=True)
    splits = make_dataset(cfg)
    data_module = ABCDDataModule(
        splits=splits, index=cfg.index, dataloader=cfg.dataloader
    )
    model = Network(
        input_dim=data_module.input_dim,
        model=cfg.model,
        optimizer=cfg.optimizer,
        scheduler=cfg.scheduler,
        ignore_index=cfg.dataloader.ignore_index,
    )
    trainer, checkpoint = make_trainer(
        cfg.trainer, checkpoints=cfg.filepaths.checkpoints
    )
    trainer.fit(model, datamodule=data_module)
    print(f"Best validation loss: {checkpoint.best_model_score:.4f}")
    outputs, labels = predict(
        trainer,
        model,
        data_module,
        checkpoint_path=checkpoint.best_model_path,
        ignore_index=cfg.dataloader.ignore_index,
    )
    predictions = make_predictions(
        splits["test"], index=cfg.index, outputs=outputs, labels=labels
    )
    cfg.filepaths.predictions.parent.mkdir(parents=True, exist_ok=True)
    predictions.write_parquet(cfg.filepaths.predictions)
    quartile_auroc = auroc(outputs, labels)
    print(f"Test AUROC by quartile at t+1: {quartile_auroc.tolist()}")
    print(f"Test AUROC (macro): {quartile_auroc.mean():.4f}")


if __name__ == "__main__":
    main()
