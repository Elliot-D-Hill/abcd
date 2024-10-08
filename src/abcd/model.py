from pathlib import Path
from torch import nn, isnan
from torch.optim.sgd import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.ops import MLP
from lightning import LightningModule
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    RichProgressBar,
    EarlyStopping,
)
from lightning.pytorch.loggers import TensorBoardLogger
from lightning import Trainer
from torchmetrics.functional.classification import multiclass_auroc, binary_auroc
from abcd.config import Config
from abcd.utils import get_best_checkpoint


class RNN(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        num_layers,
        dropout,
    ):
        super().__init__()
        self.rnn = nn.RNN(
            input_dim,
            hidden_dim,
            dropout=dropout if num_layers > 1 else 0.0,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity="tanh",
        )
        self.fc = nn.Linear(in_features=hidden_dim, out_features=output_dim)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out)
        return out


def make_metrics(step, loss, outputs, labels) -> dict:
    metrics = {f"{step}_loss": loss}
    if step == "val":
        labels = labels.long()
        if outputs.shape[-1] > 1:
            auroc_score = multiclass_auroc(
                preds=outputs,
                target=labels,
                num_classes=outputs.shape[-1],
                average="none",
            )
            metrics.update(
                {f"auc_{i}": val for i, val in enumerate(auroc_score, start=1)}
            )
        else:
            outputs = outputs.squeeze(1)
            auroc_score = binary_auroc(preds=outputs, target=labels)
            metrics.update({"auc": auroc_score})
    return metrics


def drop_nan(outputs, labels):
    is_not_nan = ~isnan(labels)
    return outputs[is_not_nan], labels[is_not_nan]


class Network(LightningModule):
    def __init__(
        self,
        model_hyperparameters: dict,
        optimizer_hyperparameters: dict,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.model = make_architecture(**model_hyperparameters)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = SGD(self.model.parameters(), **optimizer_hyperparameters)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", patience=2, factor=0.1
        )

    def forward(self, inputs):
        return self.model(inputs)

    def step(self, step: str, batch):
        inputs, labels = batch
        outputs = self(inputs)
        outputs, labels = drop_nan(outputs, labels)
        loss = self.criterion(outputs.squeeze(1), labels)
        metrics = make_metrics(step, loss, outputs, labels)
        self.log_dict(metrics, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step("train", batch)

    def validation_step(self, batch, batch_idx):
        return self.step("val", batch)

    def test_step(self, batch, batch_idx):
        self.step("test", batch)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        inputs, labels = batch
        outputs = self(inputs)
        outputs, labels = drop_nan(outputs, labels)
        return outputs, labels

    def configure_optimizers(self):
        scheduler_config = {
            "scheduler": self.scheduler,
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 2,
        }
        return {"optimizer": self.optimizer, "lr_scheduler": scheduler_config}


def make_trainer(config: Config, checkpoint: bool) -> Trainer:
    early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=4)
    callbacks = [early_stopping, RichProgressBar()]
    if checkpoint:
        checkpoint_callback = ModelCheckpoint(
            dirpath=config.filepaths.data.results.checkpoints,
            filename="{epoch}_{step}_{val_loss:.3f}",
            save_top_k=1,
            verbose=config.verbose,
            monitor="val_loss",
            mode="min",
            every_n_train_steps=config.logging.checkpoint_every_n_steps,
        )
        callbacks.append(
            checkpoint_callback,
        )
    logger = (
        TensorBoardLogger(save_dir=config.filepaths.data.results.logs)
        if config.log
        else False
    )
    trainer = Trainer(
        logger=logger,
        callbacks=callbacks,
        enable_checkpointing=checkpoint,
        gradient_clip_val=config.training.gradient_clip,
        max_epochs=config.training.max_epochs,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=config.logging.log_every_n_steps,
        fast_dev_run=config.fast_dev_run,
        check_val_every_n_epoch=1,
    )
    return trainer


def make_architecture(
    method: str,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int,
    dropout: float,
):
    match method:
        case "rnn":
            return RNN(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
            )
        case "mlp":
            hidden_channels = [hidden_dim] * num_layers
            return MLP(
                in_channels=input_dim,
                hidden_channels=hidden_channels + [output_dim],
                dropout=dropout,
            )
        case _:
            raise ValueError(f"Invalid method '{method}'. Choose from: 'rnn' or 'mlp'")


def make_model(
    input_dim: int,
    output_dim: int,
    momentum: float,
    nesterov: bool,
    method: str = "rnn",
    hidden_dim: int = 256,
    num_layers: int = 2,
    dropout: float = 0.0,
    lr: float = 0.01,
    weight_decay: float = 0.0,
    checkpoints: Path | None = None,
):
    if checkpoints:
        best_model_path = get_best_checkpoint(ckpt_folder=checkpoints, mode="min")
        return Network.load_from_checkpoint(checkpoint_path=best_model_path)
    model_hyperparameters = {
        "method": method,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "dropout": dropout,
    }
    optimizer_hyperparameters = {
        "lr": lr,
        "weight_decay": weight_decay,
        "momentum": momentum,
        "nesterov": nesterov,
    }
    return Network(
        model_hyperparameters=model_hyperparameters,
        optimizer_hyperparameters=optimizer_hyperparameters,
    )
