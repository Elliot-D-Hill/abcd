from functools import partial
from pathlib import Path

import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers import TensorBoardLogger
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim.sgd import SGD
from torchmetrics.functional.classification import multiclass_auroc
from torchvision.ops import MLP

from abcd.cfg import Config
from abcd.utils import get_best_checkpoint


class SequenceModel(nn.Module):
    def __init__(
        self,
        method: type[nn.Module],
        input_dim,
        output_dim,
        hidden_dim,
        num_layers,
        dropout,
    ):
        super().__init__()
        self.rnn = method(
            input_size=input_dim,
            hidden_size=hidden_dim,
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


def generate_mask(seq_len):
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mask = mask.masked_fill(mask == 1, float("-inf")).masked_fill(mask == 0, float(0.0))
    return mask


class Transformer(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        num_layers,
        num_heads,
        dropout,
        max_seq_len,
    ):
        super().__init__()

        self.positional_encoding = nn.Parameter(torch.randn(1, max_seq_len, input_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.output_fc = nn.Linear(in_features=hidden_dim, out_features=output_dim)

    def forward(self, x: torch.Tensor):
        x = x + self.positional_encoding
        mask = generate_mask(x.size(1)).to(x.device)
        padding_mask = (x == 0.0).all(dim=-1)
        out = self.transformer_encoder(x, mask=mask, src_key_padding_mask=padding_mask)
        out = out.mean(dim=1)
        out = self.output_fc(out)
        return out


def make_metrics(step, loss, outputs, labels) -> dict:
    metrics = {f"{step}_loss": loss}
    if step == "val":
        labels = labels.long()
        auroc_score = multiclass_auroc(
            preds=outputs,
            target=labels,
            num_classes=outputs.shape[-1],
            average="none",
        )
        metrics.update({"auc": auroc_score.mean().item()})
    return metrics


def drop_nan(outputs, labels):
    is_not_nan = ~torch.isnan(labels)
    return outputs[is_not_nan], labels[is_not_nan]


class Network(LightningModule):
    def __init__(
        self,
        model_hyperparameters: dict,
        optimizer_hyperparameters: dict,
        lambda_l1: float,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.model = make_architecture(**model_hyperparameters)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = SGD(self.model.parameters(), **optimizer_hyperparameters)
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=1, T_mult=1)
        self.lambda_l1 = lambda_l1

    def forward(self, inputs):
        return self.model(inputs)

    def step(self, step: str, batch):
        inputs, labels = batch
        outputs = self(inputs)
        outputs, labels = drop_nan(outputs, labels)
        loss = self.criterion(outputs.squeeze(1), labels)
        loss = loss + self.lambda_l1 * torch.norm(self.model.rnn.weight_ih_l0)
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

    def cfgure_optimizers(self):
        scheduler_cfg = {
            "scheduler": self.scheduler,
            "interval": "step",
            "frequency": 1,
        }
        return {"optimizer": self.optimizer, "lr_scheduler": scheduler_cfg}


def make_trainer(max_epochs: int, cfg: Config, checkpoint: bool) -> Trainer:
    callbacks: list = [RichProgressBar()]
    if checkpoint:
        checkpoint_callback = ModelCheckpoint(
            dirpath=cfg.filepaths.data.results.checkpoints,
            filename="{epoch}_{step}_{val_loss:.3f}",
            save_last=True,
            verbose=cfg.verbose,
        )
        callbacks.append(checkpoint_callback)
    logger = (
        TensorBoardLogger(save_dir=cfg.filepaths.data.results.logs)
        if cfg.log
        else False
    )
    trainer = Trainer(
        logger=logger,
        callbacks=callbacks,
        enable_checkpointing=checkpoint,
        gradient_clip_val=cfg.training.gradient_clip,
        max_epochs=max_epochs,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=cfg.logging.log_every_n_steps,
        fast_dev_run=cfg.fast_dev_run,
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
    sequence_model = partial(
        SequenceModel,
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )
    match method:
        case "rnn":
            return sequence_model(method=nn.RNN)
        case "lstm":
            return sequence_model(method=nn.LSTM)
        case "transformer":
            return Transformer(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=4,
                dropout=dropout,
                max_seq_len=4,
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
    nesterov: bool,
    momentum: float = 0.9,
    method: str = "rnn",
    hidden_dim: int = 256,
    num_layers: int = 2,
    dropout: float = 0.0,
    lr: float = 0.01,
    weight_decay: float = 0.0,
    lambda_l1: float = 0.0,
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
        lambda_l1=lambda_l1,
    )
