from functools import partial

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

from abcd.config import Config, Model


class MultiLayerPerceptron(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        hidden_channels = ([hidden_dim] * num_layers) + [output_dim]
        self.mlp = MLP(
            in_channels=input_dim,
            hidden_channels=hidden_channels,
            activation_layer=nn.SiLU,
            norm_layer=nn.LayerNorm,
            dropout=dropout,
        )

    def forward(self, x):
        return self.mlp(x)


class SequenceModel(nn.Module):
    def __init__(
        self,
        method: type[nn.Module],
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.rnn = method(
            input_size=input_dim,
            hidden_size=hidden_dim,
            dropout=dropout if num_layers > 1 else 0.0,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
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
        self.positional_embedding = nn.Embedding(
            num_embeddings=max_seq_len, embedding_dim=input_dim, padding_idx=0
        )
        hidden_dim = num_heads * round(hidden_dim / num_heads)
        self.input_fc = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
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
        mask = generate_mask(x.size(1)).to(x.device)
        padding_mask = (x == 0.0).all(dim=-1)
        indices = (~padding_mask).cumsum(dim=1) * (~padding_mask).long()
        x = x + self.positional_embedding(indices)
        out = self.input_fc(x)
        out = self.transformer_encoder(
            out, mask=mask, src_key_padding_mask=padding_mask
        )
        out = self.output_fc(out)
        return out


def make_metrics(step, loss, outputs, labels) -> dict:
    metrics = {f"{step}_loss": loss}
    if step == "val":
        auroc_score = multiclass_auroc(
            preds=outputs,
            target=labels.long(),
            num_classes=outputs.shape[-1],
            average="none",
        )
        metrics.update({"auc": auroc_score.mean().item()})
    return metrics


class Network(LightningModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.model = make_architecture(cfg=cfg.model)
        reduction = "none" if cfg.experiment.analysis == "propensity" else "mean"
        self.criterion = nn.CrossEntropyLoss(reduction=reduction)
        self.optimizer = SGD(self.model.parameters(), **cfg.optimizer.model_dump())
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=1, T_mult=1)
        self.propensity = cfg.experiment.analysis == "propensity"

    def forward(self, inputs):
        return self.model(inputs)

    def step(self, step: str, batch: tuple[torch.Tensor, ...]):
        inputs, labels = batch
        outputs: torch.Tensor = self(inputs)
        is_not_nan = ~torch.isnan(labels).flatten()
        outputs = outputs.flatten(0, 1)[is_not_nan]
        labels = labels.flatten()[is_not_nan]
        loss: torch.Tensor = self.criterion(outputs, labels)  # .long()
        if loss.isnan().any():
            print("Loss is NaN")
            exit()
        metrics = make_metrics(step, loss, outputs, labels)
        self.log_dict(metrics, prog_bar=True)
        return loss

    def propensity_step(self, step: str, batch: tuple[torch.Tensor, ...]):
        inputs, labels, propensity = batch
        outputs: torch.Tensor = self(inputs)
        is_not_nan = ~torch.isnan(labels).flatten()
        outputs = outputs.flatten(0, 1)[is_not_nan]
        labels = labels.flatten()[is_not_nan]
        loss: torch.Tensor = self.criterion(outputs, labels.long())
        propensity = propensity.flatten()[is_not_nan]
        loss = loss * (1 / (propensity + 1e-6))
        loss = loss.mean()
        metrics = make_metrics(step, loss, outputs, labels)
        self.log_dict(metrics, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        if self.propensity:
            return self.propensity_step("train", batch)
        return self.step("train", batch)

    def validation_step(self, batch, batch_idx):
        if self.propensity:
            return self.propensity_step("val", batch)
        return self.step("val", batch)

    def test_step(self, batch, batch_idx):
        if self.propensity:
            return self.propensity_step("test", batch)
        self.step("test", batch)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if self.propensity:
            inputs, labels, _ = batch
        else:
            inputs, labels = batch
        outputs = self(inputs)
        is_not_nan = ~torch.isnan(labels).flatten()
        outputs = outputs.flatten(0, 1)[is_not_nan]
        labels = labels.flatten()[is_not_nan]
        return outputs, labels

    def configure_optimizers(self):
        scheduler_cfg = {
            "scheduler": self.scheduler,
            "interval": "step",
            "frequency": 1,
        }
        return {"optimizer": self.optimizer, "lr_scheduler": scheduler_cfg}


def make_trainer(cfg: Config, checkpoint: bool) -> Trainer:
    callbacks: list = []
    if cfg.trainer.enable_progress_bar:
        callbacks.append(RichProgressBar())
    if checkpoint:
        checkpoint_callback = ModelCheckpoint(
            dirpath=cfg.filepaths.data.results.checkpoints,
            filename="{epoch}_{step}_{val_loss:.4f}",
            save_last=True,
            verbose=cfg.verbose,
            save_top_k=0,
        )
        callbacks.append(checkpoint_callback)
    logger = TensorBoardLogger(save_dir=cfg.filepaths.data.results.logs)
    logger = logger if cfg.log else False
    trainer = Trainer(
        logger=logger,
        callbacks=callbacks,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=cfg.logging.log_every_n_steps,
        fast_dev_run=cfg.fast_dev_run,
        check_val_every_n_epoch=1,
        enable_checkpointing=checkpoint,
        **cfg.trainer.dict(),
    )
    return trainer


def make_architecture(cfg: Model):
    hparams = cfg.dict(exclude={"method", "l1_lambda"})
    sequence_model = partial(SequenceModel, **hparams)
    match cfg.method:
        case "linear":
            return nn.Linear(in_features=cfg.input_dim, out_features=cfg.output_dim)
        case "mlp":
            return MultiLayerPerceptron(**hparams)
        case "rnn":
            return sequence_model(method=nn.RNN)
        case "lstm":
            return sequence_model(method=nn.LSTM)
        case "transformer":
            return Transformer(
                num_heads=4, max_seq_len=4, **hparams
            )  # FIXME: max_seq_len
        case _:
            raise ValueError(
                f"Invalid method '{cfg.method}'. Choose from: 'rnn', 'lstm', 'mlp', or 'transformer'"
            )
