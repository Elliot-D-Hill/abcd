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
            activation_layer=nn.ReLU,
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
        self, input_dim, output_dim, hidden_dim, num_layers, num_heads, dropout
    ):
        super().__init__()
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
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(hidden_dim)
        )
        self.output_fc = nn.Linear(in_features=hidden_dim, out_features=output_dim)

    def forward(self, x: torch.Tensor):
        mask = generate_mask(x.size(1)).to(x.device).bool()
        padding_mask = (x == 0.0).all(dim=-1).bool()
        out = self.input_fc(x)
        out = self.transformer(out, mask=mask, src_key_padding_mask=padding_mask)
        out = self.output_fc(out)
        return out


class AutoEncoderClassifer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super().__init__()
        hidden_channels: list[int] = [hidden_dim] * num_layers
        self.encoder = MLP(
            in_channels=input_dim,
            hidden_channels=hidden_channels,
            activation_layer=nn.ReLU,
            norm_layer=nn.LayerNorm,
            dropout=dropout,
        )
        self.decoder = MLP(
            in_channels=hidden_dim,
            hidden_channels=hidden_channels + [input_dim],
            activation_layer=nn.ReLU,
            norm_layer=nn.LayerNorm,
            dropout=dropout,
        )
        self.linear = nn.Linear(in_features=hidden_dim, out_features=output_dim)

    def forward(self, x):
        encoding = self.encoder(x)
        decoding = self.decoder(encoding)
        output = self.linear(encoding)
        return output, decoding


class Network(LightningModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.model = make_architecture(cfg=cfg.model)
        self.criterion = nn.CrossEntropyLoss(reduction="none")
        self.optimizer = SGD(self.model.parameters(), **cfg.optimizer.dict())
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=1, T_mult=1)
        self.propensity = cfg.experiment.analysis == "propensity"
        self.mse = nn.MSELoss()

    def forward(self, inputs):
        return self.model(inputs)

    def drop_nan(
        self, outputs: torch.Tensor, labels: torch.Tensor, propensity: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        is_not_nan = ~torch.isnan(labels).flatten()
        outputs = outputs.flatten(0, 1)
        outputs = outputs[is_not_nan]
        labels = labels.flatten()
        labels = labels[is_not_nan]
        if self.propensity:
            propensity = propensity.flatten()[is_not_nan]
            return outputs, labels, propensity
        return outputs, labels, propensity

    def autoencoder_step(self, step: str, batch: tuple[torch.Tensor, ...]):
        inputs, labels, propensity = batch
        outputs, decoding = self(inputs)
        outputs, labels, propensity = self.drop_nan(outputs, labels, propensity)
        mse_loss = self.mse(inputs, decoding)
        loss = self.criterion(outputs, labels.long())
        loss = mse_loss + loss.mean()
        self.log_dict({f"{step}_loss": loss}, prog_bar=True)
        return loss

    def classifer_step(self, step: str, batch: tuple[torch.Tensor, ...]):
        inputs, labels, propensity = batch
        outputs: torch.Tensor = self(inputs)
        outputs, labels, propensity = self.drop_nan(outputs, labels, propensity)
        loss: torch.Tensor = self.criterion(outputs, labels.long())
        if self.propensity:
            loss = loss * propensity
        loss = loss.mean()
        self.log_dict({f"{step}_loss": loss}, prog_bar=True)
        return loss

    def step(self, step: str, batch: tuple[torch.Tensor, ...]):
        if isinstance(self.model, AutoEncoderClassifer):
            return self.autoencoder_step(step, batch)
        return self.classifer_step(step, batch)

    def training_step(self, batch, batch_idx):
        return self.step("train", batch)

    def validation_step(self, batch, batch_idx):
        return self.step("val", batch)

    def test_step(self, batch, batch_idx):
        self.step("test", batch)

    def predict_step(
        self, batch: tuple[torch.Tensor, ...], batch_idx, dataloader_idx=0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs, labels, _ = batch
        if isinstance(self.model, AutoEncoderClassifer):
            outputs, _ = self(inputs)
        else:
            outputs = self(inputs)
        outputs, labels, _ = self.drop_nan(outputs, labels, _)
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
    hparams = cfg.dict(exclude={"method"})
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
        case "autoencoder":
            return AutoEncoderClassifer(**hparams)
        case "transformer":
            return Transformer(num_heads=4, **hparams)
        case _:
            raise ValueError(
                f"Invalid method '{cfg.method}'. Choose from: 'rnn', 'lstm', 'mlp', 'autoencoder', or 'transformer'"
            )
