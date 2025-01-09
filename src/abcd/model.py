from functools import partial

import torch
from lightning import LightningModule
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim.sgd import SGD
from torchmetrics import AUROC
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


class Network(LightningModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.model = make_architecture(cfg=cfg.model)
        self.criterion = nn.CrossEntropyLoss(
            reduction="none", ignore_index=cfg.evaluation.ignore_index
        )
        self.optimizer = SGD(self.parameters(), **cfg.optimizer.model_dump())
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=1, T_mult=1)
        self.propensity = cfg.experiment.analysis == "propensity"
        self.auroc = AUROC(
            num_classes=cfg.model.output_dim,
            task="multiclass",
            ignore_index=cfg.evaluation.ignore_index,
        )

    def forward(self, inputs):
        return self.model(inputs)

    def step(self, step: str, batch: tuple[torch.Tensor, ...]):
        inputs, labels, propensity = batch
        outputs: torch.Tensor = self(inputs)
        loss: torch.Tensor = self.criterion(outputs, labels)
        if self.propensity:
            loss = loss * propensity
        loss = loss.mean()
        self.log(f"{step}_loss", loss, prog_bar=True, sync_dist=True)
        return loss

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
        outputs = self(inputs)
        return outputs, labels

    def configure_optimizers(self):
        scheduler_cfg = {
            "scheduler": self.scheduler,
            "interval": "epoch",
            "frequency": 1,
        }
        return {"optimizer": self.optimizer, "lr_scheduler": scheduler_cfg}


class AutoEncoderClassifer(LightningModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.encoder = nn.Linear(
            in_features=cfg.model.input_dim, out_features=cfg.model.encoding_dim
        )
        self.decoder = nn.Linear(
            in_features=cfg.model.encoding_dim, out_features=cfg.model.input_dim
        )
        self.model = make_architecture(cfg=cfg.model, input_dim=cfg.model.encoding_dim)
        self.optimizer = SGD(self.parameters(), **cfg.optimizer.model_dump())
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=1, T_mult=1)
        self.cros_entropy = nn.CrossEntropyLoss(
            ignore_index=cfg.evaluation.ignore_index
        )
        self.mse = nn.MSELoss()
        self.auroc = AUROC(
            num_classes=cfg.model.output_dim,
            task="multiclass",
            ignore_index=cfg.evaluation.ignore_index,
        )
        self.save_hyperparameters()

    def forward(self, encoding):
        return self.model(encoding)

    def step(self, step: str, batch: tuple[torch.Tensor, ...]):
        inputs, labels, _ = batch
        encoding = self.encoder(inputs)
        decoding = self.decoder(encoding)
        mse_loss = self.mse(inputs, decoding)
        outputs = self(encoding)
        ce_loss = self.cros_entropy(outputs, labels)
        loss = ce_loss + mse_loss
        self.log(f"{step}_loss", loss, prog_bar=True, sync_dist=True)
        return loss

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
        encoding = self.encoder(inputs)
        outputs = self(encoding)
        return outputs, labels

    def configure_optimizers(self):
        scheduler_cfg = {
            "scheduler": self.scheduler,
            "interval": "epoch",
            "frequency": 1,
        }
        return {"optimizer": self.optimizer, "lr_scheduler": scheduler_cfg}


def make_architecture(cfg: Model, input_dim: int | None = None):
    if input_dim is not None:
        hparams = cfg.model_dump(exclude={"input_dim", "method", "encoding_dim"})
        hparams["input_dim"] = input_dim
    else:
        hparams = cfg.model_dump(exclude={"method", "encoding_dim"})
    sequence_model = partial(SequenceModel, **hparams)
    match cfg.method:
        case "linear":
            return nn.Linear(
                in_features=hparams["input_dim"], out_features=cfg.output_dim
            )
        case "mlp":
            return MultiLayerPerceptron(**hparams)
        case "rnn":
            return sequence_model(method=nn.RNN)
        case "lstm":
            return sequence_model(method=nn.LSTM)
        case "transformer":
            return Transformer(num_heads=4, **hparams)
        case _:
            raise ValueError(
                f"Invalid method '{cfg.method}'. Choose from: 'linear', 'rnn', 'lstm', 'mlp', or 'transformer'"
            )
