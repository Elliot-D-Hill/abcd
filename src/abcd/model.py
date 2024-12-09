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
        self.positional_encoding = nn.Parameter(torch.randn(1, max_seq_len, input_dim))
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
        x = x + self.positional_encoding
        mask = generate_mask(x.size(1)).to(x.device)
        padding_mask = (x == 0.0).all(dim=-1)
        out = self.input_fc(x)
        out = self.transformer_encoder(
            out, mask=mask, src_key_padding_mask=padding_mask
        )
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
    def __init__(self, cfg: Config):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.model = make_architecture(cfg=cfg.model)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = SGD(
            self.model.parameters(), **cfg.optimizer.model_dump(exclude={"lambda_l1"})
        )
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=1, T_mult=1)
        self.lambda_l1 = cfg.optimizer.lambda_l1

    def forward(self, inputs):
        return self.model(inputs)

    # def l1_loss(self):
    #     match self.model:
    #         case SequenceModel():
    #             return self.lambda_l1 * torch.norm(self.model.rnn.weight_ih_l0)
    #         case MultiLayerPerceptron():
    #             return self.lambda_l1 * torch.norm(self.model.mlp[0].weight)
    #         case _:
    #             raise ValueError(
    #                 f"L1 regularization not implemented for {type(self.model)}"
    #             )

    def step(self, step: str, batch):
        inputs, labels = batch
        outputs = self(inputs).flatten(0, 1)
        labels = labels.flatten()
        outputs, labels = drop_nan(outputs, labels)
        loss = self.criterion(outputs, labels)
        loss = loss  # + self.l1_loss()
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
        scheduler_cfg = {
            "scheduler": self.scheduler,
            "interval": "step",
            "frequency": 1,
        }
        return {"optimizer": self.optimizer, "lr_scheduler": scheduler_cfg}


def make_trainer(cfg: Config, checkpoint: bool) -> Trainer:
    callbacks: list = [RichProgressBar()]
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
        enable_checkpointing=checkpoint,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=cfg.logging.log_every_n_steps,
        fast_dev_run=cfg.fast_dev_run,
        check_val_every_n_epoch=1,
        **cfg.trainer.model_dump(),
    )
    return trainer


def make_architecture(cfg: Model):
    hparams = cfg.model_dump(exclude={"method"})
    sequence_model = partial(SequenceModel, **hparams)
    match cfg.method:
        case "rnn":
            return sequence_model(method=nn.RNN)
        case "lstm":
            return sequence_model(method=nn.LSTM)
        case "transformer":
            return Transformer(
                num_heads=4, max_seq_len=2, **hparams
            )  # FIXME: max_seq_len
        case "mlp":
            return MultiLayerPerceptron(**hparams)
        case _:
            raise ValueError(
                f"Invalid method '{cfg.method}'. Choose from: 'rnn', 'lstm', 'mlp', 'moe', or 'transformer'"
            )
