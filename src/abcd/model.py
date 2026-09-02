import torch
from lightning import LightningModule
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.sgd import SGD

from .config import Model, Optimizer, Scheduler

type Batch = tuple[torch.Tensor, torch.Tensor]


class Network(LightningModule):
    """LSTM over a subject's events; each step predicts the next event's p-factor quartile."""

    def __init__(
        self,
        input_dim: int,
        model: Model,
        optimizer: Optimizer,
        scheduler: Scheduler,
        ignore_index: int,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=model.hidden_dim,
            num_layers=model.num_layers,
            dropout=model.dropout,
            batch_first=True,
        )
        self.fc = nn.Linear(in_features=model.hidden_dim, out_features=model.output_dim)
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.optimizer_config = optimizer
        self.scheduler_config = scheduler

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.lstm(inputs)  # (B, T, H)
        return self.fc(hidden)  # (B, T, C)

    def step(self, stage: str, batch: Batch) -> torch.Tensor:
        inputs, labels = batch
        outputs = self(inputs).permute(0, 2, 1)  # (B, C, T)
        loss = self.criterion(outputs, labels)
        self.log(f"{stage}_loss", loss)
        return loss

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        return self.step("train", batch)

    def validation_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        return self.step("val", batch)

    def predict_step(self, batch: Batch, batch_idx: int) -> Batch:
        inputs, labels = batch
        return self(inputs), labels

    def configure_optimizers(self) -> dict:
        optimizer = SGD(self.parameters(), **self.optimizer_config.model_dump())
        scheduler = ReduceLROnPlateau(optimizer, **self.scheduler_config.model_dump())
        lr_scheduler = {"scheduler": scheduler, "monitor": "val_loss"}
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
