import torch
from lightning import LightningModule
from torch import nn
from torch.optim.sgd import SGD

from abcd.config import Config
from abcd.model import make_architecture


class AutoEncoder(LightningModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.encoder = make_architecture(cfg=cfg.model)
        self.decoder = make_architecture(cfg=cfg.model)
        self.optimizer = SGD(self.parameters(), **cfg.optimizer.model_dump())
        self.mse = nn.MSELoss()

    def forward(self, inputs):
        encoding = self.encoder(inputs)
        decoding = self.decoder(encoding)
        return encoding, decoding

    def step(self, step: str, batch: tuple[torch.Tensor, ...]):
        inputs, _, _ = batch
        _, decoding = self(inputs)
        loss = self.mse(inputs, decoding)
        self.log_dict({f"{step}_loss": loss}, prog_bar=True)
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
        inputs, _, _ = batch
        encoding, _ = self(inputs)
        return encoding

    def configure_optimizers(self):
        return {"optimizer": self.optimizer}
