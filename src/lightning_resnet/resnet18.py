# adapted from https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/cifar10-baseline.html # noqa: E501
import logging
from collections import OrderedDict
from typing import Optional

import lightning as L
import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch import nn
from torchmetrics.functional import accuracy
from torchvision.models.resnet import ResNet

logger = logging.Logger(__name__)


def state_dict_from_safetensors(sf_filepath: str) -> OrderedDict:
    state_dict = OrderedDict()
    with safe_open(sf_filepath, framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
    return state_dict


def create_model(
    repo_or_dir: str,
    model: str,
    safetensors_path: Optional[str],
    num_classes: int,
) -> ResNet:

    logger.warning("random init of weights")
    model = torch.hub.load(
        repo_or_dir=repo_or_dir,
        model=model,
        weights=None,
        num_classes=num_classes,
    )

    # adjust model to work with cifar10 size iamges
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
    )
    model.maxpool = nn.Identity()

    return model


class ResNet18(L.LightningModule):
    def __init__(
        self,
        num_classes: int,
        repo_or_dir: str = "pytorch/vision:v0.10.0",
        model: str = "resnet18",
        safetensors_path: Optional[str] = None,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.model: ResNet = create_model(
            repo_or_dir=repo_or_dir,
            model=model,
            safetensors_path=safetensors_path,
            num_classes=num_classes,
        )
        self.num_classes = num_classes

        if safetensors_path:
            logger.warning(f"loading state_dict from {safetensors_path}")
            state_dict = state_dict_from_safetensors(
                sf_filepath=safetensors_path
            )
            self.load_state_dict(state_dict, strict=True)

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(
            preds, y, task="multiclass", num_classes=self.num_classes
        )

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        # using CLI to override
        raise NotImplementedError()
