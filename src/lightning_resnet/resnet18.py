# adapted from https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/cifar10-baseline.html # noqa: E501
import logging

import lightning as L
import torch
import torch.nn.functional as F
from torch import nn
from torchmetrics.functional import accuracy
from torchvision.models.resnet import ResNet

logger = logging.Logger(__name__)


def create_model(
    repo_or_dir: str, model: str, pretrained: bool, num_classes: int
) -> ResNet:
    if pretrained:
        raise NotImplementedError("TODO: conv1")
        # logger.warning("using pretrained weights for IMAGENET1K_V1")
        # weights = ResNet18_Weights.IMAGENET1K_V1
    else:
        logger.warning("random init of weights")
        weights = None

    model = torch.hub.load(
        repo_or_dir=repo_or_dir,
        model=model,
        weights=weights,
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
        pretrained: bool = False,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.model: ResNet = create_model(
            repo_or_dir=repo_or_dir,
            model=model,
            pretrained=pretrained,
            num_classes=num_classes,
        )
        self.num_classes = num_classes

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
