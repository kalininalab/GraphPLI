from typing import Any

import torch
from torchmetrics import MetricCollection, MeanAbsoluteError, MeanSquaredError, ExplainedVariance, Accuracy, AUROC, \
    MatthewsCorrCoef
import torch.nn.functional as F

from src.training.classification import ClassificationModel
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

from src.training.data.data import TwoGraphData


class RegressionModel(ClassificationModel):
    def __init__(self, test_names, **kwargs):
        super().__init__(test_names, **kwargs)
        self.threshold = 0.02
        if self.threshold is None:
            self.train_metrics, self.val_metrics, self.test_metrics = self._set_metrics()
        else:
            (self.train_metrics, self.val_metrics, self.test_metrics), \
                (self.train_class_metrics, self.val_class_metrics, self.test_class_metrics) = \
                self._set_metrics(threshold=self.threshold)

    def to(self, *args: Any, **kwargs: Any) -> "DeviceDtypeModuleMixin":
        super().to(*args, **kwargs)
        if self.threshold is not None:
            for metric in self.test_class_metrics:
                metric.to(*args, **kwargs)

    def _set_metrics(self, prefix: str = "", threshold: float = None, num_classes: int = 1):
        reg_metrics = MetricCollection([MeanAbsoluteError(), MeanSquaredError(), ExplainedVariance()])
        if threshold is None:
            return (
                reg_metrics.clone(prefix=prefix + "train/"),
                reg_metrics.clone(prefix=prefix + "val/"),
                [reg_metrics.clone(prefix=prefix + f"{name}/") for name in self.test_names],
            )
        return (
            reg_metrics.clone(prefix=prefix + "train/"),
            reg_metrics.clone(prefix=prefix + "val/"),
            [reg_metrics.clone(prefix=prefix + f"{name}/") for name in self.test_names],
        ), super()._set_metrics(prefix=prefix, num_classes=2, threshold=threshold)

    def loss(self, pred, target):
        return F.mse_loss(pred, target)

    def training_step(self, data: TwoGraphData, batch_idx: int = -1, dataloader_idx: int = 0) -> dict:
        ss = super().training_step(data, batch_idx)
        if self.threshold is not None:
            target = torch.where(ss["labels"] > self.threshold, 1, 0)
            self.train_class_metrics.update(ss["preds"], target)
        return ss

    def validation_step(self, data: TwoGraphData, batch_idx: int = -1, dataloader_idx: int = 0) -> dict:
        ss = super().validation_step(data, batch_idx)
        if self.threshold is not None:
            target = torch.where(ss["labels"] > self.threshold, 1, 0)
            self.val_class_metrics.update(ss["preds"], target)
        return ss

    def test_step(self, data: TwoGraphData, batch_idx: int = -1, dataloader_idx: int = 0) -> dict:
        ss = super().test_step(data, batch_idx, dataloader_idx)
        if self.threshold is not None:
            target = torch.where(ss["labels"] > self.threshold, 1, 0)
            self.test_class_metrics[dataloader_idx].update(ss["preds"], target)
        return ss

    def on_training_epoch_end(self):
        super().on_training_epoch_end()
        if self.threshold is not None:
            self.log_all(self.train_class_metrics.compute())
            self.train_class_metrics.reset()

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        if self.threshold is not None:
            self.log_all(self.val_class_metrics.compute())
            self.val_class_metrics.reset()

    def on_test_epoch_end(self):
        super().on_test_epoch_end()
        if self.threshold is not None:
            for i, name in enumerate(self.test_names):
                self.log_all(self.test_class_metrics[i].compute())
                self.test_class_metrics[i].reset()

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        raise NotImplementedError()

    def test_dataloader(self) -> EVAL_DATALOADERS:
        raise NotImplementedError()

    def val_dataloader(self) -> EVAL_DATALOADERS:
        raise NotImplementedError()

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        raise NotImplementedError()
