from typing import Any, Optional

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS, LRSchedulerTypeUnion
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torchmetrics import MetricCollection, Accuracy, AUROC, MatthewsCorrCoef

from src.training.data.data import TwoGraphData
from src.training.layers.encoder.graph_encoder import GraphEncoder
from src.training.layers.encoder.pickle_encoder import PickleEncoder
from src.training.layers.encoder.pretrained_encoder import PretrainedEncoder
from src.training.layers.encoder.sweet_net_encoder import SweetNetEncoder
from src.training.layers.mlp import MLP
from src.training.lr_schedules.LWCA import LinearWarmupCosineAnnealingLR
from src.training.lr_schedules.LWCAWR import LinearWarmupCosineAnnealingWarmRestartsLR
from src.training.utils.cli import remove_arg_prefix

encoders = {
    "graph": GraphEncoder,
    "sweetnet": SweetNetEncoder,
    "pretrained": PretrainedEncoder,
    "pickle": PickleEncoder,
}


class ClassificationModel(LightningModule):
    """Model for DTI prediction as a classification problem."""

    def __init__(self, test_names, **kwargs):
        super().__init__()
        self.test_names = test_names
        self.save_hyperparameters()
        self.batch_size = kwargs["datamodule"]["batch_size"]
        self._determine_feat_method(
            kwargs["model"]["feat_method"],
            kwargs["model"]["prot"]["hidden_dim"],
            kwargs["model"]["drug"]["hidden_dim"],
        )
        self.prot_encoder = encoders[kwargs["model"]["prot"]["method"]](**kwargs["model"]["prot"])
        self.drug_encoder = encoders[kwargs["model"]["drug"]["method"]](**kwargs["model"]["drug"])
        self.mlp = MLP(input_dim=self.embed_dim, out_dim=1, **kwargs["model"]["mlp"], classify=True)

        self.train_metrics, self.val_metrics, self.test_metrics = self._set_metrics()

    def _set_metrics(self, prefix: str = "", threshold: float = 0.5, num_classes: int = 2):
        """Initialize classification metrics for this sample"""
        if num_classes == 2:
            metrics = MetricCollection(
                [
                    Accuracy(task="binary"),
                    AUROC(task="binary"),
                    MatthewsCorrCoef(task="binary"),
                    # DyBiAUROC(),
                    # DyBiMCC(),
                    # Decider(),
                    # DistOverlap(),
                ]
            )
        else:
            metrics = MetricCollection(
                [
                    Accuracy(task="multiclass"),
                    AUROC(task="multiclass"),
                    MatthewsCorrCoef(num_classes=num_classes),
                    # DistOverlap(),
                ]
            )
        return (
            metrics.clone(prefix=prefix + "train/"),
            metrics.clone(prefix=prefix + "val/"),
            [metrics.clone(prefix=prefix + f"{name}/") for name in self.test_names],
        )

    def to(self, *args: Any, **kwargs: Any) -> "DeviceDtypeModuleMixin":
        super().to(*args, **kwargs)
        for metric in self.test_metrics:
            metric.to(*args, **kwargs)

    def forward(self, prot: dict, drug: dict = None) -> dict:
        """Forward the data though the classification model"""
        if drug is None and ((isinstance(prot, list) or isinstance(prot, tuple)) and len(prot) == 2):
            prot, drug = prot
            prot["x"] = prot["x"].squeeze()
            prot["edge_index"] = prot["edge_index"].squeeze()
            prot["batch"] = prot["batch"].squeeze()
            drug["x"] = drug["x"].squeeze()
            drug["edge_index"] = drug["edge_index"].squeeze()
            drug["batch"] = drug["batch"].squeeze()
        if "prot_x" in prot:
            prot = remove_arg_prefix("prot_", prot)
            drug = remove_arg_prefix("drug_", drug)
        prot_embed, _ = self.prot_encoder(prot)
        drug_embed, _ = self.drug_encoder(drug)
        joint_embedding = self.merge_features(drug_embed, prot_embed)

        pred, embed = self.mlp(joint_embedding)
        return dict(
            pred=pred,
            embed=embed,
            prot_embed=prot_embed,
            drug_embed=drug_embed,
            joint_embed=joint_embedding,
        )

    def loss(self, pred, target):
        return F.binary_cross_entropy_with_logits(pred, target)

    def shared_step(self, data: TwoGraphData) -> dict:
        """Step that is the same for train, validation and test.

        Returns:
            dict: dict with different metrics - losses, accuracies etc. Has to contain 'loss'.
        """
        prot = remove_arg_prefix("prot_", data)
        drug = remove_arg_prefix("drug_", data)
        fwd_dict = self.forward(prot, drug)
        labels = data["label"].unsqueeze(1)

        # weights = torch.where(labels == 0, self.pos_weight.to(labels.device),
        #                       self.neg_weight.to(labels.device)).float()
        loss = self.loss(fwd_dict["pred"], labels.float())

        return dict(loss=loss, preds=torch.sigmoid(fwd_dict["pred"].detach()), labels=labels.detach())

    def _determine_feat_method(self, feat_method: str, drug_hidden_dim: int = None, prot_hidden_dim: int = None, **kwargs,):
        """Which method to use for concatenating drug and protein representations."""
        if feat_method == "concat":
            self.merge_features = self._concat
            self.embed_dim = drug_hidden_dim + prot_hidden_dim
        elif feat_method == "element_l2":
            assert drug_hidden_dim == prot_hidden_dim
            self.merge_features = self._element_l2
            self.embed_dim = drug_hidden_dim
        elif feat_method == "element_l1":
            assert drug_hidden_dim == prot_hidden_dim
            self.merge_features = self._element_l1
            self.embed_dim = drug_hidden_dim
        elif feat_method == "mult":
            assert drug_hidden_dim == prot_hidden_dim
            self.merge_features = self._mult
            self.embed_dim = drug_hidden_dim
        else:
            raise ValueError("unsupported feature method")

    def _concat(self, drug_embed: Tensor, prot_embed: Tensor) -> Tensor:
        """Concatenation."""
        return torch.cat((drug_embed, prot_embed), dim=1)

    def _element_l2(self, drug_embed: Tensor, prot_embed: Tensor) -> Tensor:
        """L2 distance."""
        return torch.sqrt(((drug_embed - prot_embed) ** 2) + 1e-6).float()

    def _element_l1(self, drug_embed: Tensor, prot_embed: Tensor) -> Tensor:
        """L1 distance."""
        return (drug_embed - prot_embed).abs()

    def _mult(self, drug_embed: Tensor, prot_embed: Tensor) -> Tensor:
        """Multiplication."""
        return drug_embed * prot_embed

    def training_step(self, data: TwoGraphData, batch_idx: int = -1, dataloader_idx: int = 0) -> dict:
        """What to do during training step."""
        ss = self.shared_step(data)
        self.train_metrics.update(ss["preds"], ss["labels"])
        self.log("train/loss", ss["loss"], batch_size=self.batch_size, add_dataloader_idx=False)
        return ss

    def validation_step(self, data: TwoGraphData, batch_idx: int = -1, dataloader_idx: int = 0) -> dict:
        """What to do during validation step. Also logs the values for various callbacks."""
        ss = self.shared_step(data)
        self.val_metrics.update(ss["preds"], ss["labels"])
        self.log("val/loss", ss["loss"], batch_size=self.batch_size, add_dataloader_idx=False)
        return ss

    def test_step(self, data: TwoGraphData, batch_idx: int = -1, dataloader_idx: int = 0) -> dict:
        """What to do during test step. Also logs the values for various callbacks."""
        ss = self.shared_step(data)
        self.test_metrics[dataloader_idx].update(ss["preds"], ss["labels"])
        self.log(f"{self.test_names[dataloader_idx]}/loss", ss["loss"], batch_size=self.batch_size, add_dataloader_idx=False)
        return ss

    def log_all(self, metrics: dict):
        """Log all metrics."""
        for k, v in metrics.items():
            self.log(k, v, add_dataloader_idx=False)

    def on_training_epoch_end(self):
        """What to do at the end of a training epoch. Logs everything."""
        metrics = self.train_metrics.compute()
        self.train_metrics.reset()
        self.log_all(metrics)

    def on_validation_epoch_end(self):
        """What to do at the end of a validation epoch. Logs everything."""
        metrics = self.val_metrics.compute()
        self.val_metrics.reset()
        self.log_all(metrics)

    def on_test_epoch_end(self):
        """What to do at the end of a test epoch. Logs everything."""
        for dataloader_idx in range(len(self.test_metrics)):
            metrics = self.test_metrics[dataloader_idx].compute()
            self.test_metrics[dataloader_idx].reset()
            self.log_all(metrics)

    def lr_scheduler_step(self, scheduler: LRSchedulerTypeUnion, metric: Optional[Any]) -> None:
        scheduler.step(self.current_epoch)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        raise NotImplementedError()

    def test_dataloader(self) -> EVAL_DATALOADERS:
        raise NotImplementedError()

    def val_dataloader(self) -> EVAL_DATALOADERS:
        raise NotImplementedError()

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        raise NotImplementedError()

    def configure_optimizers(self):
        """Configure the optimizer and/or lr schedulers"""
        opt_params = self.hparams.model["optimizer"]

        params = []
        if hasattr(self, "mlp"):
            params.append({"params": self.mlp.parameters(), "lr": opt_params["lr"]})
        if hasattr(self, "prot_encoder"):
            params.append({"params": self.prot_encoder.parameters(), "lr": opt_params["lr"]})
        if hasattr(self, "drug_encoder"):
            params.append({"params": self.drug_encoder.parameters(), "lr": opt_params["lr"]})
        if hasattr(self, "prot_node_classifier"):
            params.append({"params": self.prot_node_classifier.parameters(), "lr": opt_params["lr"]})
        if hasattr(self, "drug_node_classifier"):
            params.append({"params": self.drug_node_classifier.parameters(), "lr": opt_params["lr"]})

        optimizer = Adam(params=self.parameters(), lr=opt_params["lr"], betas=(0.9, 0.95))

        return {
            "optimizer": optimizer,
            "lr_scheduler": self.parse_lr_scheduler(optimizer, opt_params, opt_params["lr_schedule"]),
        }

    def parse_lr_scheduler(self, optimizer, opt_params, lr_params):
        """Parse learning rate scheduling based on config args"""
        lr_scheduler = {"monitor": lr_params["monitor"]}
        if lr_params["module"] == "rlrop":
            lr_scheduler["scheduler"] = ReduceLROnPlateau(
                optimizer,
                verbose=True,
                factor=lr_params["factor"],
                patience=lr_params["patience"],
            )
        elif lr_params["module"] == "lwcawr":
            lr_scheduler["scheduler"] = LinearWarmupCosineAnnealingWarmRestartsLR(
                optimizer,
                warmup_epochs=lr_params["warmup_epochs"],
                start_lr=float(lr_params["start_lr"]),
                peak_lr=float(opt_params["lr"]),
                cos_restart_dist=lr_params["cos_restart_dist"],
                cos_eta_min=float(lr_params["min_lr"]),
            )
        elif lr_params["module"] == "lwca":
            lr_scheduler["scheduler"] = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=lr_params["warmup_epochs"],
                max_epochs=lr_params["cos_restart_dist"],
                eta_min=float(lr_params["min_lr"]),
                warmup_start_lr=float(lr_params["start_lr"]),
            )
        elif lr_params["module"] == "calr":
            lr_scheduler["scheduler"] = CosineAnnealingLR(
                optimizer,
                T_max=float(lr_params["max_epochs"]),
            )
        else:
            lr_scheduler["scheduler"] = ReduceLROnPlateau(
                optimizer,
                verbose=True,
                factor=0.1,
                patience=20,
            )

        return lr_scheduler
