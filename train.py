import os
from argparse import ArgumentParser

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, RichModelSummary, RichProgressBar
from pytorch_lightning.loggers import WandbLogger
from torch_geometric.loader import DataLoader

from src.training.data.datamodules import DTIDataModule
from src.training.classification import ClassificationModel
from src.training.data.datasets import DTIDataset
from src.training.regression import RegressionModel
from src.training.utils.cli import read_config, get_git_hash

torch.multiprocessing.set_sharing_strategy("file_system")
os.environ["WANDB_CACHE_DIR"] = "/scratch/SCRATCH_SAS/roman/.cache/wandb"


models = {
    "class": ClassificationModel,
    "reg": RegressionModel,
}


def train(**kwargs):
    """Train the whole model"""
    seed_everything(kwargs["seed"])
    datamodule = DTIDataModule(**kwargs["datamodule"])
    datamodule.setup()
    datamodule.update_config(kwargs)
    test_loaders = [datamodule.test_dataloader()]
    test_names = ["test"]
    if kwargs["datamodule"].get("add_test", None) is not None:
        for name, add_data in kwargs["datamodule"]["add_test"].items():
            test_names.append(name)
            test_loaders.append(DataLoader(
                DTIDataset(add_data, kwargs["datamodule"]["exp_name"], split="test"),
                **datamodule._dl_kwargs(False)
            ))
    if len(test_loaders) == 1:
        test_loaders = test_loaders[0]

    logger = WandbLogger(
        log_model=True,
        project=kwargs["datamodule"]["exp_name"],
        name=kwargs["name"],
        # offline=True,
    )
    logger.experiment.config.update(kwargs)

    callbacks = [
        ModelCheckpoint(save_last=True, **kwargs["checkpoints"]),
        RichModelSummary(),
        RichProgressBar(),
    ]
    trainer = Trainer(
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=25,
        enable_model_summary=False,
        **kwargs["trainer"],
    )

    model = models[datamodule.config["snakemake"]["parse_dataset"]["task"]](test_names=test_names, **kwargs)

    trainer.fit(model, datamodule)
    trainer.test(ckpt_path="best", dataloaders=test_loaders)


if __name__ == "__main__":
    parser = ArgumentParser(prog="Model Trainer")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()

    orig_config = read_config(args.config)
    orig_config["git_hash"] = get_git_hash()  # to know the version of the code
    train(**orig_config)
