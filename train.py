import os
from argparse import ArgumentParser

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, RichModelSummary, RichProgressBar
from pytorch_lightning.loggers import WandbLogger

from src.training.data.datamodules import DTIDataModule
from src.training.classification import ClassificationModel
from src.training.utils.cli import read_config, get_git_hash

torch.multiprocessing.set_sharing_strategy("file_system")
os.environ["WANDB_CACHE_DIR"] = "/scratch/SCRATCH_SAS/roman/.cache/wandb"


def train(**kwargs):
    """Train the whole model"""
    seed_everything(kwargs["seed"])
    datamodule = DTIDataModule(**kwargs["datamodule"])
    datamodule.setup()
    datamodule.update_config(kwargs)

    logger = WandbLogger(
        log_model='all',
        project=kwargs["datamodule"]["exp_name"],
        name=kwargs["name"],
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

    model = ClassificationModel(**kwargs)

    trainer.fit(model, datamodule)
    trainer.test(ckpt_path="best", datamodule=datamodule)


if __name__ == "__main__":
    parser = ArgumentParser(prog="Model Trainer")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()

    orig_config = read_config(args.config)
    orig_config["git_hash"] = get_git_hash()  # to know the version of the code
    train(**orig_config)
