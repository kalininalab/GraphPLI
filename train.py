import os
from argparse import ArgumentParser

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, RichModelSummary, RichProgressBar
from pytorch_lightning.loggers import WandbLogger
import git
import yaml

from src.training.data.datamodules import DTIDataModule
from src.training.classification import ClassificationModel

torch.multiprocessing.set_sharing_strategy("file_system")
os.environ["WANDB_CACHE_DIR"] = "/scratch/SCRATCH_SAS/roman/.cache/wandb"


def get_git_hash():
    """Get the git hash of the current repository."""
    repo = git.Repo(search_parent_directories=True)
    return repo.head.object.hexsha


def read_config(filename: str) -> dict:
    """Read in yaml config for training."""
    with open(filename, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config


def remove_arg_prefix(prefix: str, kwargs: dict) -> dict:
    """Removes the prefix from all the args.

    Args:
        prefix (str): prefix to remove (`drug_`, `prot_` or `mlp_` usually)
        kwargs (dict): dict of arguments

    Returns:
        dict: Sub-dict of arguments
    """
    new_kwargs = {}
    prefix_len = len(prefix)
    for key, value in kwargs.items():
        if key.startswith(prefix):
            new_key = key[prefix_len:]
            if new_key == "x_batch":
                new_key = "batch"
            new_kwargs[new_key] = value
    return new_kwargs


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
