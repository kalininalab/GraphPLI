import os
import random

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, RichModelSummary, RichProgressBar
from pytorch_lightning.loggers import WandbLogger

from src.data.datamodules import DTIDataModule
from src.models.dti.classification import ClassificationModel
from src.utils.cli import read_config, get_git_hash

torch.multiprocessing.set_sharing_strategy("file_system")
os.environ["WANDB_CACHE_DIR"] = "/scratch/SCRATCH_SAS/roman/.cache/wandb"


models = {
    "class": ClassificationModel,
}


def train(**kwargs):
    """Train the whole model"""
    seed_everything(kwargs["seed"])
    seeds = random.sample(range(1, 100), kwargs["runs"])

    folder = os.path.join(
        "tb_logs",
        f"dti_{kwargs['datamodule']['exp_name']}",
        f"{kwargs['datamodule']['filename'].split('/')[-1].split('.')[0]}",
    )
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    if len(os.listdir(folder)) == 0:
        next_version = 0
    else:
        next_version = str(
            int(
                [d for d in os.listdir(folder) if "version" in d and os.path.isdir(os.path.join(folder, d))][-1].split(
                    "_"
                )[1]
            )
            + 1
        )

    for i, seed in enumerate(seeds):
        print(f"Run {i+1} of {kwargs['runs']} with seed {seed}")
        kwargs["seed"] = seed
        single_run(folder, next_version, **kwargs)


def single_run(folder, version, **kwargs):
    """Does a single run."""
    seed_everything(kwargs["seed"])
    datamodule = DTIDataModule(**kwargs["datamodule"])
    datamodule.setup()
    datamodule.update_config(kwargs)

    logger = WandbLogger(
        log_model='all',
        project="glylec_dti",
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

        limit_train_batches=50,
        limit_val_batches=50,
        limit_test_batches=50,

        log_every_n_steps=25,
        enable_model_summary=False,
        **kwargs["trainer"],
    )

    weights = [1, 1]

    model = ClassificationModel(pos_weight=weights[0], neg_weight=weights[1], **kwargs)

    trainer.fit(model, datamodule)
    trainer.test(ckpt_path="best", datamodule=datamodule)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(prog="Model Trainer")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()

    orig_config = read_config(args.config)
    orig_config["git_hash"] = get_git_hash()  # to know the version of the code
    train(**orig_config)
