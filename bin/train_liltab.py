import json
import os
import yaml
import pytorch_lightning as pl
import typer

from liltab.data.datasets import PandasDataset
from liltab.data.dataloaders import (
    FewShotDataLoader,
    ComposedDataLoader,
    RepeatableOutputComposedDataLoader,
)
from liltab.data.factory import ComposedDataLoaderFactory
from liltab.model.heterogenous_attributes_network import HeterogenousAttributesNetwork
from liltab.train.trainer import HeterogenousAttributesNetworkTrainer
from loguru import logger
from pathlib import Path
from torch import nn
from typing_extensions import Annotated

app = typer.Typer()


@app.command(help="Run training liltab model.")
def main(
    config_path: Annotated[
        Path,
        typer.Option(..., help="Path to experiment configuration."),
    ] = Path("config/liltab_encoder_training.yaml")
):
    pl.seed_everything(123)

    logger.info("Loading config")
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.CLoader)

    logger.info("Loading data")
    train_loader = ComposedDataLoaderFactory.create_composed_dataloader_from_path(
        Path(config["train_data_path"]),
        PandasDataset,
        {"encode_categorical_response": True},
        FewShotDataLoader,
        {"support_size": config["support_size"], "query_size": config["query_size"]},
        ComposedDataLoader,
        batch_size=config["batch_size"],
    )
    val_loader = ComposedDataLoaderFactory.create_composed_dataloader_from_path(
        Path(config["val_data_path"]),
        PandasDataset,
        {"encode_categorical_response": True},
        FewShotDataLoader,
        {"support_size": config["support_size"], "query_size": config["query_size"]},
        RepeatableOutputComposedDataLoader,
        batch_size=config["batch_size"],
    )

    logger.info("Creating model")
    model = HeterogenousAttributesNetwork(
        hidden_representation_size=config["hidden_representation_size"],
        n_hidden_layers=config["n_hidden_layers"],
        hidden_size=config["hidden_size"],
        dropout_rate=config["dropout_rate"],
        is_classifier=config["is_classifier"],
        inner_activation_function=nn.Tanh(),
    )

    results_path = Path("results") / config["name"]

    trainer = HeterogenousAttributesNetworkTrainer(
        n_epochs=config["num_epochs"],
        gradient_clipping=config["gradient_clipping"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        representation_penalty_weight=config.get("representation_penalty_weight", 0),
        early_stopping_intervals=20,
        check_val_every_n_epoch=config.get("check_val_every_n_epoch", 100),
        loss=nn.CrossEntropyLoss(),
        file_logger=True,
        tb_logger=True,
        model_checkpoints=True,
        results_path=results_path,
    )

    logger.info("Pretraining encoder")
    trainer.pretrain_encoder(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    with open(max(results_path.iterdir(), key=os.path.getctime) / "config.json", "w") as f:
        json.dump(config, f, indent=4)


if __name__ == "__main__":
    app()
