import pytorch_lightning as pl
import shutil
import typer
import yaml

from datetime import datetime
from loguru import logger
from pathlib import Path
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from typing_extensions import Annotated

from engine.dataset2vec.model import Dataset2Vec
from engine.dataset2vec.data import Dataset2VecLoader, Dataset2VecValidationLoader
from engine.dataset2vec.train import LightningWrapper

app = typer.Typer()


@app.command(help="Trains Dataset2Vec.")
def main(
    config_path: Annotated[Path, typer.Option(..., help="Path to configuration file.")] = Path(
        "config/dataset2vec_training.yaml"
    ),
    checkpoint_path: Annotated[Path, typer.Option(..., help="Path to checkpoint.")] = None,
):
    logger.info("Loading config")
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.CLoader)

    logger.info("Loading data")
    train_loader = Dataset2VecLoader(
        data_path=Path(config["train_data_path"]),
        batch_size=config["train_batch_size"],
        n_batches=config["train_n_batches"],
    )
    val_loader = Dataset2VecValidationLoader(
        data_path=Path(config["val_data_path"]), batch_size=config["val_batch_size"]
    )

    if checkpoint_path:
        logger.info("Loading model from checkpoint")
        wrapper = LightningWrapper.load_from_checkpoint(checkpoint_path)
    else:
        logger.info("Creating model")
        encoder = Dataset2Vec(
            f_dense_hidden_size=config["f_dense_hidden_size"],
            f_res_hidden_size=config["f_res_hidden_size"],
            f_res_n_hidden=config["f_res_n_hidden"],
            f_dense_out_hidden_size=config["f_dense_out_hidden_size"],
            f_block_repetitions=config["f_block_repetitions"],
            g_layers_sizes=config["g_layers_sizes"],
            h_dense_hidden_size=config["h_dense_hidden_size"],
            h_res_hidden_size=config["h_res_hidden_size"],
            h_res_n_hidden=config["h_res_n_hidden"],
            h_dense_out_hidden_size=config["h_dense_out_hidden_size"],
            h_block_repetitions=config["h_block_repetitions"],
        )
        wrapper = LightningWrapper(
            encoder,
            gamma=config["gamma"],
            learning_rate=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )

    logger.info("Setting up training")
    name = config["name"]
    output_path = Path("results") / name / datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    output_path.mkdir(parents=True, exist_ok=False)
    shutil.copy(config_path, output_path / "config.yaml")

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=config["early_stopping_patience"], mode="min"
    )
    checkpointing = ModelCheckpoint(
        dirpath=output_path / "checkpoints",
        filename="{epoch}-{val_loss:.2f}",
        monitor="val_loss",
        verbose=0,
        save_top_k=1,
        mode="min",
        save_on_train_epoch_end=False,
    )

    tb_logger = TensorBoardLogger(save_dir=output_path / "tensorboard", name="")

    trainer = pl.Trainer(
        max_epochs=config["max_epochs"],
        check_val_every_n_epoch=config["check_val_every_n_epoch"],
        log_every_n_steps=1,
        callbacks=[early_stopping, checkpointing],
        logger=tb_logger,
    )

    logger.info("Starting training")
    trainer.fit(wrapper, train_loader, val_loader)
    logger.info("Training finished")


if __name__ == "__main__":
    app()
