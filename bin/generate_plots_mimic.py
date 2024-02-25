import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer

from liltab.train.utils import LightningWrapper as LiltabWrapper
from engine.dataset2vec.train import LightningWrapper as D2vWrapper
from liltab.data.datasets import PandasDataset
from loguru import logger
from pathlib import Path
from sklearn.manifold import TSNE
from pytorch_lightning import seed_everything
from torch import no_grad
from typing_extensions import Annotated

app = typer.Typer()


DATASETS = [
    "hypotension_diagnosed-10",
    "diabetes_diagnosed-10",
    "hypertensive_diagnosed-10",
]


@app.command(help="Select representation plots for MIMIC datasets.")
def main(
    input_data_path: Annotated[
        Path, typer.Option(..., help="Path to visualization input path")
    ] = Path("data/mimic/mini_holdout"),
    liltab_encoder_path: Annotated[Path, typer.Option(..., help="Path to liltab encoder")] = Path(
        "models/liltab.ckpt"
    ),
    d2v_encoder_path: Annotated[Path, typer.Option(..., help="Path to liltab encoder")] = Path(
        "models/d2v.ckpt"
    ),
    n_points_on_plot: Annotated[
        int,
        typer.Option(
            ...,
            help="Number of point on plots",
        ),
    ] = 100,
    output_plots_path: Annotated[
        Path, typer.Option(..., help="Path to output plots of representations")
    ] = Path("results/representation_plots"),
) -> None:
    output_plots_path.mkdir(exist_ok=True, parents=True)
    seed_everything(123)
    logger.info("Loading liltab encoder")
    liltab_encoder = LiltabWrapper.load_from_checkpoint(liltab_encoder_path).model

    logger.info("Generating liltab representations")
    output_frames = []
    for dataset in DATASETS:
        dataset_path = input_data_path / dataset
        for task_path in list(sorted(dataset_path.iterdir()))[:n_points_on_plot]:
            df = pd.read_csv(task_path / "test.csv")
            dataset_liltab = PandasDataset(
                df, preprocess_data=True, encode_categorical_response=True
            )
            X, y = dataset_liltab.X, dataset_liltab.y
            with no_grad():
                representation = (
                    liltab_encoder.encode_support_set(X, y).mean(axis=0).detach().numpy().tolist()
                )
            output_frames.append(
                pd.DataFrame({"representation": [representation], "label": [dataset]})
            )

    logger.info("Generating liltab TSNE embeddings.")
    concatenated = pd.concat(output_frames).reset_index(drop=True)
    X = np.array(concatenated.representation.values.tolist())
    tsne = TSNE(n_components=2, perplexity=10)
    X = tsne.fit_transform(X)
    df = concatenated.drop(columns=["representation"])
    df["x"] = X[:, 0]
    df["y"] = X[:, 1]
    df.to_csv(output_plots_path / "liltab_mimic.csv", index=False)

    logger.info("Generating liltab scatter plot.")
    plt.rcParams["figure.figsize"] = (14, 6)
    p = sns.scatterplot(x=df.x, y=df.y, hue=df.label, s=60)
    plt.legend(loc=(0, 1.05), ncol=4, frameon=False)
    p.set_ylabel("y", fontsize=22)
    p.set_xlabel("x", fontsize=22)
    p.tick_params(labelsize=18)
    p.tick_params(labelsize=13)
    plt.setp(p.get_legend().get_title(), fontsize=22)
    plt.setp(p.get_legend().get_texts(), fontsize=22)
    plt.savefig(output_plots_path / "liltab_mimic_representations.png")
    plt.clf()

    logger.info("Loading d2v encoder")
    encoder = D2vWrapper.load_from_checkpoint(d2v_encoder_path).encoder

    logger.info("Generating d2v representations")
    output_frames = []
    for dataset in DATASETS:
        dataset_path = input_data_path / dataset
        for task_path in list(sorted(dataset_path.iterdir()))[:n_points_on_plot]:
            df = pd.read_csv(task_path / "test.csv")
            dataset_liltab = PandasDataset(
                df, preprocess_data=True, encode_categorical_response=True
            )
            X, y = dataset_liltab.X, dataset_liltab.y
            with no_grad():
                representation = encoder(X, y).detach().numpy().tolist()
            output_frames.append(
                pd.DataFrame({"representation": [representation], "label": [dataset]})
            )

    logger.info("Generating d2v TSNE embeddings.")
    concatenated = pd.concat(output_frames).reset_index(drop=True)
    X = np.array(concatenated.representation.values.tolist())
    tsne = TSNE(n_components=2, perplexity=10)
    X = tsne.fit_transform(X)
    df = concatenated.drop(columns=["representation"])
    df["x"] = X[:, 0]
    df["y"] = X[:, 1]
    df.to_csv(output_plots_path / "d2v_mimic.csv", index=False)

    logger.info("Generating d2v scatter plot.")
    plt.rcParams["figure.figsize"] = (14, 6)
    p = sns.scatterplot(x=df.x, y=df.y, hue=df.label, s=60)
    plt.legend(loc=(0, 1.05), ncol=4, frameon=False)
    p.set_ylabel("y", fontsize=22)
    p.set_xlabel("x", fontsize=22)
    p.tick_params(labelsize=18)
    p.tick_params(labelsize=13)
    plt.setp(p.get_legend().get_title(), fontsize=22)
    plt.setp(p.get_legend().get_texts(), fontsize=22)
    plt.savefig(output_plots_path / "d2v_mimic_representations.png")


if __name__ == "__main__":
    app()
