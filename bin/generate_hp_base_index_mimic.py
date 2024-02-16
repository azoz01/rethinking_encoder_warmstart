import numpy as np
import pandas as pd
import shutil
import pickle as pkl
import typer

from dataset2vec.train import LightningWrapper as D2vWrapper
from liltab.train.utils import LightningWrapper as LiltabWrapper
from loguru import logger
from itertools import starmap
from pathlib import Path
from pytorch_lightning import seed_everything
from scipy.stats import rankdata
from typing_extensions import Annotated
from torch import Tensor, no_grad
from tqdm import tqdm

TRAINING_DATASETS = [
    "alcohol_diagnosed-10",
    "hypotension_diagnosed-10",
    "diabetes_diagnosed-10",
    "anemia_diagnosed-10",
    "purpura_diagnosed-10",
    "respiratory_diagnosed-10",
    "atrial_diagnosed-10",
    "hypertensive_diagnosed-10",
    "overweight_diagnosed-10",
    "lipoid_diagnosed-10",
]


def task_in_training_datasets(name):
    return any([dataset in name for dataset in TRAINING_DATASETS])


app = typer.Typer()


@app.command(help="Generate hyperparameter base.")
def main(
    ho_path: Annotated[
        Path, typer.Option(..., help="Path to hyperparameter configurations")
    ] = Path("results/logistic_hpo_mimic/logistic_parameters_base.pkl"),
    scores_path: Annotated[
        Path, typer.Option(..., help="Path to hyperparameter scores across tasks")
    ] = Path("results/logistic_hpo_mimic/logistic_scores_base.pkl"),
    liltab_encoder_path: Annotated[
        Path, typer.Option(..., help="Path to liltab encoder path")
    ] = Path("models/liltab.ckpt"),
    d2v_encoder_path: Annotated[Path, typer.Option(..., help="Path to d2v encoder path")] = Path(
        "models/d2v.ckpt"
    ),
    output_results_path: Annotated[Path, typer.Option(..., help="Path to input tasks")] = Path(
        "results/hyperparameters_index_mimic"
    ),
) -> None:
    seed_everything(123)
    if output_results_path.exists():
        shutil.rmtree(output_results_path)
    output_results_path.mkdir()

    logger.info("Loading encoders")
    liltab_encoder = LiltabWrapper.load_from_checkpoint(liltab_encoder_path).model
    d2v_encoder = D2vWrapper.load_from_checkpoint(d2v_encoder_path).encoder

    encoders = {
        "liltab": lambda X, y: liltab_encoder.encode_support_set(X, y),
        "d2v": d2v_encoder,
    }

    logger.info("Loading hyperparameters with scores")
    with open(ho_path, "rb") as f:
        hpo = pkl.load(f)

    with open(scores_path, "rb") as f:
        scores = pkl.load(f)
    scores = scores["<class 'sklearn.linear_model._logistic.LogisticRegression'>"]

    logger.info("Generating ranks")
    items = list(scores.items())
    items = list(filter(lambda item: task_in_training_datasets(item[0]), items))
    best_flags = list(
        map(
            lambda x: (
                (np.array(x[1]) == np.max(x[1])) & ((np.array(x[1]) == np.max(x[1])).sum() == 1)
            ).astype(int),
            items,
        )
    )

    average_bests = np.array(
        [np.mean([b[i] for b in best_flags]) for i in range(len(best_flags[0]))]
    )
    final_ranks = rankdata(average_bests, method="ordinal")
    ranked_combinations = dict(zip(final_ranks, hpo))

    rows = [
        pd.DataFrame({"rank": [idx], "logistic_best_hpo": [hpo]})
        for idx, hpo in ranked_combinations.items()
    ]
    output_ranks = pd.concat(rows).reset_index(drop=True).sort_values("rank")
    output_ranks.to_parquet(output_results_path / "ranks.parquet")

    logger.info("Indexing optimization results using encoders")

    def generate_record(task_path, scores):
        task = task_path.split("/")[-1]
        task_path = Path(task_path)
        test = pd.read_csv(task_path / "test.csv")
        X_test, y_test = Tensor(test.values[:, :-1]), Tensor(test.values[:, -1]).unsqueeze(1)
        with no_grad():
            liltab_encoding = encoders["liltab"](X_test, y_test).detach().numpy().tolist()
            d2v_encoding = encoders["d2v"](X_test, y_test).detach().numpy().tolist()
        best_idx = np.argmax(scores)
        best_hpo = hpo[best_idx]
        best_score = scores[best_idx]
        return pd.DataFrame(
            {
                "task": [task],
                "task_path": [str(task_path)],
                "liltab_encoding": [liltab_encoding],
                "d2v_encoding": [d2v_encoding],
                "logistic_best_hpo": [best_hpo],
                "logistic_best_score": [best_score],
            }
        )

    records = list(starmap(generate_record, tqdm(items)))
    output_index = pd.concat(records).reset_index(drop=True)

    output_index.to_parquet(output_results_path / "index.parquet", index=False)


if __name__ == "__main__":
    app()
