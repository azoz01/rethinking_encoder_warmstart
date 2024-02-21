import json
import pickle as pkl
import shutil
from itertools import starmap
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from liltab.train.utils import LightningWrapper as LiltabWrapper
from loguru import logger
from pytorch_lightning import seed_everything
from scipy.stats import rankdata
from torch import Tensor, no_grad
from tqdm import tqdm
from typing_extensions import Annotated

from engine.dataset2vec.train import LightningWrapper as D2vWrapper


def process_hpo_base(hpo, scores, encoders, tasks_to_index):
    logger.info("Generating ranks")
    items = list(scores.items())
    items = list(filter(lambda item: item[0].split("/")[-2] in tasks_to_index, items))
    best_flags = list(
        map(
            lambda x: (
                (np.array(x[1]) == np.max(x[1]))
                & ((np.array(x[1]) == np.max(x[1])).sum() <= 3)
            ).astype(int),
            items,
        )
    )

    average_bests = np.array(
        [np.mean([b[i] for b in best_flags]) for i in range(len(best_flags[0]))]
    )
    final_ranks = rankdata(-average_bests, method="ordinal")
    ranked_combinations = dict(zip(final_ranks, hpo))

    rows = [
        pd.DataFrame({"rank": [idx], "best_hpo": [hpo]})
        for idx, hpo in ranked_combinations.items()
    ]
    output_ranks = pd.concat(rows).reset_index(drop=True).sort_values("rank")

    logger.info("Indexing optimization results using encoders")

    def generate_record(task_path, scores):
        task = task_path.split("/")[-1]
        task_path = Path(task_path)
        test = pd.read_csv(task_path / "test.csv")
        X_test, y_test = Tensor(test.values[:, :-1]), Tensor(
            test.values[:, -1]
        ).unsqueeze(1)
        with no_grad():
            liltab_encoding = (
                encoders["liltab"](X_test, y_test).detach().numpy().tolist()
            )
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
                "best_hpo": [best_hpo],
                "best_score": [best_score],
            }
        )

    records = list(starmap(generate_record, tqdm(items)))
    output_index = pd.concat(records).reset_index(drop=True)

    return output_index, output_ranks


app = typer.Typer()


@app.command(help="Generate hyperparameter base.")
def main(
    hpo_base_path: Annotated[
        Path, typer.Option(..., help="Path to hyperparameter optimisation results")
    ] = Path("results/hpo_mimic"),
    liltab_encoder_path: Annotated[
        Path, typer.Option(..., help="Path to liltab encoder path")
    ] = Path("models/liltab.ckpt"),
    d2v_encoder_path: Annotated[
        Path, typer.Option(..., help="Path to d2v encoder path")
    ] = Path("models/d2v.ckpt"),
    output_results_path: Annotated[
        Path, typer.Option(..., help="Path to input tasks")
    ] = Path("results/hyperparameters_index_mimic"),
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

    with open("config/mimic_splits.json") as f:
        folds = json.load(f)
    for i, fold in enumerate(folds):
        logger.info(f"Processing fold {i}")
        logger.info("Processing logistic regression")
        with open(hpo_base_path / "logistic_parameters_base.pkl", "rb") as f:
            hpo = pkl.load(f)

        with open(hpo_base_path / "logistic_scores_base.pkl", "rb") as f:
            scores = pkl.load(f)[
                "<class 'sklearn.linear_model._logistic.LogisticRegression'>"
            ]

        output_index_logistic, output_ranks_logistic = process_hpo_base(
            hpo, scores, encoders, fold["train_tasks"]
        )
        output_index_logistic = output_index_logistic.rename(
            columns={
                "best_hpo": "best_hpo_logistic",
                "best_score": "best_score_logistic",
            }
        )
        output_ranks_logistic = output_ranks_logistic.rename(
            columns={"best_hpo": "best_hpo_logistic"}
        )

        output_index_logistic.to_parquet(
            output_results_path / f"index_fold_{i}.parquet", index=False
        )
        output_ranks_logistic.to_parquet(
            output_results_path / f"ranks_fold_{i}.parquet", index=False
        )


if __name__ == "__main__":
    app()
