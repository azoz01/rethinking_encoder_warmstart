import json
import numpy as np
import pandas as pd
import shutil
import pickle as pkl
import typer

from engine.dataset2vec.train import LightningWrapper as D2vWrapper
from liltab.train.utils import LightningWrapper as LiltabWrapper
from loguru import logger
from itertools import starmap
from pathlib import Path
from pytorch_lightning import seed_everything
from scipy.stats import rankdata
from sklearn.preprocessing import OneHotEncoder
from typing_extensions import Annotated
from torch import Tensor, no_grad
from tqdm import tqdm


app = typer.Typer()


def process_hpo_base(hpo, scores, encoders, data_path, tasks_to_index):
    logger.info("Generating ranks")
    filtered_scores = dict(
        list(
            filter(
                lambda en: any([en[0].split("/")[-1] == task for task in tasks_to_index]),
                scores.items(),
            )
        )
    )
    items = list(filtered_scores.items())
    best_flags = list(
        map(
            lambda x: ((np.array(x[1]) == np.max(x[1]))).astype(int),
            items,
        )
    )

    average_bests = np.array(
        [np.mean([b[i] for b in best_flags]) for i in range(len(best_flags[0]))]
    )
    final_ranks = rankdata(-average_bests, method="ordinal")
    ranked_combinations = dict(zip(final_ranks, hpo))
    rows = [
        pd.DataFrame({"rank": [idx], "best_hpo": [hpo]}) for idx, hpo in ranked_combinations.items()
    ]
    output_ranks = pd.concat(rows).reset_index(drop=True).sort_values("rank")

    logger.info("Generating index")

    def generate_record(task_path, scores):
        task_path = str(data_path / "/".join(str(task_path).split("/")[-1:]))
        task = task_path.split("/")[-1]
        task_path = Path(task_path)
        test = pd.read_csv(task_path / "test.csv")
        X_test, y_test = Tensor(test.values[:, :-1]), test.iloc[:, [-1]]
        if y_test.nunique().values[0] > 2:
            oh = OneHotEncoder(sparse_output=False)
            y_test = Tensor(oh.fit_transform(y_test))
        else:
            y_test = Tensor(y_test.values)
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
                "best_hpo": [best_hpo],
                "best_score": [best_score],
            }
        )

    records = list(starmap(generate_record, tqdm(items)))
    output_index = pd.concat(records).reset_index(drop=True)

    return output_index, output_ranks


@app.command(help="Generate hyperparameter base.")
def main(
    hpo_base_path: Annotated[
        Path, typer.Option(..., help="Path to hyperparameter optimisation results")
    ] = Path("results/hpo_uci"),
    data_path: Annotated[Path, typer.Option(..., help="Path to tasks to index")] = Path(
        "data/uci/splitted"
    ),
    liltab_encoder_path: Annotated[
        Path, typer.Option(..., help="Path to liltab encoder path")
    ] = Path("models/liltab.ckpt"),
    d2v_encoder_path: Annotated[Path, typer.Option(..., help="Path to d2v encoder path")] = Path(
        "models/d2v.ckpt"
    ),
    output_results_path: Annotated[Path, typer.Option(..., help="Path to input tasks")] = Path(
        "results/hyperparameters_index_uci"
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

    with open("config/uci_splits.json") as f:
        folds = json.load(f)
    for i, fold in enumerate(folds):
        logger.info(f"Processing fold {i}")
        logger.info("Processing logistic regression")
        with open(hpo_base_path / "logistic_parameters_base.pkl", "rb") as f:
            hpo = pkl.load(f)

        with open(hpo_base_path / "logistic_scores_base.pkl", "rb") as f:
            scores = pkl.load(f)["<class 'sklearn.linear_model._logistic.LogisticRegression'>"]
        output_index_logistic, output_ranks_logistic = process_hpo_base(
            hpo, scores, encoders, data_path, fold["train_tasks"]
        )
        output_index_logistic = output_index_logistic.rename(
            columns={"best_hpo": "best_hpo_logistic", "best_score": "best_score_logistic"}
        )
        output_ranks_logistic = output_ranks_logistic.rename(
            columns={"best_hpo": "best_hpo_logistic"}
        )

        logger.info("Processing xgboost")
        with open(hpo_base_path / "xgboost_parameters_base.pkl", "rb") as f:
            hpo = pkl.load(f)

        with open(hpo_base_path / "xgboost_scores_base.pkl", "rb") as f:
            scores = pkl.load(f)["<class 'xgboost.sklearn.XGBClassifier'>"]

        output_index_xgboost, output_ranks_xgboost = process_hpo_base(
            hpo, scores, encoders, data_path, fold["train_tasks"]
        )
        output_index_xgboost = output_index_xgboost.rename(
            columns={"best_hpo": "best_hpo_xgboost", "best_score": "best_score_xgboost"}
        )
        output_ranks_xgboost = output_ranks_xgboost.rename(columns={"best_hpo": "best_hpo_xgboost"})

        output_index = output_index_logistic.merge(
            output_index_xgboost[["task", "best_hpo_xgboost", "best_score_xgboost"]],
            how="inner",
            on=["task"],
        )
        output_ranks = output_ranks_logistic.merge(
            output_ranks_xgboost, how="inner", on="rank"
        ).sort_values("rank")

        output_index.to_parquet(output_results_path / f"index_fold_{i}.parquet", index=False)
        output_ranks.to_parquet(output_results_path / f"ranks_fold_{i}.parquet", index=False)


if __name__ == "__main__":
    app()
