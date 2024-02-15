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
from sklearn.preprocessing import OneHotEncoder
from typing_extensions import Annotated
from torch import Tensor, no_grad
from tqdm import tqdm


app = typer.Typer()


@app.command(help="Generate hyperparameter base.")
def main(
    ho_path: Annotated[
        Path, typer.Option(..., help="Path to hyperparameter configurations")
    ] = Path("results/logistic_hpo_uci/logistic_parameters_base.pkl"),
    scores_path: Annotated[
        Path, typer.Option(..., help="Path to hyperparameter scores across tasks")
    ] = Path("results/logistic_hpo_uci/logistic_scores_base.pkl"),
    data_path: Annotated[Path, typer.Option(..., help="Path to tasks to index")] = Path(
        "data/uci/splitted/train"
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

    logger.info("Loading hyperparameters with scores")
    with open(ho_path, "rb") as f:
        hpo = pkl.load(f)

    with open(scores_path, "rb") as f:
        scores = pkl.load(f)
    scores = scores["<class 'sklearn.linear_model._logistic.LogisticRegression'>"]

    logger.info("Generating ranks")
    items = list(scores.items())
    best_flags = list(
        map(
            lambda x: (
                (np.array(x[1]) == np.min(x[1])) & ((np.array(x[1]) == np.min(x[1])).sum() == 1)
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
        pd.DataFrame({"rank": [idx], "logistic_best_hpo": [hpo]})
        for idx, hpo in ranked_combinations.items()
    ]
    output_ranks = pd.concat(rows).reset_index(drop=True).sort_values("rank")
    output_ranks.to_parquet(output_results_path / "ranks.parquet")

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
                "logistic_best_hpo": [best_hpo],
                "logistic_best_score": [best_score],
            }
        )

    records = list(starmap(generate_record, tqdm(items)))
    output_index = pd.concat(records).reset_index(drop=True)

    output_index.to_parquet(output_results_path / "index.parquet", index=False)


if __name__ == "__main__":
    app()
