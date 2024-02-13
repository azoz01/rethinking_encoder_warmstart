import numpy as np
import pandas as pd
import shutil
import typer

from loguru import logger
from pathlib import Path
from pytorch_lightning import seed_everything
from sklearn.preprocessing import OneHotEncoder
from typing_extensions import Annotated


def sample_random_subset(a):
    if isinstance(a, int):
        a = np.arange(a)

    if len(a) == 1:
        return a
    subset_idx = [np.random.uniform() < 0.5 for _ in a]
    if np.sum(subset_idx) == 0:
        return a
    return a[subset_idx]


def sample_data(X, y):
    n_rows = X.shape[0]
    assert n_rows >= 8

    n_features = X.shape[1]
    n_targets = y.shape[1]

    max_q = min(int(np.log2(n_rows)), 8)
    q = np.random.choice(np.arange(3, max_q + 1))
    n_rows_to_select = 2**q
    rows_idx = np.random.choice(n_rows, n_rows_to_select)
    features_idx = sample_random_subset(n_features)
    targets_idx = sample_random_subset(n_targets)

    X = X.iloc[rows_idx, features_idx]
    y = y.iloc[rows_idx, targets_idx]

    return X, y


def generate_tasks(path, root_output_path: Path):
    df = pd.read_csv(path)
    X, y = df.iloc[:, :-1], df.iloc[:, [-1]]
    ohe = OneHotEncoder(sparse_output=False)
    ohe.set_output(transform="pandas")
    if y.nunique().values > 2:
        y = ohe.fit_transform(y)

    output_path = root_output_path / "/".join(str(path).split("/")[4:])
    output_path = output_path.with_suffix("")
    output_path.mkdir(parents=True, exist_ok=True)
    for i in range(100):
        (output_path / str(i)).mkdir()
        X_sample, y_sample = sample_data(X, y)
        X_sample.to_csv(output_path / str(i) / "X.csv", index=False)
        y_sample.to_csv(output_path / str(i) / "y.csv", index=False)


app = typer.Typer()


@app.command(help="Generate subsets of UCI set which will be used for visualization")
def main(
    raw_samples_path: Annotated[
        Path, typer.Option(..., help="Path to source data with train and test splits")
    ] = Path("data/uci/raw"),
    root_output_path: Annotated[Path, typer.Option(..., help="Path to output tasks")] = Path(
        "data/uci/visualization"
    ),
) -> None:
    seed_everything(123)
    if root_output_path.exists():
        shutil.rmtree(root_output_path)

    for sample_path in raw_samples_path.iterdir():
        for dataset in sample_path.iterdir():
            logger.info(dataset)
            generate_tasks(dataset, root_output_path)


if __name__ == "__main__":
    app()
