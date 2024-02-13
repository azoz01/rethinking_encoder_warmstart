from pathlib import Path
from shutil import rmtree
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import typer
from loguru import logger
from pytorch_lightning import seed_everything
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing_extensions import Annotated


def get_datasets(data_path: Path) -> Dict[str, pd.DataFrame]:
    file_paths = data_path.glob("*.csv")
    datasets = {file_path.stem: pd.read_csv(file_path) for file_path in file_paths}
    return datasets


def _mini_batch_stratified(
    data_idx: pd.DataFrame,
    remaining_idx: np.ndarray,
    train_size: int,
    test_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    total_size = train_size + test_size

    stratify_value = data_idx[remaining_idx, 1]
    _, batch_idx, _, batch_clasess = train_test_split(
        data_idx[remaining_idx, 0],
        data_idx[remaining_idx, 1],
        test_size=total_size,
        stratify=stratify_value,
    )

    stratify_batch_value = batch_clasess

    train_batch, test_batch, _, _ = train_test_split(
        batch_idx,
        batch_clasess,
        test_size=test_size,
        stratify=stratify_batch_value,
    )

    return train_batch, test_batch


def _mini_batch_equally(
    data_idx: pd.DataFrame,
    remaining_idx: np.ndarray,
    train_size: int,
    test_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    class_0 = data_idx[remaining_idx, :][data_idx[remaining_idx, 1] == 0][:, 0]
    class_1 = data_idx[remaining_idx, :][data_idx[remaining_idx, 1] == 1][:, 0]

    if class_1.shape[0] < (train_size + test_size) // 2 + 1:
        raise StopIteration
    if class_0.shape[0] < (train_size + test_size) // 2 + 1:
        raise StopIteration

    train_size_0 = train_size // 2 + 1 if train_size % 2 != 0 else train_size // 2
    test_size_0 = test_size // 2 + 1 if test_size % 2 != 0 else test_size // 2

    train_idx_0 = np.random.choice(class_0, size=train_size_0, replace=False)
    train_idx_1 = np.random.choice(class_1, size=train_size // 2, replace=False)
    test_idx_0 = np.random.choice(class_0, size=test_size // 2, replace=False)
    test_idx_1 = np.random.choice(class_1, size=test_size_0, replace=False)

    train_batch = np.concatenate([train_idx_0, train_idx_1], axis=0)
    test_batch = np.concatenate([test_idx_0, test_idx_1], axis=0)

    np.random.shuffle(train_batch)
    np.random.shuffle(test_batch)

    return train_batch, test_batch


def get_mini_holdout_idx(
    data: pd.DataFrame, train_size: int, test_size: int, stratify: bool, equally: bool
) -> Tuple[np.ndarray, np.ndarray]:
    if stratify and equally:
        raise ValueError("illegal combination of the arguments")

    total_size = train_size + test_size
    data_idx = data.reset_index(drop=False).iloc[:, [0, -1]].to_numpy().copy()
    remaining_idx = data_idx[:, 0].copy()

    while remaining_idx.shape[0] >= total_size:
        if stratify:
            train_batch, test_batch = _mini_batch_stratified(
                data_idx, remaining_idx, train_size, test_size
            )

        if equally:
            try:
                train_batch, test_batch = _mini_batch_equally(
                    data_idx, remaining_idx, train_size, test_size
                )
            except StopIteration:
                return

        remaining_idx = np.setdiff1d(
            remaining_idx, np.concatenate([train_batch, test_batch], axis=0)
        )

        yield (train_batch, test_batch)


def prepare_data(
    df_train: pd.DataFrame, df_test: pd.DataFrame, encode_cat: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    if encode_cat:
        raise NotImplementedError

    X_train, y_train, X_test, y_test = (
        df_train.iloc[:, :-1],
        df_train.iloc[:, -1],
        df_test.iloc[:, :-1],
        df_test.iloc[:, -1],
    )

    imputer = SimpleImputer()
    imputer.set_output(transform="pandas")

    imputer.fit(X_train)

    X_train_transformed = imputer.transform(X_train)
    X_test_transformed = imputer.transform(X_test)

    scaler = StandardScaler()
    scaler.set_output(transform="pandas")

    scaler.fit(X_train_transformed)

    X_train_transformed = scaler.transform(X_train_transformed)
    X_test_transformed = scaler.transform(X_test_transformed)

    X_train_transformed["target"] = y_train
    X_test_transformed["target"] = y_test

    return X_train_transformed, X_test_transformed


app = typer.Typer()


@app.command(
    help="""create mini holdout tasks for regular models (e.g. linear regression).
    Each task is a directory with two files: train and test"""
)
def main(
    train_size: Annotated[int, typer.Option(..., help="size of the training set")] = 4,
    test_size: Annotated[int, typer.Option(..., help="size of the test set")] = 27,
    equally: Annotated[
        bool,
        typer.Option(
            ...,
            """
            if True, then each train and test set will have
            approximately equal number of observation from  each class
            """,
        ),
    ] = True,
    stratify: Annotated[
        bool,
        typer.Option(
            ...,
            """
            if True, then each train and test set will be
            stratified along with y value
            """,
        ),
    ] = False,
    stg_data_path: Annotated[Path, typer.Option(..., help="path to staging data")] = Path(
        "data/mimic/stg/most_important"
    ),
    out_data_path: Annotated[Path, typer.Option(..., help="Path to output data")] = Path(
        "data/mimic/mini_holdout"
    ),
) -> None:
    seed_everything(123)
    datasets = get_datasets(stg_data_path)

    rmtree(out_data_path, ignore_errors=True)
    for name, dataset in datasets.items():
        dir_path = Path(out_data_path) / name
        dir_path.mkdir(exist_ok=False, parents=True)

        for idx, idxs in enumerate(
            get_mini_holdout_idx(dataset, train_size, test_size, stratify, equally)
        ):
            train_idx, test_idx = idxs

            train_df = dataset.iloc[train_idx, :]
            test_df = dataset.iloc[test_idx, :]

            train_df, test_df = prepare_data(train_df, test_df)

            task_dir = Path(dir_path / f"task-{str(idx).zfill(5)}")
            task_dir.mkdir()

            train_df.to_csv(task_dir / "train.csv", index=False)
            test_df.to_csv(task_dir / "test.csv", index=False)

            logger.info(f"idx: {idx}")


if __name__ == "__main__":
    app()
