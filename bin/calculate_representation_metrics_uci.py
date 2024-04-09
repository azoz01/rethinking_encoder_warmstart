import json
import warnings
from functools import reduce
from itertools import chain
from operator import add
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from liltab.train.utils import LightningWrapper as LiltabWrapper
from loguru import logger
from torch import Tensor
from tqdm import tqdm

from engine.dataset2vec.data import get_preprocessing_pipeline
from engine.dataset2vec.train import LightningWrapper as D2vWrapper
from engine.representation_metrics import (
    get_encoder_accuracy,
    get_encoder_ch_index,
    get_metrics_summary,
)

warnings.simplefilter("ignore")

DATA_PATH = Path("data/uci/raw")
LILTAB_ENCODER_PATH = "models/liltab.ckpt"
D2V_ENCODER_PATH = "models/d2v.ckpt"
NUMBER_OF_POINTS_PER_DATASET = 100
OUTPUT_PATH = Path("results/representation_metrics")


def get_sample_for_accuracy(datasets):
    if np.random.uniform() <= 0.5:
        return sample_from_same_datasets(datasets)
    else:
        return sample_from_two_datasets(datasets)


def get_sample_for_ch(dataset):
    X, y = dataset
    rows_idx = sample_with_random_size(X.shape[0]).tolist()
    cols_idx = sample_with_random_size(X.shape[1]).tolist()
    return index_tensor(X, rows_idx, cols_idx), y[rows_idx]


def sample_from_same_datasets(datasets):
    idx = np.random.choice(len(datasets))
    X, y = datasets[idx]

    all_rows_idx = np.arange(X.shape[0])

    first_rows_idx = sample_with_random_size(all_rows_idx).tolist()
    first_cols_idx = sample_with_random_size(X.shape[1]).tolist()

    second_rows_idx = sample_with_random_size(
        np.setdiff1d(all_rows_idx, first_rows_idx)
    ).tolist()
    second_cols_idx = sample_with_random_size(X.shape[1]).tolist()
    return (
        (index_tensor(X, first_rows_idx, first_cols_idx), y[first_rows_idx]),
        (
            index_tensor(X, second_rows_idx, second_cols_idx),
            y[second_rows_idx],
        ),
        1,
    )


def sample_from_two_datasets(datasets):  # -> tuple[tuple, tuple, Literal[0]]:
    idx1, idx2 = np.random.choice(len(datasets), size=2)
    X1, y1 = datasets[idx1]
    X2, y2 = datasets[idx2]

    first_rows_idx = sample_with_random_size(X1.shape[0]).tolist()
    first_cols_idx = sample_with_random_size(X1.shape[1]).tolist()

    second_rows_idx = sample_with_random_size(X2.shape[0]).tolist()
    second_cols_idx = sample_with_random_size(X2.shape[1]).tolist()

    return (
        (index_tensor(X1, first_rows_idx, first_cols_idx), y1[first_rows_idx]),
        (
            index_tensor(X2, second_rows_idx, second_cols_idx),
            y2[second_rows_idx],
        ),
        0,
    )


def sample_with_random_size(arr):
    if isinstance(arr, int):
        arr = np.arange(arr)
    size = np.random.choice(np.arange(1, len(arr)))
    return np.random.choice(arr, size=size)


def index_tensor(tensor, row_idx, col_idx):
    return tensor[row_idx].T[col_idx].T


def main():
    pl.seed_everything(123)

    logger.info("Loading data")
    dataframes = [
        pd.read_csv(data_path) for data_path in sorted(DATA_PATH.iterdir())
    ]
    datasets = [
        (
            (
                Tensor(
                    get_preprocessing_pipeline()
                    .fit_transform(df.iloc[:, :-1])
                    .values
                ),
                Tensor(df.iloc[:, -1].values).reshape(-1, 1),
            )
        )
        for df in dataframes
    ]

    logger.info("Loading encoders")
    liltab_encoder = LiltabWrapper.load_from_checkpoint(
        LILTAB_ENCODER_PATH
    ).model
    d2v_encoder = D2vWrapper.load_from_checkpoint(D2V_ENCODER_PATH).encoder
    encoders = {
        "liltab": lambda X, y: liltab_encoder.encode_support_set(X, y).mean(
            dim=0
        ),
        "d2v": d2v_encoder,
    }

    metrics = {
        "d2v": {"ch_index": [], "accuracy": []},
        "liltab": {"ch_index": [], "accuracy": []},
    }

    logger.info("Calculating accuracy")
    for _ in tqdm(range(15)):
        calibration_tasks = [
            get_sample_for_accuracy(datasets) for _ in range(1000)
        ]
        evaluation_tasks = [
            get_sample_for_accuracy(datasets) for _ in range(5000)
        ]
        for encoder_name in ["d2v", "liltab"]:
            encoder = encoders[encoder_name]
            with torch.no_grad():
                accuracy = get_encoder_accuracy(
                    calibration_tasks, evaluation_tasks, encoder
                )
            metrics[encoder_name]["accuracy"].append(accuracy)

    logger.info("Calculating CH index")
    for _ in tqdm(range(15)):
        tasks = [
            [
                get_sample_for_ch(dataset)
                for _ in range(NUMBER_OF_POINTS_PER_DATASET)
            ]
            for dataset in datasets
        ]
        tasks = list(chain(*tasks))
        labels = reduce(
            add,
            [[i] * NUMBER_OF_POINTS_PER_DATASET for i in range(len(datasets))],
        )
        for encoder_name in ["d2v", "liltab"]:
            encoder = encoders[encoder_name]
            with torch.no_grad():
                ch_index = get_encoder_ch_index(tasks, labels, encoder)
            metrics[encoder_name]["ch_index"].append(ch_index)

    logger.info("Saving results")
    OUTPUT_PATH.mkdir(exist_ok=True, parents=True)
    with open(OUTPUT_PATH / "uci.json", "w") as f:
        json.dump(metrics, f, indent=4)

    metrics_summary = {
        "d2v": {
            "ch_index": get_metrics_summary(metrics["d2v"]["ch_index"]),
            "accuracy": get_metrics_summary(metrics["d2v"]["accuracy"]),
        },
        "liltab": {
            "ch_index": get_metrics_summary(metrics["liltab"]["ch_index"]),
            "accuracy": get_metrics_summary(metrics["liltab"]["accuracy"]),
        },
    }
    with open(OUTPUT_PATH / "uci_summary.json", "w") as f:
        json.dump(metrics_summary, f, indent=4)


if __name__ == "__main__":
    main()
