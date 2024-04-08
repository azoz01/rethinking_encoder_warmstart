import json
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import warnings

from itertools import chain
from functools import reduce
from operator import add
from liltab.train.utils import LightningWrapper as LiltabWrapper
from loguru import logger
from pathlib import Path
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

DATA_PATH = Path("data/mimic/mini_holdout")
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
    idx = np.random.choice(len(dataset))
    return dataset[idx]


def sample_from_same_datasets(datasets):
    dataset_idx = np.random.choice(len(datasets))
    dataset = datasets[dataset_idx]
    task_1_idx, task_2_idx = np.random.choice(len(dataset), size=2)
    return dataset[task_1_idx], dataset[task_2_idx], 1


def sample_from_two_datasets(datasets):
    dataset_1_idx, dataset_2_idx = np.random.choice(len(datasets), size=2)
    dataset_1, dataset_2 = datasets[dataset_1_idx], datasets[dataset_2_idx]
    task_1_idx = np.random.choice(len(dataset_1))
    task_2_idx = np.random.choice(len(dataset_2))

    return dataset_1[task_1_idx], dataset_2[task_2_idx], 0


def main():
    pl.seed_everything(123)

    logger.info("Loading data")
    tasks_paths = list(sorted(DATA_PATH.rglob("*test.csv")))
    datasets = dict()

    for task_path in tqdm(tasks_paths):
        task_name = str(task_path).split("/")[-3]
        if task_name not in datasets:
            datasets[task_name] = []
        df = pd.read_csv(task_path)
        X, y = Tensor(
            get_preprocessing_pipeline().fit_transform(df.iloc[:, :-1]).values
        ), Tensor(df.iloc[:, -1].values).reshape(-1, 1)
        datasets[task_name].append((X, y))

    datasets = list(datasets.values())

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
    for _ in tqdm(range(10)):
        calibration_tasks = [
            get_sample_for_accuracy(datasets) for _ in range(1000)
        ]
        evaluation_tasks = [
            get_sample_for_accuracy(datasets) for _ in range(10000)
        ]
        for encoder_name in ["d2v", "liltab"]:
            encoder = encoders[encoder_name]
            with torch.no_grad():
                accuracy = get_encoder_accuracy(
                    calibration_tasks, evaluation_tasks, encoder
                )
            metrics[encoder_name]["accuracy"].append(accuracy)

    logger.info("Calculating CH index")
    for _ in tqdm(range(10)):
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
    with open(OUTPUT_PATH / "mimic.json", "w") as f:
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
    with open(OUTPUT_PATH / "mimic_summary.json", "w") as f:
        json.dump(metrics_summary, f, indent=4)


if __name__ == "__main__":
    main()
