import numpy as np
import pandas as pd
import torch

from copy import deepcopy
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class Dataset2VecLoader:
    def __init__(self, data_path: Path, batch_size: int = 32, n_batches: int = 100):
        self.data_path = data_path
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.Xs = [pd.read_csv(path).iloc[:, :-1] for path in data_path.iterdir()]
        self.pipelines = [get_preprocessing_pipeline() for _ in self.Xs]
        self.Xs = [
            torch.Tensor(pipeline.fit_transform(X).values)
            for pipeline, X in zip(self.pipelines, self.Xs)
        ]
        self.ys = [
            torch.Tensor(pd.read_csv(path).iloc[:, [-1]].values) for path in data_path.iterdir()
        ]
        self.n_datasets = len(self.Xs)

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        return deepcopy(self)

    def __next__(self):
        return [self.__get_single_example() for _ in range(self.batch_size)]

    def __get_single_example(self):
        datasets_ids = np.random.choice(self.n_datasets, 2, replace=False)
        dataset_1_idx = datasets_ids[0]
        dataset_2_idx = dataset_1_idx if np.random.uniform() < 0.5 else datasets_ids[1]

        X1, y1 = self.Xs[dataset_1_idx], self.ys[dataset_1_idx]
        X2, y2 = self.Xs[dataset_2_idx], self.ys[dataset_2_idx]

        rows_idx_1, features_idx_1, targets_idx_1 = self.__sample_batch_idx(X1, y1)
        X1 = X1[rows_idx_1].T[features_idx_1].T
        y1 = y1[rows_idx_1].T[targets_idx_1].T

        rows_idx_2, features_idx_2, targets_idx_2 = self.__sample_batch_idx(X2, y2)
        X2 = X2[rows_idx_2].T[features_idx_2].T
        y2 = y2[rows_idx_2].T[targets_idx_2].T

        return (X1, y1, X2, y2, dataset_1_idx == dataset_2_idx)

    def __sample_batch_idx(self, X, y):
        n_rows = X.shape[0]
        assert n_rows >= 8

        n_features = X.shape[1]
        n_targets = y.shape[1]

        max_q = min(int(np.log2(n_rows)), 8)
        q = np.random.choice(np.arange(3, max_q + 1))
        n_rows_to_select = 2**q
        rows_idx = np.random.choice(n_rows, n_rows_to_select)
        features_idx = self.__sample_random_subset(n_features)
        targets_idx = self.__sample_random_subset(n_targets)

        return rows_idx, features_idx, targets_idx

    def __sample_random_subset(self, a):
        if isinstance(a, int):
            a = np.arange(a)

        if len(a) == 1:
            return a
        subset_idx = [np.random.uniform() < 0.5 for _ in a]
        if np.sum(subset_idx) == 0:
            return a
        return a[subset_idx]


class Dataset2VecValidationLoader:
    def __init__(self, data_path: Path, batch_size: int = 32):
        loader = Dataset2VecLoader(data_path=data_path, batch_size=batch_size, n_batches=1)
        self.batch = next(loader)

    def __len__(self):
        return 1

    def __iter__(self):
        return deepcopy(self)

    def __next__(self):
        return self.batch


def get_preprocessing_pipeline():
    cat_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one-hot", OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
        ]
    ).set_output(transform="pandas")

    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    ).set_output(transform="pandas")

    pipeline = Pipeline(
        [
            (
                "transformers",
                make_column_transformer(
                    (
                        cat_pipeline,
                        make_column_selector(dtype_include=("object", "category")),
                    ),
                    (num_pipeline, make_column_selector(dtype_include=np.number)),
                ),
            )
        ]
    ).set_output(transform="pandas")
    return pipeline
