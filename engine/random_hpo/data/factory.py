import pandas as pd

from pathlib import Path
from meta_tuner.data.datasets import (
    PandasDatasets,
    OpenmlPandasDatasets,
    LazyPandasDatasets,
)
from typing import List
from openml import datasets


class PandasDatasetsFactory:
    """
    Factory to automize creation of PandasDatasets from
    various sources.
    """

    @staticmethod
    def create_from_dir(path: str | Path) -> PandasDatasets:
        """
        Args:
            path (str | Path): directory path from which all *.csv files
                will be downloaded to PandasDatasets

        Returns:
            PandasDatasets: datasets wrapper with all data from directory.
        """
        datasets_paths = list(Path.glob(path, "*.csv"))
        data_names = list(map(lambda x: x.stem, datasets_paths))
        datasets = list(map(lambda x: pd.read_csv(x), datasets_paths))

        pandas_datasets = PandasDatasets(datasets, data_names)

        return pandas_datasets

    @staticmethod
    def create_from_openml(ids: int | List[int]) -> OpenmlPandasDatasets:
        """
        Args:
            id (int | List[int]): id (or array of ids) refering to
                datasets from OpenML site (https://www.openml.org/)

        Returns:
            PandasDatasets: datasets wrapper with all data specified
            by ids.
        """
        if isinstance(ids, int):
            ids = [ids]

        openml_datasets = datasets.get_datasets(
            ids, download_data=True, download_qualities=False
        )
        openml_data = list(map(lambda x: x.get_data()[0], openml_datasets))
        openml_names = list(map(lambda x: x.name, openml_datasets))

        openml_datasets = OpenmlPandasDatasets(openml_data, ids, openml_names)

        return openml_datasets

    @staticmethod
    def create_from_dir_lazy(
        path: str | Path, download_datasets: bool = True
    ) -> LazyPandasDatasets:
        """
        Args:
            path (str | Path): directory path from which all *.csv files
                will be downloaded to LazyPandasDatasets
            download_datasets (bool, optional): if True, LazyPandasDatasets will
                download data while reading from csv, default to True.

        Returns:
            LazyPandasDatasets: datasets wrapper with all data from directory.
        """
        datasets_paths = list(Path.glob(path, "*.csv"))

        pandas_datasets = LazyPandasDatasets(
            datasets_paths, download_datasets=download_datasets
        )

        return pandas_datasets
