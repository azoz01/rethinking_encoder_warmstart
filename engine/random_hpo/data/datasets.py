import pandas as pd

from typing import List, Generator, override
from pathlib import Path


class PandasDatasets:
    """
    Wrapper to list of the datasets. Allows to refer to
    items by slices, list of integers and list of datasets names.
    """

    def __init__(
        self,
        datasets: List[pd.DataFrame],
        datasets_names: List[str] = None,
    ) -> None:
        """
        Args:
            datasets (List[pd.DataFrame]): list of pandas datasets
            datasets_names (List[str], optional): names of datasets, optional. Defaults to None.
        """
        super().__init__()
        self.datasets = datasets
        self.datasets_names = datasets_names

        self.__check_init()

    def __check_init(self) -> None:
        if self.datasets_names:
            assert len(self.datasets) == len(self.datasets_names)

    def __iter__(self) -> Generator[pd.DataFrame, None, None]:
        for i in range(len(self.datasets)):
            yield self[i]

    def __getitem__(
        self, items: int | str | slice | List[int | str]
    ) -> pd.DataFrame | List[pd.DataFrame]:
        try:
            if isinstance(items, slice):
                return self.datasets[items]
            elif isinstance(items, int):
                return self.datasets[items]
            elif isinstance(items, str):
                return self.datasets[self.datasets_names.index(items)]
            elif isinstance(items, list):
                if isinstance(items[0], int):
                    return [self.datasets[i] for i in items]
                elif isinstance(items[0], str):
                    return [self.datasets[self.datasets_names.index(i)] for i in items]
        except IndexError:
            raise IndexError("Provided index is out of range.")
        except ValueError:
            raise IndexError("Provided dataset name does not exist.")

    def __len__(self):
        return len(self.datasets)

    def to_dir(self, dir_path: str | Path, parents: bool = False) -> None:
        dir_path = Path(dir_path)

        dir_path.mkdir(parents=parents, exist_ok=False)

        if self.datasets_names is not None:
            files_names = [f"{name}.csv" for name in self.datasets_names]
        else:
            files_names = [f"data_{i}.csv" for i in range(len(self.datasets))]

        for df, name in zip(self, files_names):
            df.to_csv(dir_path / name)


class OpenmlPandasDatasets(PandasDatasets):
    """
    Extension to PandasDatasets. Implements refering
    to datasets by their OpenML indexes.
    """

    def __init__(
        self,
        datasets: List[pd.DataFrame],
        openml_ids: List[int],
        datasets_names: List[str] = None,
    ) -> None:
        """
        Args:
            datasets (List[pd.DataFrame]): list of pandas dataframes
            openml_ids (List[int]): list of OpenML ids
            datasets_names (List[str], optional): names of datasets, optional. Defaults to None.
        """
        super().__init__(datasets, datasets_names)
        self.openml_ids = openml_ids
        self.__check_init()

    @override
    def __check_init(self) -> None:
        assert len(self.openml_ids) == len(self.datasets)
        if self.datasets_names:
            assert len(self.datasets) == len(self.datasets_names)

    def __getitem__(
        self, items: int | str | slice | List[int | str]
    ) -> pd.DataFrame | List[pd.DataFrame]:
        return super().__getitem__(items)

    class _IndexWrapper:
        """
        Index wrapper to perform multiindexing in
        pandas way.
        """

        def __init__(self, obj, openml_ids) -> None:
            self._obj = obj
            self._openml_ids = openml_ids

        def __getitem__(self, items):
            try:
                if isinstance(items, int):
                    return self._obj[self._openml_ids.index(items)]
                elif isinstance(items, list):
                    if isinstance(items[0], int):
                        return self._obj[[self._openml_ids.index(i) for i in items]]
            except ValueError:
                raise IndexError("Provided index is not in present.")

    @property
    def oml_loc(self) -> List[pd.DataFrame] | pd.DataFrame:
        """
        Property-like object to implement pandas-like
        solution to multiple indexing functionality.

        Returns:
            List[pd.DataFrame] | pd.DataFrame: subset of datasets.
        """
        return self._IndexWrapper(self, self.openml_ids)


class LazyPandasDatasets(PandasDatasets):
    """
    Extension to PandasDatasets. Allows to lazy downloading
    datasets from directory and not storing data in memory.
    """

    def __init__(
        self,
        datasets_paths: List[str | Path],
        download_datasets: bool = True,
    ) -> None:
        """
        Args:
            datasets_paths (List[str  |  Path]): path to files
            download_datasets (bool, optional): If True, datasets will
                not be downloaded to object. Defaults to True.
        """
        datasets_path_cls = [Path(path) for path in datasets_paths]
        self.datasets_paths = datasets_path_cls
        self.download_datasets = download_datasets
        self.datasets_names = [path.stem for path in self.datasets_paths]
        self.datasets = [None for _ in self.datasets_paths]

    @override
    def __getitem__(
        self, items: int | str | slice | List[int | str]
    ) -> pd.DataFrame | List[pd.DataFrame]:
        try:
            if isinstance(items, slice):
                return self.__evaluate_datasets(items)
            elif isinstance(items, int):
                return self.__evaluate_datasets(items)
            elif isinstance(items, str):
                idx = self.datasets_names.index(items)
                return self.__evaluate_datasets(idx)
            elif isinstance(items, list):
                if isinstance(items[0], int):
                    return self.__evaluate_datasets(items)
                elif isinstance(items[0], str):
                    idxs = [self.datasets_names.index(i) for i in items]
                    return self.__evaluate_datasets(idxs)
        except IndexError:
            raise IndexError("Provided index is out of range.")
        except ValueError:
            raise IndexError("Provided dataset name does not exist.")

    def __evaluate_datasets(
        self, items: List[int] | int | slice
    ) -> pd.DataFrame | List[pd.DataFrame]:
        if isinstance(items, int):
            return self.__evaluate_dataset(items)
        elif isinstance(items, slice) or isinstance(items, list):
            if isinstance(items, slice):
                items = list(range(items.stop)[items])
            dfs = []
            for item in items:
                dfs.append(self.__evaluate_dataset(item))
            return dfs

    def __evaluate_dataset(self, items: int):
        if self.datasets[items] is not None:
            return self.datasets[items]
        else:
            df = pd.read_csv(self.datasets_paths[items])
            if self.download_datasets:
                self.datasets[items] = df
            return df
