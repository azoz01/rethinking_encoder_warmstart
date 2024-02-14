import pandas as pd

from openml import datasets
from typing import List, Dict, Callable

from .utils import meta_extractors


class MetaDataExtractor:
    __slots__ = {"meta_extractors"}

    def __init__(self, load_default: bool = True) -> None:
        if load_default:
            self.meta_extractors = meta_extractors
        else:
            self.meta_extractors = {}

    def add_extractor(
        self, extractor_name: str, extractor: Callable[..., float]
    ) -> None:
        self.meta_extractors[extractor_name] = extractor

    def get_from_openml(self, ids: int | List[int]) -> Dict[str, float]:
        """
        Download available metadata from OpenML. Filter out
        metadata that is not in meta_extractor keys.

        Args:
            ids (int | List[int]): id or ids of dataset(s).
            If list provided, metadata is returned in list with
            order as in ids list.

        Returns:
            Dict[str, float]: metadata
        """
        ids_ = [ids] if isinstance(ids, int) else ids

        dataset_list = datasets.get_datasets(
            dataset_ids=ids_, download_qualities=True, download_data=False
        )

        qualities = [dataset.qualities for dataset in dataset_list]

        for quality in qualities:
            self.__remove_not_allowed_meta(quality)

        if isinstance(ids, int):
            qualities = qualities[0]

        return qualities

    def get_metadata(
        self, X: pd.DataFrame, y: pd.DataFrame | pd.Series
    ) -> Dict[str, float]:
        """
        Extract metadata from pandas dataframe. Metadata
        is specified in meta_extractors dictionary.

        Args:
            X (pd.DataFrame): dataframe with features
            y (pd.DataFrame | pd.Series): dataframe or series with target value

        Returns:
            Dict[str, float]: metadata
        """

        metadata = {}

        for extractor_name, extractor in self.meta_extractors.items():
            metadata[extractor_name] = extractor(X, y)

        return metadata

    def get_missing_metadata(
        self, X: pd.DataFrame, y: pd.DataFrame | pd.Series, metadata: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Extract missing metadata.

        Args:
            X (pd.DataFrame): dataframe with features
            y (pd.DataFrame | pd.Series): dataframe or series with target value
            metadata (Dict[str, float]): metadata dictionary with missing values.
                Both None in values and missing pairs (key, value) is filled.

        Returns:
            Dict[str, float]: _description_
        """
        filled_metadata = {}

        for metadata_name, metadata_value in metadata.items():
            if metadata_name in set(self.meta_extractors.keys()):
                if metadata_value is None:
                    extractor = self.meta_extractors[metadata_name]
                    extracted_metadata_value = extractor(X, y)
                    filled_metadata[metadata_name] = extracted_metadata_value
                else:
                    filled_metadata[metadata_name] = metadata_value

        missing_metadata = set(self.meta_extractors.keys()) - set(metadata.keys())

        for missing_metadata_name in missing_metadata:
            extractor = self.meta_extractors[missing_metadata_name]
            extracted_metadata_value = extractor(X, y)
            filled_metadata[missing_metadata_name] = extracted_metadata_value

        return filled_metadata

    def __remove_not_allowed_meta(self, metadata: Dict[str, float]) -> Dict[str, float]:
        allowed_qualities = set(self.meta_extractors.keys())
        meta_to_delete = set()
        for metadata_item in metadata.keys():
            if metadata_item not in allowed_qualities:
                meta_to_delete.add(metadata_item)

        for quality in meta_to_delete:
            metadata.pop(quality)

        return metadata
