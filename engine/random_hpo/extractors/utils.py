import numpy as np

from typing import Dict, Callable


def NumberOfFeatures(X, y=None) -> float:
    return X.shape[1]


def NumberOfInstances(X, y=None) -> float:
    return X.shape[0]


def NumberOfNumericFeatures(X, y=None) -> float:
    return X.select_dtypes(include=np.number).shape[1]


meta_extractors: Dict[str, Callable[..., float]] = {
    "NumberOfFeatures": NumberOfFeatures,
    "NumberOfInstances": NumberOfInstances,
    "NumberOfNumericFeatures": NumberOfNumericFeatures,
}
