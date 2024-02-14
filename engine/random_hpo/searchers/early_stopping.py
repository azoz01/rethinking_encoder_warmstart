import numpy as np

from abc import ABC, abstractmethod
from typing import Dict, List


class GenericEarlyStopping(ABC):
    @abstractmethod
    def is_stop(self, values: Dict[str, List]) -> bool:
        ...


class DummyEarlyStopping(GenericEarlyStopping):
    def is_stop(self, values: Dict[str, List]) -> bool:
        return False


class NoImprovementEarlyStopping(GenericEarlyStopping):
    def __init__(self, n_iteration: int = 100, lowest_best: bool = True) -> None:
        super().__init__()
        self.n_iteration = n_iteration
        self.lowest_best = lowest_best

    def is_stop(self, values: Dict[str, List]) -> bool:
        scores = values["mean_score"]

        if len(scores) < self.n_iteration:
            return False

        best_score = np.max(scores)
        best_from_last_n = np.max(scores[-1 : -self.n_iteration : -1])

        is_improvement = best_score == best_from_last_n

        return not is_improvement
