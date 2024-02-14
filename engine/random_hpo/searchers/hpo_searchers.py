import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .early_stopping import DummyEarlyStopping, GenericEarlyStopping
from .search_grid import RandomGrid
from .search_results import _SearchResults


class GenericHPOSearch(ABC):
    def __init__(
        self,
        model: any,
        random_grid: RandomGrid,
        early_stopping: Optional[GenericEarlyStopping] = None,
    ) -> None:
        self.model = model
        self.random_grid = random_grid
        self.early_stopping = early_stopping or DummyEarlyStopping()
        self._search_results = _SearchResults()

    @property
    def search_results(self):
        return self._search_results.get_results()

    @abstractmethod
    def search(
        X: pd.DataFrame,
        y: pd.DataFrame,
        scoring: Callable[..., float],
        n_iter: int = 100,
        cv: int = 5,
        preprocessor_X: Callable[..., np.ndarray] = None,
        preprocessor_y: Callable[..., np.ndarray] = None,
        encode_y: bool = False,
    ) -> Dict[str, any]:
        ...

    def _get_cv_indexes(self, size: int, cv: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        all_indexes = np.arange(size)
        fold_size = size // cv
        folds = []

        remain_indexes = np.copy(all_indexes)

        for _ in range(cv):
            fold = np.random.choice(remain_indexes, fold_size, replace=False)
            remain_indexes = np.setdiff1d(remain_indexes, fold)
            folds.append((fold, np.delete(all_indexes, fold)))

        return folds

    def _encode_y(self, y: pd.Series) -> pd.DataFrame:
        y = y.astype("category")
        y = pd.get_dummies(y, drop_first=True)

        return y

    def _override_model_hpo(self, hpo: Dict[str, any]) -> None:
        model_ = deepcopy(self.model)
        for key, value in hpo.items():
            setattr(model_, key, value)

        return model_


class RandomSearch(GenericHPOSearch):
    """
    Implementation of random search. Contains informaiton about searching process
    in dictionary `search results`. Can be run multiple times without removing
    information about previous runs.
    """

    def __init__(
        self,
        model: any,
        random_grid: RandomGrid,
        early_stopping: Optional[GenericEarlyStopping] = None,
    ) -> None:
        super().__init__(model, random_grid, early_stopping)

    def get_best_hpo(self, min_best: bool):
        res = self._search_results.get_results()
        best_idx = np.argmax(res["score"] * (-1 if min_best else 1))
        return res["hpo"][best_idx]

    def search_holdout(
        self,
        train_X: pd.DataFrame,
        train_y: pd.DataFrame,
        test_X: pd.DataFrame,
        test_y: pd.DataFrame,
        scoring: Callable[..., float],
        n_iter: int = 100,
    ) -> None:
        for i in range(n_iter):
            hpo = self.random_grid.pick()

            model = self._override_model_hpo(hpo)
            model.fit(train_X, train_y)
            pred_y = model.predict_proba(test_X)

            if train_y.unique().shape[0] > 2:
                score = scoring(test_y, pred_y, multi_class="ovo")
            else:
                score = scoring(test_y, pred_y[:, 1])

            self._search_results.add("score", score)
            self._search_results.add("hpo", hpo)

            is_stop = self.early_stopping.is_stop(self.search_results)
            if is_stop:
                warnings.warn(f"Searching ended after {i+1} iteration due to early stopping.")
                break

    def search(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        scoring: Callable[..., float],
        n_iter: int = 100,
        cv: int = 5,
        encode_y: bool = False,
    ) -> None:
        """
        Evaluate model on provided data with randomly
        selected hyperparameters.

        Args:
            X (pd.DataFrame): dataframe with features.
            y (pd.DataFrame): dataframe with target.
            scoring (Callable[..., float]): scoring function. Should take
                array of y_hat, y as input and return single value.
            n_iter (int, optional): Number of iteration. Defaults to 100.
            cv (int, optional): Number of cross validaiton folds. Defaults to 5.
            preprocessor_X (Callable[..., np.ndarray], optional): Preprocessing
                object. Sholud have fit(), transform() and fit_transform()
                functions. Defaults to None.
            preprocessor_y (Callable[..., np.ndarray], optional): preprocessing
                object. Sholud have fit(), transform() and fit_transform()
                functions. Defaults to None. Defaults to None.
            encode_y (bool, optional): if set as true, y will be hot-one
                encoded before all operations. Defaults to False.
        """
        if encode_y:
            y = super()._encode_y(y)
        for i in range(n_iter):
            hpo = self.random_grid.pick()
            folds = super()._get_cv_indexes(X.shape[0], cv)
            scores = []
            for fold in folds:
                train_X, test_X = X.iloc[fold[0], :], X.iloc[fold[1], :]
                train_y, test_y = y.iloc[fold[0], :], y.iloc[fold[1], :]

                model = self._override_model_hpo(hpo)
                model.fit(train_X, train_y)
                pred_y = model.predict(test_X)

                score = scoring(test_y, pred_y)
                scores.append(score)

            self._search_results.add("scores", scores)
            self._search_results.add("hpo", hpo)
            self._search_results.add("mean_score", np.mean(scores))
            self._search_results.add("std_score", np.std(scores))

            is_stop = self.early_stopping.is_stop(self.search_results)
            if is_stop:
                warnings.warn(f"Searching ended after {i+1} iteration due to early stopping.")
                break
