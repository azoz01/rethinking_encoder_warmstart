import pickle as pkl
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from .searchers.search_grid import ConditionalGrid, CubeGrid


def get_logistic_regression_grid(init_seed: int = None) -> ConditionalGrid:
    grid_base = CubeGrid()
    grid_base.add(
        "tol", values=[0.0001, 0.001], space="real", distribution="loguniform"
    )
    grid_base.add("C", values=[0.0001, 10000], space="real", distribution="loguniform")
    grid_base.add(
        "solver",
        values=["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"],
        space="cat",
    )

    grid_liblinear = CubeGrid()
    grid_liblinear.add("intercept_scaling", values=[0.001, 1], space="real")
    grid_liblinear.add("penalty", values=["l1", "l2"], space="cat")

    grid_liblinear_ext = CubeGrid()
    grid_liblinear_ext.add("dual", values=[True, False], space="cat")

    grid_saga = CubeGrid()
    grid_saga.add("penalty", values=["elasticnet", "l1", "l2", None], space="cat")
    grid_saga.add("l1_ratio", values=[0, 1], space="real")

    grid_others = CubeGrid()
    grid_others.add("penalty", values=["l2", None], space="cat")

    cond_grid = ConditionalGrid(init_seed=init_seed)
    cond_grid.add_cube(grid_base)
    cond_grid.add_cube(grid_liblinear, lambda hpo: hpo["solver"] == "liblinear")
    cond_grid.add_cube(
        grid_liblinear_ext,
        lambda hpo: hpo["solver"] == "liblinear" and hpo["penalty"] == "l2",
    )
    cond_grid.add_cube(grid_saga, lambda hpo: hpo["solver"] == "saga")
    cond_grid.add_cube(
        grid_others, lambda hpo: hpo["solver"] not in ("liblinear", "saga")
    )

    cond_grid.reset_seed(123)

    return cond_grid


def get_predefined_logistic_regression() -> LogisticRegression:
    model = LogisticRegression(random_state=123, max_iter=500)
    return model


def get_xgboost_grid() -> ConditionalGrid:
    base_grid = CubeGrid()
    base_grid.add("n_estimators", values=[1, 1000], space="int")
    base_grid.add("learning_rate", values=[0, 1], space="real", distribution="uniform")
    base_grid.add("booster", values=["gblinear", "gbtree"], space="cat")
    base_grid.add("subsample", values=[0.5, 1], space="real")
    base_grid.add("max_depth", values=[6, 15], space="int")
    base_grid.add(
        "min_child_weight", values=[2, 256], space="real", distribution="uniform"
    )
    base_grid.add("colsample_bytree", values=[0.2, 1], space="real")
    base_grid.add("colsample_bylevel", values=[0.2, 1], space="real")

    cond_grid = ConditionalGrid()
    cond_grid.add_cube(base_grid)
    cond_grid.reset_seed(123)

    return cond_grid


def get_predefined_xgboost_classifier() -> XGBClassifier:
    model = XGBClassifier(random_state=123, n_jobs=-1)
    return model


def get_datasets(path: Path) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    root_path = Path(path)

    directories = root_path.glob("*")
    data_tuples = {}

    for dataset_path in directories:
        df_train = pd.read_csv(dataset_path / "train.csv")
        df_test = pd.read_csv(dataset_path / "test.csv")
        data_tuples[str(dataset_path)] = (df_train, df_test)

    return data_tuples


def put_results(path: Path, name: str, model: str, data: Dict[str, any]) -> None:
    path = Path(path)
    if path.is_file():
        with open(path, "rb") as f:
            obj = pkl.load(f)
            obj[model][name] = data
        with open(path, "wb") as f:
            pkl.dump(obj, f)

    else:
        with open(path, "wb") as f:
            pkl.dump({model: {name: data}}, f)
