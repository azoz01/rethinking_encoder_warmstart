import numpy as np
import optuna
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import torch
import typer
import warnings

from copy import deepcopy
from dataset2vec.train import LightningWrapper as D2vWrapper
from liltab.train.utils import LightningWrapper as LiltabWrapper
from functools import partial
from itertools import product
from loguru import logger
from optuna.samplers import TPESampler
from pathlib import Path
from pytorch_lightning import seed_everything
from scipy.linalg import LinAlgWarning
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch import Tensor
from tqdm import tqdm
from typing import Callable, Annotated

app = typer.Typer()

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Only needed for liltab experiment
training_datasets = [
    "alcohol_diagnosed-10",
    "hypotension_diagnosed-10",
    "diabetes_diagnosed-10",
    "anemia_diagnosed-10",
    "purpura_diagnosed-10",
    "respiratory_diagnosed-10",
    "atrial_diagnosed-10",
    "hypertensive_diagnosed-10",
    "overweight_diagnosed-10",
    "lipoid_diagnosed-10",
]
test_datasets = [
    "heart_diagnosed-10",
    "ischematic_diagnosed-10",
]

# Conversion of possible solver penalty pairs to mitigate
# issue with non-supported conditional categorical variables.
logistic_solver_penalty_pairs = [
    ("liblinear", "l1"),
    ("liblinear", "l2"),
    ("saga", "elasticnet"),
    ("saga", "l1"),
    ("saga", "l2"),
    ("saga", None),
    ("lbfgs", None),
    ("lbfgs", "l2"),
    ("newton-cg", None),
    ("newton-cg", "l2"),
    ("newton-cholesky", None),
    ("newton-cholesky", "l2"),
    ("sag", None),
    ("sag", "l2"),
]


def find_closest_by_mean(representation: Tensor, index: pd.Series, select_top_n: int):
    distances = index.apply(
        lambda x: ((representation.mean(axis=0) - (np.stack(x, axis=0)).mean(axis=0) ** 2)).sum()
    ).values
    top_n_idx = np.argsort(distances)[:select_top_n]
    return top_n_idx


def find_closest_by_cdist(representation: Tensor, index: pd.Series, select_top_n: int):
    distances = index.apply(
        lambda x: (
            torch.cdist(
                representation, torch.from_numpy(np.stack(x, axis=0)).type(torch.FloatTensor)
            )
        ).sum()
    ).values
    top_n_idx = np.argsort(distances)[:select_top_n]
    return top_n_idx


def find_closest_greedy(representation: Tensor, index: pd.Series, select_top_n: int):
    all_index = np.concatenate(list(map(lambda arr: np.stack(arr, axis=0), index.values)), axis=0)
    dists = torch.cdist(representation, Tensor(all_index))
    mins = (torch.argmin(dists, dim=1) // 27).tolist()
    mins = np.random.choice(mins, select_top_n).tolist()
    return mins


def find_closest_mixed(
    representation_liltab: Tensor,
    index_liltab: pd.Series,
    representation_d2v: Tensor,
    index_d2v: pd.Series,
    select_top_n: int,
):
    # d2v
    distances_d2v = index_d2v.apply(lambda x: ((representation_d2v - x) ** 2).sum()).values
    top_n_idx_d2v = np.argsort(distances_d2v)[:select_top_n].tolist()

    # liltab
    distances_liltab = index_liltab.apply(
        lambda x: (
            (representation_liltab.mean(axis=0) - (np.stack(x, axis=0)).mean(axis=0) ** 2)
        ).sum()
    ).values
    top_n_idx_liltab = np.argsort(distances_liltab)[:select_top_n].tolist()

    return np.random.choice(top_n_idx_d2v + top_n_idx_liltab, select_top_n).tolist()


@app.command(help="Generate search index for warmstart.")
def main(
    experiment: Annotated[
        str,
        typer.Option(..., help="Experiment. Either uci or mimic"),
    ],
    index_path: Annotated[
        Path,
        typer.Option(..., help="Path to indexed hparams."),
    ],
    ranks_path: Annotated[
        Path,
        typer.Option(..., help="Path to ranks of HP configurations."),
    ],
    datasets_path: Annotated[
        Path,
        typer.Option(..., help="Path to directories with datasets."),
    ],
    liltab_encoder_path: Annotated[
        Path,
        typer.Option(..., help="Path to liltab encoder."),
    ] = Path("models/liltab.ckpt"),
    d2v_encoder_path: Annotated[
        Path,
        typer.Option(..., help="Path to dataset2vec encoder."),
    ] = Path("models/d2v.ckpt"),
    optimisation_iterations: Annotated[
        int,
        typer.Option(..., help="Number of iterations per HPO optimisation."),
    ] = 20,
    warmstart_trials_count: Annotated[
        int,
        typer.Option(..., help="Number of trials with warmstart."),
    ] = 5,
    init_trials_count: Annotated[
        int,
        typer.Option(..., help="Number of initial trials."),
    ] = -1,
    search_strategy: Annotated[
        str,
        typer.Option(
            ...,
            help="Strategy of vector search of liltab warmstart. "
            "Can be mean, cdist_sum, greedy, mixed",
        ),
    ] = "mean",
    fix_seed: Annotated[
        bool,
        typer.Option(..., help="Output path of results."),
    ] = False,
    output_db_name: Annotated[
        str,
        typer.Option(..., help="Name of output database."),
    ] = "ho_base",
    output_path: Annotated[
        Path,
        typer.Option(..., help="Output path of results."),
    ] = Path("results/warmstart_dbs"),
):
    if fix_seed:
        seed_everything(123)
    output_path.mkdir(exist_ok=True)

    if init_trials_count == -1:
        init_trials_count = warmstart_trials_count

    logger.info("Loading encoders")
    liltab_encoder = LiltabWrapper.load_from_checkpoint(liltab_encoder_path).model
    d2v_encoder = D2vWrapper.load_from_checkpoint(d2v_encoder_path)
    best_params_index = pd.read_parquet(index_path)
    if experiment == "mimic":
        best_params_index = best_params_index.loc[best_params_index.task.isin(training_datasets)]
    hparams_ranks = pd.read_parquet(ranks_path).sort_values("rank")

    encoders = {
        "liltab": lambda X, y: liltab_encoder.encode_support_set(X, y),
        "d2v": d2v_encoder.encoder,
    }

    search_strategies = {
        "mean": find_closest_by_mean,
        "cdist_sum": find_closest_by_cdist,
        "greedy": find_closest_greedy,
        "mixed": find_closest_mixed,
    }

    def select_tasks_for_warmstart_liltab(
        X: pd.DataFrame,
        y: pd.DataFrame,
        index: pd.DataFrame,
        select_top_n: int,
        encoder_name: str = None,
        strategy: str = "mixed",
    ) -> pd.DataFrame:
        if y.nunique().values[0] > 2:
            oh = OneHotEncoder(sparse_output=False).set_output(transform="pandas")
            y = oh.fit_transform(y)
        X, y = Tensor(X.values), Tensor(y.values)
        if strategy == "mixed":
            encoder_liltab = encoders["liltab"]
            task_representation_liltab = encoder_liltab(X, y).detach()
            encoder_d2v = encoders["d2v"]
            task_representation_d2v = encoder_d2v(X, y).detach()
            top_n_idx = find_closest_mixed(
                task_representation_liltab,
                index["liltab_encoding"],
                task_representation_d2v,
                index["d2v_encoding"],
                select_top_n,
            )
        else:
            encoder = encoders[encoder_name]
            task_representation = encoder(X, y).detach()
            col = f"{encoder_name}_encoding"
            search = search_strategies[strategy]
            top_n_idx = search(task_representation, index[col], select_top_n)
        return deepcopy(index.iloc[top_n_idx])

    def select_tasks_for_warmstart_d2v(
        X: pd.DataFrame, y: pd.DataFrame, index: pd.DataFrame, select_top_n: int, encoder_name: str
    ) -> pd.DataFrame:
        if y.nunique().values[0] > 2:
            oh = OneHotEncoder(sparse_output=False).set_output(transform="pandas")
            y = oh.fit_transform(y)
        X, y = Tensor(X.values), Tensor(y.values)
        encoder = encoders[encoder_name]
        task_representation = encoder(X, y).detach().numpy()
        col = f"{encoder_name}_encoding"
        distances = index[col].apply(lambda x: ((task_representation - x) ** 2).sum()).values
        top_n_idx = np.argsort(distances)[:select_top_n]
        return deepcopy(index.iloc[top_n_idx])

    def select_task_for_warmstart_by_rank(select_top_n: int) -> pd.DataFrame:
        return hparams_ranks.iloc[:select_top_n]

    def optimize_logistic_regression(
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_test: pd.DataFrame,
        task_id: str,
        search_strategy: str,
        task_encoder: str = "liltab",
        warmstart_trials_count: int = 5,
        n_trials: int = 20,
        db_name: str = "ho_base",
        output_path: Path = Path("results/warmstart_dbs"),
    ) -> None:
        study = optuna.create_study(
            storage=f"sqlite:///{str(output_path) + '/' + db_name}.db",
            study_name=(
                f"{task_id}_logistic_{task_encoder}"
                f"{f'_warmstart_{warmstart_trials_count}' if warmstart_trials_count > 0 else ''}"
            ),
            direction="maximize",
            sampler=TPESampler(n_startup_trials=init_trials_count),
        )
        with torch.no_grad():
            if task_encoder == "d2v":
                warmstart = select_tasks_for_warmstart_d2v(
                    X_test, y_test, best_params_index, warmstart_trials_count, task_encoder
                )
            elif task_encoder == "rank":
                warmstart = select_task_for_warmstart_by_rank(warmstart_trials_count)
            elif task_encoder == "liltab":
                warmstart = select_tasks_for_warmstart_liltab(
                    X_test,
                    y_test,
                    best_params_index,
                    warmstart_trials_count,
                    task_encoder,
                    search_strategy,
                )
            else:
                raise ValueError(f"Invalid encoder: {task_encoder}")
        for _, row in warmstart.iterrows():
            warmstart_trial = row["logistic_best_hpo"]
            solver_penalty = (warmstart_trial["solver"], warmstart_trial["penalty"])
            solver_penalty_idx = logistic_solver_penalty_pairs.index(solver_penalty)
            warmstart_trial = deepcopy(warmstart_trial)
            warmstart_trial["solver_penalty_idx"] = solver_penalty_idx
            del warmstart_trial["solver"]
            del warmstart_trial["penalty"]
            study.enqueue_trial(warmstart_trial)
        study.optimize(
            partial(
                objective_logistic_regression,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                metric=roc_auc_score,
            ),
            n_trials=n_trials,
        )

    def objective_logistic_regression(
        trial: optuna.Trial,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_test: pd.DataFrame,
        metric: Callable,
    ) -> float:
        solver_penalty = logistic_solver_penalty_pairs[
            trial.suggest_int("solver_penalty_idx", 0, len(logistic_solver_penalty_pairs) - 1)
        ]
        params = {
            "tol": trial.suggest_float("tol", 1e-4, 1e-3, log=True),
            "C": trial.suggest_float("C", 1e-4, 1e4, log=True),
            "solver": solver_penalty[0],
            "penalty": solver_penalty[1],
            "max_iter": 500,
        }
        if params["solver"] == "liblinear":
            params["intercept_scaling"] = trial.suggest_float("intercept_scaling", 1e-3, 1)
            if params["penalty"] == "l2":
                params["dual"] = trial.suggest_categorical("dual", [True, False])
        elif params["solver"] == "saga":
            params["l1_ratio"] = trial.suggest_float("l1_ratio", 0, 1)

        model = LogisticRegression(**params)
        model.fit(X_train, y_train.values.reshape(-1))
        test_proba = model.predict_proba(X_test)
        if test_proba.shape[1] == 2:
            test_proba = test_proba[:, 1]
            return metric(y_test, test_proba)
        else:
            return metric(y_test.values.ravel(), test_proba, multi_class="ovo")

    optimizations = [optimize_logistic_regression]
    kwargs_list = [
        {"task_encoder": "d2v", "warmstart_trials_count": 0},  # baseline
        {"task_encoder": "d2v", "warmstart_trials_count": warmstart_trials_count},  # d2v eval
        {"task_encoder": "rank", "warmstart_trials_count": warmstart_trials_count},  # rank eval
        {"task_encoder": "liltab", "warmstart_trials_count": warmstart_trials_count},  # liltab eval
    ]

    def calculate_experiment_for_task(task_path: Path, task_id: str) -> None:
        train_data = pd.read_csv(task_path / "train.csv")
        test_data = pd.read_csv(task_path / "test.csv")
        X_train, y_train = train_data.iloc[:, :-1], train_data.iloc[:, [-1]]
        X_test, y_test = test_data.iloc[:, :-1], test_data.iloc[:, [-1]]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings("ignore", category=LinAlgWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            for optimization, kwargs in product(optimizations, kwargs_list):
                optimization(
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    n_trials=optimisation_iterations,
                    task_id=task_id,
                    db_name=output_db_name,
                    search_strategy=search_strategy,
                    output_path=output_path,
                    **kwargs,
                )

    if experiment == "uci":
        paths = list(datasets_path.iterdir())
        it = tqdm(paths)
        for task_path in it:
            it.set_description(str(task_path))
            calculate_experiment_for_task(task_path, task_path.stem)
    else:
        for task in test_datasets:
            dataset_path = datasets_path / task
            dataset_name = dataset_path.name.split("-")[0]
            tasks = list(dataset_path.iterdir())[:300]
            for id_path in tqdm(tasks):
                task_id = f"{dataset_name}-{id_path.name.split('-')[1]}"
                calculate_experiment_for_task(id_path, task_id)


if __name__ == "__main__":
    app()
