import shutil
import typer
import pickle as pkl
import warnings

from loguru import logger
from pathlib import Path
from pytorch_lightning import seed_everything
from sklearn.metrics import roc_auc_score as metric
from tqdm import tqdm
from typing_extensions import Annotated

from engine.random_hpo.utils import (
    get_datasets,
    get_logistic_regression_grid,
    get_predefined_logistic_regression,
    put_results,
)
from engine.random_hpo.searchers.hpo_searchers import RandomSearch

app = typer.Typer()


@app.command(
    help="""create mini holdout tasks for regular models (e.g. linear regression).
    Each task is a directory with two files: train and test"""
)
def main(
    input_data_path: Annotated[Path, typer.Option(..., help="Path to input tasks")] = Path(
        "data/mimic/mini_holdout"
    ),
    n_tasks_per_dataset: Annotated[
        int,
        typer.Option(
            ...,
            "Number of tasks per dataset.",
        ),
    ] = 10,
    number_of_random_combinations: Annotated[
        int,
        typer.Option(
            ...,
            "Number of combinations in random search.",
        ),
    ] = 200,
    output_results_path: Annotated[Path, typer.Option(..., help="Path to input tasks")] = Path(
        "results/logistic_hpo"
    ),
) -> None:
    seed_everything(123)
    if output_results_path.exists():
        shutil.rmtree(output_results_path)
    output_results_path.mkdir()
    logger.info("Loading datasets")
    dataloaders = get_datasets(input_data_path)

    logger.info("Start searching")
    progress_bar = tqdm(
        [(key, data_tuple) for key, data_tuple in dataloaders.items()],
    )

    logger.info(f"Number of tasks: {len(dataloaders.keys())}")

    for dir_name, data_tuple in progress_bar:
        if int(dir_name[-5:]) >= n_tasks_per_dataset:
            continue

        grid = get_logistic_regression_grid()
        model = get_predefined_logistic_regression()
        searcher = RandomSearch(model, grid)

        train_X, train_y, test_X, test_y = (
            data_tuple[0].iloc[:, :-1],
            data_tuple[0].iloc[:, -1],
            data_tuple[1].iloc[:, :-1],
            data_tuple[1].iloc[:, -1],
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            searcher.search_holdout(
                train_X, train_y, test_X, test_y, metric, n_iter=number_of_random_combinations
            )

        put_results(
            output_results_path / "logistic_scores_base.pkl",
            dir_name,
            str(model.__class__),
            searcher.search_results["score"],
        )

    with open(output_results_path / "logistic_parameters_base.pkl", "wb") as f:
        pkl.dump(searcher.search_results["hpo"], f)


if __name__ == "__main__":
    main()
