import shutil
import typer
import pickle as pkl
import warnings

from loguru import logger
from pathlib import Path
from pytorch_lightning import seed_everything
from sklearn.metrics import roc_auc_score as metric
from typing_extensions import Annotated

from engine.random_hpo.utils import (
    get_datasets,
    get_logistic_regression_grid,
    get_predefined_logistic_regression,
    get_predefined_xgboost_classifier,
    get_xgboost_grid,
    put_results,
)
from engine.random_hpo.searchers.hpo_searchers import RandomSearch

app = typer.Typer()


@app.command(help="Generate hyperparameter base.")
def main(
    input_data_path: Annotated[Path, typer.Option(..., help="Path to input tasks")] = Path(
        "data/uci/splitted"
    ),
    number_of_random_combinations: Annotated[
        int,
        typer.Option(
            ...,
            help="Number of combinations in random search.",
        ),
    ] = 100,
    output_results_path: Annotated[Path, typer.Option(..., help="Path to input tasks")] = Path(
        "results/hpo_uci"
    ),
) -> None:
    seed_everything(123)
    if output_results_path.exists():
        shutil.rmtree(output_results_path)
    output_results_path.mkdir()
    logger.info("Loading datasets")
    dataloaders = get_datasets(input_data_path)

    logger.info("Start searching")
    progress_bar = ((key, data_tuple) for key, data_tuple in dataloaders.items())

    logger.info(f"Number of tasks: {len(dataloaders.keys())}")

    for dir_name, data_tuple in progress_bar:
        try:
            if "test" in str(dir_name):
                logger.warning(f"skipping {dir_name}")
                continue
            train_X, train_y, test_X, test_y = (
                data_tuple[0].iloc[:, :-1],
                data_tuple[0].iloc[:, -1],
                data_tuple[1].iloc[:, :-1],
                data_tuple[1].iloc[:, -1],
            )

            # Logistic regression
            logistic_grid = get_logistic_regression_grid()
            logistic_model = get_predefined_logistic_regression()
            logistic_searcher = RandomSearch(logistic_model, logistic_grid)

            logger.info(f"Start training {dir_name}")

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                logistic_searcher.search_holdout(
                    train_X, train_y, test_X, test_y, metric, n_iter=number_of_random_combinations
                )

            put_results(
                output_results_path / "logistic_scores_base.pkl",
                dir_name,
                str(logistic_model.__class__),
                logistic_searcher.search_results["score"],
            )

            # XGboost
            xgboost_grid = get_xgboost_grid()
            xgboost_model = get_predefined_xgboost_classifier()
            xgboost_searcher = RandomSearch(xgboost_model, xgboost_grid)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                xgboost_searcher.search_holdout(
                    train_X, train_y, test_X, test_y, metric, n_iter=number_of_random_combinations
                )

            put_results(
                output_results_path / "xgboost_scores_base.pkl",
                dir_name,
                str(xgboost_model.__class__),
                xgboost_searcher.search_results["score"],
            )

        except Exception as e:
            logger.error(e)
            logger.error(f"{dir_name} FAILED")

    with open(output_results_path / "logistic_parameters_base.pkl", "wb") as f:
        pkl.dump(logistic_searcher.search_results["hpo"], f)

    with open(output_results_path / "xgboost_parameters_base.pkl", "wb") as f:
        pkl.dump(xgboost_searcher.search_results["hpo"], f)


if __name__ == "__main__":
    app()
