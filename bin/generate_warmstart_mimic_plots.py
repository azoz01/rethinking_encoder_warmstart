import optuna
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer

from matplotlib import rcParams
from loguru import logger
from pathlib import Path
from tqdm import tqdm
from typing_extensions import Annotated

app = typer.Typer()

models = ["logistic"]


@app.command(help="Generate hyperparameter base.")
def main(
    database_path: Annotated[Path, typer.Option(..., help="Path to database to analyze")] = Path(
        "results/warmstart_dbs/mimic.db"
    ),
    output_plots_path: Annotated[Path, typer.Option(..., help="Path to output plots")] = Path(
        "results/warmstart_plots"
    ),
) -> None:
    output_plots_path.mkdir(exist_ok=True)

    logger.info("Processing studies names")
    res = optuna.get_all_study_summaries("sqlite:///" + str(database_path))

    def suffix_func(name):
        if "liltab" in name:
            return "liltab"
        if "d2v_warmstart" in name:
            return "d2v"
        if "rank" in name:
            return "rank"
        return "baseline"

    def model_name_func(name):
        for model in models:
            if model in name:
                return model

    name_re = re.compile("(.*-[0-9]+)_")
    study_full_names = list(map(lambda study: study.study_name, res))
    study_methods = list(map(lambda name: suffix_func(name), study_full_names))
    study_models = list(map(lambda name: model_name_func(name), study_full_names))
    study_names = list(map(lambda name: name_re.search(name)[0][:-1], study_full_names))

    logger.info("Processing studies")
    p_bar = tqdm(
        (zip(study_names, study_methods, study_models, study_full_names)), total=len(study_names)
    )

    df = pd.DataFrame()

    for name, method, model, full_name in p_bar:
        study = optuna.load_study(study_name=full_name, storage="sqlite:///" + str(database_path))

        study_scores = list(map(lambda trail: trail.value, study.trials))
        study_scores_best = list(np.maximum.accumulate(study_scores))
        study_number = list(map(lambda trail: trail.number, study.trials))
        n = len(study_number)

        new_record = pd.DataFrame(
            {
                "score": study_scores_best,
                "no": study_number,
                "method": [method for _ in range(n)],
                "model": [model for _ in range(n)],
                "name": [name for _ in range(n)],
            }
        )
        df = pd.concat([df, new_record])

    logger.info("Generating ADTM plots")
    scores_min_max = (
        df.groupby(["name", "model"], as_index=False)
        .agg({"score": ["min", "max"]})
        .reset_index(drop=False)
    )
    scores_min_max.columns = scores_min_max.columns.get_level_values(0)
    scores_min_max.columns = scores_min_max.columns[:-2].tolist() + ["min", "max"]
    df_with_min_max = df.merge(scores_min_max, "left", ["name", "model"])
    df_with_min_max["score"] = (df_with_min_max["score"] - df_with_min_max["min"]) / (
        df_with_min_max["max"] - df_with_min_max["min"]
    )
    df_with_min_max["max_minus_score"] = 1 - df_with_min_max["score"]
    rcParams["figure.figsize"] = (16, 9)
    sns.lineplot(
        df_with_min_max.loc[df_with_min_max.model == "logistic"],
        x="no",
        y="max_minus_score",
        hue="method",
    ).set_title("ADTM")
    plt.savefig(output_plots_path / "adtm_mimic_logistic.png")
    plt.clf()

    logger.info("Generating count of tasks when method is best in i-th iteration")
    df_grouped = df.sort_values("score").groupby(["name", "model", "no"]).last().reset_index()
    results = df_grouped.groupby(["no", "model", "method"]).count().reset_index()
    results["model"].unique()
    sns.lineplot(
        results[results["model"] == "logistic"], x="no", y="score", hue="method"
    ).set_title("Count of tasks when method is best in i-th iteration")
    plt.savefig(output_plots_path / "best_tasks_mimic_logistic.png")
    plt.clf()

    logger.info("Generating average rank in i-th iteration")
    ranks = df.groupby(["name", "model", "no"])["score"].rank("average", ascending=False)
    df_copy = df.copy()
    df_copy["rank"] = ranks

    rank_df = df_copy.groupby(["no", "model", "method"]).mean("rank").reset_index()
    sns.lineplot(rank_df[rank_df["model"] == "logistic"], x="no", y="rank", hue="method").set_title(
        "Average rank in i-th iteration"
    )
    plt.savefig(output_plots_path / "avg_rank_mimic_logistic.png")
    plt.clf()


if __name__ == "__main__":
    app()
