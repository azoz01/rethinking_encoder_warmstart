import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from engine.cd_plot import draw_cd_diagram

models = ["logistic", "xgboost"]


def suffix_func(name):
    if "liltab" in name:
        return "liltab"
    if "d2v_warmstart" in name:
        return "d2v"
    if "rank" in name:
        return "rank"
    return "no-warmstart"


def model_name_func(name):
    for model in models:
        if model in name:
            return model


name_re = re.compile("(.*-[0-9]+)_")


def generate_plot_df(study_path):
    res = optuna.get_all_study_summaries(f"sqlite:///{study_path}")
    study_full_names = list(map(lambda study: study.study_name, res))
    study_methods = list(map(lambda name: suffix_func(name), study_full_names))
    study_models = list(map(lambda name: model_name_func(name), study_full_names))
    study_names = list(
        map(lambda m: re.split(r"_logistic|_xgboost", m)[0], study_full_names)
    )

    p_bar = tqdm(
        (zip(study_names, study_methods, study_models, study_full_names)),
        total=len(study_names),
    )

    df = pd.DataFrame()

    for name, method, model, full_name in p_bar:
        study = optuna.load_study(
            study_name=full_name, storage=f"sqlite:///{study_path}"
        )

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
    return df


paths = list(sorted(Path("results/warmstart_dbs/uci").iterdir()))
plot_dfs = list(map(generate_plot_df, tqdm(paths)))
for i, d in enumerate(plot_dfs):
    d["fold"] = i
df = pd.concat(plot_dfs, axis=0)
df.to_csv("results/warmstart_plots/uci_df.csv", index=False)


# Count of tasks when method is best in i-th iteration
df = pd.read_csv("results/warmstart_plots/uci_df.csv")
df_grouped = (
    df.sort_values("score")
    .groupby(["fold", "name", "model", "no"])
    .last()
    .reset_index()
)
results = df_grouped.groupby(["fold", "no", "model", "method"]).count().reset_index()
models = ["logistic", "xgboost"]
fold_ids = list(range(5))

fig, ax = plt.subplots(ncols=5, nrows=2, figsize=(25, 9))

for i, model in enumerate(models):
    for j, fold in enumerate(fold_ids):
        p = sns.lineplot(
            results[
                (results["model"] == model) & (results["fold"] == fold)
            ].sort_values("method"),
            x="no",
            y="score",
            hue="method",
            ax=ax[i][j],
        )
        p.set_ylabel("Number of tasks when method is best")
        p.set_xlabel("Iteration")
        p.set_title(f"model - {model} fold_id - {fold}")
        if i == 0 and j == len(fold_ids) - 1:
            p.legend(loc=(1.05, 0.75))
            plt.setp(p.get_legend().get_title(), fontsize=7)
            plt.setp(p.get_legend().get_texts(), fontsize=7)
        else:
            p.legend([], [], frameon=False)
plt.savefig("results/warmstart_plots/uci_best_counts.png")
plt.clf()

# ADTM
scores_min_max = (
    df.groupby(["fold", "name", "model"], as_index=False)
    .agg({"score": ["min", "max"]})
    .reset_index(drop=False)
)
scores_min_max.columns = scores_min_max.columns.get_level_values(0)
scores_min_max.columns = scores_min_max.columns[:-2].tolist() + ["min", "max"]

df_with_min_max = df.merge(scores_min_max, how="left", on=["fold", "name", "model"])
df_with_min_max["score"] = (df_with_min_max["score"] - df_with_min_max["min"]) / (
    df_with_min_max["max"] - df_with_min_max["min"]
)
df_with_min_max["max_minus_score"] = 1 - df_with_min_max["score"]
df_with_min_max = df_with_min_max[
    ["fold", "name", "model", "method", "no", "max_minus_score"]
]
df_with_min_max.loc[df_with_min_max.max_minus_score == 0].groupby(
    ["model", "method"]
).agg({"name": lambda x: x.nunique()})
models = ["logistic"]
fold_ids = list(range(5))

fig, ax = plt.subplots(ncols=5, nrows=1, figsize=(25, 4.5), sharey=True)

for i, model in enumerate(models):
    for j, fold in enumerate(fold_ids):
        p = sns.lineplot(
            df_with_min_max[
                (df_with_min_max["model"] == model) & (df_with_min_max["fold"] == fold)
            ].sort_values("method"),
            x="no",
            y="max_minus_score",
            hue="method",
            drawstyle="steps",
            ax=ax[j],
        )
        p.set_ylabel("Distance to maximum", fontsize=22)
        p.set_xlabel("Iteration", fontsize=22)
        p.tick_params(labelsize=18)
        p.tick_params(labelsize=18)
        p.set_title(f"Model: {model}, fold_id: {fold}", fontsize=22)
        plt.tight_layout(h_pad=5, w_pad=1)
        if i == 0 and j == len(fold_ids) - 1:
            p.legend(loc=(-2.25, 1.15), ncol=4, frameon=False)
            plt.setp(p.get_legend().get_title(), fontsize=22)
            plt.setp(p.get_legend().get_texts(), fontsize=22)
        else:
            p.legend([], [], frameon=False)
plt.savefig("results/warmstart_plots/uci_adtm_logistic.png", bbox_inches="tight")
plt.clf()

models = ["xgboost"]
fold_ids = list(range(5))

fig, ax = plt.subplots(ncols=5, nrows=1, figsize=(25, 4.5), sharey=True)

for i, model in enumerate(models):
    for j, fold in enumerate(fold_ids):
        p = sns.lineplot(
            df_with_min_max[
                (df_with_min_max["model"] == model) & (df_with_min_max["fold"] == fold)
            ].sort_values("method"),
            x="no",
            y="max_minus_score",
            hue="method",
            drawstyle="steps",
            ax=ax[j],
        )
        p.set_ylabel("Distance to maximum", fontsize=22)
        p.set_xlabel("Iteration", fontsize=22)
        p.tick_params(labelsize=18)
        p.tick_params(labelsize=18)
        p.set_title(f"Model: {model}, fold_id: {fold}", fontsize=22)
        plt.tight_layout(h_pad=5, w_pad=1)
        if i == 0 and j == len(fold_ids) - 1:
            p.legend(loc=(-2.25, 1.15), ncol=4, frameon=False)
            plt.setp(p.get_legend().get_title(), fontsize=22)
            plt.setp(p.get_legend().get_texts(), fontsize=22)
        else:
            p.legend([], [], frameon=False)
plt.savefig("results/warmstart_plots/uci_adtm_xgboost.png", bbox_inches="tight")
plt.clf()

models = ["logistic", "xgboost"]
iterations = [9, 29]
for model in models:
    for it in iterations:
        df_to_cd = (
            df_with_min_max.loc[
                (df_with_min_max.no == it) & (df_with_min_max.model == model)
            ]
            .rename(
                columns={
                    "method": "classifier_name",
                    "name": "dataset_name",
                    "max_minus_score": "accuracy",
                }
            )[["classifier_name", "dataset_name", "accuracy"]]
            .reset_index(drop=True)
        )
        df_to_cd["accuracy"] = 1 - df_to_cd["accuracy"]
        draw_cd_diagram(
            f"results/warmstart_plots/uci_{model}_{it + 1}.png",
            df_perf=df_to_cd,
            title=f"UCI - {model} - it. {it + 1}",
            labels=False,
        )
