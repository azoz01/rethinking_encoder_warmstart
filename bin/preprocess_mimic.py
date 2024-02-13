import json
import numpy as np
import pandas as pd
import re
import typer
import yaml

from collections import Counter
from functools import reduce
from operator import add
from loguru import logger
from pathlib import Path
from pytorch_lightning import seed_everything
from typing_extensions import Annotated


app = typer.Typer()


@app.command(help="Select most important features for all target variables.")
def main(
    n_features: Annotated[
        int, typer.Option(..., help="Number of most important features to select.")
    ] = 10,
    most_important_features_path: Annotated[
        Path,
        typer.Option(
            ...,
            help="Path to specification of most important features for each target variable",
        ),
    ] = Path("config/feature_selection.yaml"),
    input_data_path: Annotated[
        Path,
        typer.Option(..., help="Path to raw data path with combined target variables."),
    ] = Path("data/mimic/raw/metaMIMIC.csv"),
    output_data_path: Annotated[
        Path, typer.Option(..., help="Path where output data will be written.")
    ] = Path("data/mimic/stg/most_important"),
) -> None:
    output_data_path.mkdir(parents=True, exist_ok=True)
    seed_everything(123)

    logger.info("Loading data with configuration.")
    with open(most_important_features_path) as f:
        features = yaml.load(f, Loader=yaml.CLoader)

    df_combined = pd.read_csv(input_data_path)
    df_combined.drop(columns=["subject_id"], inplace=True)

    logger.info("Selecting target columns.")
    regexp = re.compile(".*diagnosed")
    is_target_col = np.array(
        [True if regexp.match(col_name) else False for col_name in df_combined.columns]
    )
    target_columns = df_combined.columns[is_target_col]

    df_combined = df_combined.convert_dtypes(convert_boolean=False)

    logger.info("Selecting most important features and writing output data.")
    for target_column in target_columns:
        df_temp = df_combined.loc[:, features[target_column][:n_features]]
        df_temp["target"] = df_combined[target_column]

        file_path = output_data_path / f"{target_column}-{n_features}.csv"
        df_temp.to_csv(file_path, index=False)

    features_all = []
    for feature in features.values():
        features_all += feature
    features_agg = dict(Counter(features_all))

    cnt = dict()
    for diagnose_name in features.keys():
        diagnose_features = list(features[diagnose_name][:n_features])
        diagnose_cnt = {}
        for key, value in features_agg.items():
            if key in diagnose_features:
                if value - 1 != 0:
                    diagnose_cnt[key] = value - 1

        cnt[diagnose_name] = reduce(add, diagnose_cnt.values())

    logger.info(f"Selected features: {json.dumps(cnt, indent=4)}")


if __name__ == "__main__":
    app()
