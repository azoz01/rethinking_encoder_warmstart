import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def get_generic_preprocessing() -> Pipeline:
    cat_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one-hot", OneHotEncoder(sparse=False, handle_unknown="ignore")),
        ]
    )

    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )

    pipeline = Pipeline(
        [
            (
                "transformers",
                make_column_transformer(
                    (cat_pipeline, make_column_selector(dtype_include=object)),
                    (num_pipeline, make_column_selector(dtype_include=np.number)),
                ),
            )
        ]
    )

    return pipeline


def wrap_model_with_preprocessing(
    model: any, preprocessing: Pipeline = get_generic_preprocessing()
) -> Pipeline:
    pipeline = Pipeline([("preprocessing", preprocessing), ("model", model)])
    return pipeline
