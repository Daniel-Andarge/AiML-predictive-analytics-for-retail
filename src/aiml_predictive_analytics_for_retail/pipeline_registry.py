"""Project pipelines."""

from typing import Dict

from kedro.pipeline import Pipeline, node
from .pipelines.data_processing.pipeline import create_pipeline
import logging
from .pipelines.data_processing.nodes import (
    impute_missing_values,
    remove_duplicate_rows,
    transform_date_features,
    scale_features,
    encode_categorical_features,
    create_data_pipeline,
)

data_engineering_pipeline = Pipeline(
    [
        node(
            func=impute_missing_values,
            inputs="01_raw",
            outputs="imputed_data",
            name="impute_missing_values",
        ),
        node(
            func=remove_duplicate_rows,
            inputs="imputed_data",
            outputs="deduped_data",
            name="remove_duplicate_rows",
        ),
        node(
            func=transform_date_features,
            inputs="deduped_data",
            outputs="transformed_data",
            name="transform_date_features",
        ),
        node(
            func=scale_features,
            inputs="transformed_data",
            outputs="scaled_data",
            name="scale_features",
        ),
        node(
            func=encode_categorical_features,
            inputs="scaled_data",
            outputs="encoded_data",
            name="encode_categorical_features",
        ),
        node(
            func=create_data_pipeline,
            inputs="raw_data",
            outputs="preprocessed_data",
            name="create_data_pipeline",
        ),
    ]
)
def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.
    Returns:
        A mapping from pipeline names to `Pipeline` objects.
    """
    return {
        "__default__": data_engineering_pipeline,
        "data_engineering": data_engineering_pipeline,
    }
