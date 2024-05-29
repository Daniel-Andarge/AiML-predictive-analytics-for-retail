"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.6
"""

from kedro.pipeline import Pipeline, pipeline
from kedro.pipeline import Pipeline, node
from .nodes import (
    impute_missing_values,
    remove_duplicate_rows,
    transform_date_features,
    scale_features,
    encode_categorical_features,
    create_data_pipeline,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=impute_missing_values,
            inputs="raw_data",
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
    ])
