"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.6
"""

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import FeatureUnion
from sklearn.compose import  ColumnTransformer
from sklearn.pipeline import Pipeline

def impute_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values in the dataset."""
    numeric_imputer = SimpleImputer(strategy="mean")
    categorical_imputer = SimpleImputer(strategy="most_frequent")

    numeric_features = data.select_dtypes(include="number").columns
    categorical_features = data.select_dtypes(exclude="number").columns

    imputer = ColumnTransformer(
        transformers=[
            ("num_imputer", numeric_imputer, numeric_features),
            ("cat_imputer", categorical_imputer, categorical_features),
        ]
    )

    return imputer.fit_transform(data)


def remove_duplicate_rows(data: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows from the dataset."""
    return data.drop_duplicates()


def transform_date_features(data: pd.DataFrame) -> pd.DataFrame:
    """Transform date features in the dataset."""
    data["Date"] = pd.to_datetime(data["Date"])
    return data


def scale_features(data: pd.DataFrame) -> pd.DataFrame:
    """Scale the numerical features in the dataset."""
    scaler = StandardScaler()
    return scaler.fit_transform(data)


def encode_categorical_features(data: pd.DataFrame) -> pd.DataFrame:
    """Encode the categorical features in the dataset."""
    encoder = OneHotEncoder()
    return encoder.fit_transform(data)


def create_data_pipeline(data: pd.DataFrame) -> pd.DataFrame:
    """Create a reusable data pipeline."""
    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())])
    categorical_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder())]
    )

    preprocessor = FeatureUnion(
        transformer_list=[
            ("numeric", numeric_transformer),
            ("categorical", categorical_transformer),
        ]
    )

    return preprocessor.fit_transform(data)