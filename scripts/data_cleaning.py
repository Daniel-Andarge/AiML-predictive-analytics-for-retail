import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin


class DataCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.label_encoders = {}
        self.columns_to_fillna = [
            'CompetitionDistance', 'CompetitionOpenSinceMonth',
            'CompetitionOpenSinceYear', 'Promo2SinceWeek', 'Promo2SinceYear'
        ]
        self.categorical_columns = [
            'StoreType', 'Assortment', 'StateHoliday', 'PromoInterval'
        ]
    
    def fit(self, X, y=None):
        # Fit label encoders on categorical columns
        for col in self.categorical_columns:
            self.label_encoders[col] = LabelEncoder()
            X[col] = X[col].fillna('None')
            self.label_encoders[col].fit(X[col])
        return self

    def transform(self, X):
        X = X.copy()

        # Fill missing values
        for col in self.columns_to_fillna:
            X[col] = X[col].fillna(0)

        # Convert categorical features to numerical
        for col in self.categorical_columns:
            X[col] = X[col].fillna('None')
            X[col] = self.label_encoders[col].transform(X[col])

    
        # Convert data types to float32 for LSTM
        X = X.astype(np.float32)

        return X