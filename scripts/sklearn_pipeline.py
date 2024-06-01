import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from time_series_utils import check_stationarity, difference_data, plot_acf_pacf, transform_to_supervised, scale_data
from feature_engineer import FeatureEngineer

class PromoIntervalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.promo_interval_mapping = {'Jan,Apr,Jul,Oct': 0, 'Feb,May,Aug,Nov': 1, 'Mar,Jun,Sept,Dec': 2}
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X['PromoInterval'] = X['PromoInterval'].map(self.promo_interval_mapping).fillna(-1).astype(int)
        return X

class CustomImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy='mean'):
        self.strategy = strategy
        self.imputer = SimpleImputer(strategy=self.strategy)
    
    def fit(self, X, y=None):
        self.imputer.fit(X)
        return self
    
    def transform(self, X):
        return self.imputer.transform(X)

class CustomFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_engineer = FeatureEngineer()
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return self.feature_engineer.fit_transform(X)

class CheckStationarity(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        p_value = check_stationarity(X['Sales'])
        if p_value > 0.05:
            X['Sales'] = difference_data(X['Sales'])
            print("Data was differenced to make it stationary.")
        return X

class CustomPipeline(Pipeline):
    def fit(self, X, y=None, **fit_params):
        X = self.steps[0][1].transform(X)
        return super().fit(X, y, **fit_params)

def prepare_data(train_df, store_df, directory='../data/05_model_input'):
    merged_df = train_df.merge(store_df, how="left", on="Store")
    
    pipeline = CustomPipeline([
        ('promo_interval_encoder', PromoIntervalEncoder()),
        ('promo_interval_imputer', CustomImputer(strategy='most_frequent')),
        ('competition_distance_imputer', CustomImputer(strategy='mean')),
        ('competition_open_since_imputer', CustomImputer(strategy='median')),
        ('promo2_since_imputer', CustomImputer(strategy='mean')),
        ('feature_engineer', FeatureEngineer()),  # Including Feature Engineering step
        ('check_stationarity', CheckStationarity()),
    ])

    X_train, X_val, y_train, y_val = train_test_split(merged_df.drop('Sales', axis=1), merged_df['Sales'], test_size=0.2, random_state=42)
    
    X_train = pipeline.fit_transform(X_train)
    X_val = pipeline.transform(X_val)
    
    X_train = X_train.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    
    X_train_np = X_train.to_numpy()
    X_val_np = X_val.to_numpy()
    y_train_np = y_train.to_numpy()
    y_val_np = y_val.to_numpy()
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    np.save(os.path.join(directory, 'X_train.npy'), X_train_np)
    np.save(os.path.join(directory, 'X_val.npy'), X_val_np)
    np.save(os.path.join(directory, 'y_train.npy'), y_train_np)
    np.save(os.path.join(directory, 'y_val.npy'), y_val_np)
    
    # Plot ACF and PACF
    plot_acf_pacf(merged_df['Sales'])
    
    return X_train_np, X_val_np, y_train_np, y_val_np
