
import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from feature_engineer import FeatureEngineer
from time_series_utils import check_stationarity, difference_data, plot_acf_pacf, transform_to_supervised, scale_data


def prepare_data(train_df, store_df, directory='../data/05_model_input'):
    """
    Prepare the input DataFrames by handling missing values and applying feature engineering.
    
    Args:
        train_df (pandas.DataFrame): The training DataFrame.
        store_df (pandas.DataFrame): The store information DataFrame.
        
    Returns:
        pandas.DataFrame: The training features.
        pandas.DataFrame: The validation features.
        pandas.Series: The training target variable.
        pandas.Series: The validation target variable.
    """
    # Merge the train and store DataFrames
    merged_df = train_df.merge(store_df, how="left", on="Store")
    
    # Encoding the 'PromoInterval' column
    promo_interval_mapping = {'Jan,Apr,Jul,Oct': 0, 'Feb,May,Aug,Nov': 1, 'Mar,Jun,Sept,Dec': 2}
    merged_df['PromoInterval'] = merged_df['PromoInterval'].map(promo_interval_mapping).fillna(-1).astype(int)
    
    # Handling missing values in the 'PromoInterval' column
    promo_interval_imputer = SimpleImputer(strategy='most_frequent')
    merged_df['PromoInterval'] = promo_interval_imputer.fit_transform(merged_df[['PromoInterval']])

    # Handle missing values for other columns
    competition_distance_imputer = SimpleImputer(strategy='mean')
    competition_open_since_imputer = SimpleImputer(strategy='median')
    promo2_since_imputer = SimpleImputer(strategy='mean')

    competition_distance_pipeline = Pipeline([('imputer', competition_distance_imputer)])
    competition_open_since_pipeline = Pipeline([('imputer', competition_open_since_imputer)])
    promo2_since_pipeline = Pipeline([('imputer', promo2_since_imputer)])

    merged_df['CompetitionDistance'] = competition_distance_pipeline.fit_transform(merged_df[['CompetitionDistance']])
    merged_df[['CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth']] = competition_open_since_pipeline.fit_transform(merged_df[['CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth']])
    merged_df[['Promo2SinceYear', 'Promo2SinceWeek']] = promo2_since_pipeline.fit_transform(merged_df[['Promo2SinceYear', 'Promo2SinceWeek']])

    # Feature Engineering
    feature_engineer = FeatureEngineer()
    merged_df = feature_engineer.fit_transform(merged_df)

    # Check Stationarity
    p_value = check_stationarity(merged_df['Sales'])
    if p_value > 0.05:
        merged_df['Sales'] = difference_data(merged_df['Sales'])
        print("Data was differenced to make it stationary.")

    # Plot ACF and PACF
    plot_acf_pacf(merged_df['Sales'])

    # Transform Data
    n_lags = 1  
    supervised_data = transform_to_supervised(merged_df['Sales'], n_lags)
    supervised_data.columns = ['Lag1', 'Sales']

    # Scale Data
    scaled_data, scaler = scale_data(supervised_data)

    # Split the data 
    X = scaled_data.drop(['Sales'], axis=1)
    y = scaled_data['Sales']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Reset index 
    X_train = X_train.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    
    # Convert to NumPy arrays
    X_train_np = X_train.to_numpy()
    X_val_np = X_val.to_numpy()
    y_train_np = y_train.to_numpy()
    y_val_np = y_val.to_numpy()
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Save NumPy arrays to files
    np.save(os.path.join(directory, 'X_train.npy'), X_train_np)
    np.save(os.path.join(directory, 'X_val.npy'), X_val_np)
    np.save(os.path.join(directory, 'y_train.npy'), y_train_np)
    np.save(os.path.join(directory, 'y_val.npy'), y_val_np)
    
    return X_train_np, X_val_np, y_train_np, y_val_np

