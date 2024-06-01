import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

from feature_engineer import engineer_features

def preprocess_data(train_df, store_df, output_folder_path):
    """
    Preprocess the input DataFrames by handling the 'PromoInterval' column and merging the data.
    Split the data into training and validation sets, and save the results as Parquet files.
    
    Args:
        train_df (pandas.DataFrame): The training DataFrame.
        store_df (pandas.DataFrame): The store information DataFrame.
        output_folder_path (str): The path to the folder where the preprocessed data will be saved.
        
    Returns:
        numpy.ndarray: The training features as a NumPy array.
        numpy.ndarray: The validation features as a NumPy array.
        numpy.ndarray: The training target variable as a NumPy array.
        numpy.ndarray: The validation target variable as a NumPy array.
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
    engineer_features_transformer = FunctionTransformer(engineer_features)
    engineered_df = engineer_features_transformer.fit_transform(merged_df)

    # Data type conversion
    # Converting boolean columns to float
    bool_cols = engineered_df.select_dtypes(include='bool').columns
    engineered_df[bool_cols] = engineered_df[bool_cols].astype(float)

    # Converting integer columns to float
    int_cols = engineered_df.select_dtypes(include=['int64', 'int32']).columns
    engineered_df[int_cols] = engineered_df[int_cols].astype(float)
    
    # Convert categorical (object) columns to category codes and then to float
    object_cols = engineered_df.select_dtypes(include='object').columns
    for col in object_cols:
        engineered_df[col] = engineered_df[col].astype('category').cat.codes.astype(float)

    # Split the data
    X = engineered_df.drop(columns=['Sales', 'Date'])
    y = engineered_df['Sales']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize the features for LSTM
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Reshape the data for LSTM
    X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_val_lstm = X_val_scaled.reshape((X_val_scaled.shape[0], 1, X_val_scaled.shape[1]))

    # Save the preprocessed data as NumPy arrays
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    
    np.save(os.path.join(output_folder_path, 'X_train.npy'), X_train_lstm)
    np.save(os.path.join(output_folder_path, 'X_val.npy'), X_val_lstm)
    np.save(os.path.join(output_folder_path, 'y_train.npy'), y_train.to_numpy())
    np.save(os.path.join(output_folder_path, 'y_val.npy'), y_val.to_numpy())

    return X_train_lstm, X_val_lstm, y_train.to_numpy(), y_val.to_numpy()

def build_and_train_lstm(X_train, y_train, X_val, y_val, output_model_path):
    """
    Build and train an LSTM regression model.
    
    Args:
        X_train (numpy.ndarray): The training features.
        y_train (numpy.ndarray): The training target variable.
        X_val (numpy.ndarray): The validation features.
        y_val (numpy.ndarray): The validation target variable.
        output_model_path (str): The path to save the trained model.
    """
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # Early stopping to avoid overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, callbacks=[early_stopping], verbose=1)

    # Save the model
    model.save(output_model_path)

