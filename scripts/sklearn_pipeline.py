import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from datetime import datetime


def preprocess_data(train_df, store_df, output_folder_path):
    """
    Preprocess the input DataFrames by handling the 'PromoInterval' column and merging the data. Split the data into training and validation sets, and save the results as Parquet files.
    
    Args:
        train_df (pandas.DataFrame): The training DataFrame.
        store_df (pandas.DataFrame): The store information DataFrame.
        output_folder_path (str): The path to the folder where the preprocessed data will be saved.
        
    Returns:
        pandas.DataFrame: The preprocessed training DataFrame.
        pandas.DataFrame: The unsplit preprocessed DataFrame.
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

    # Data type conversion
    merged_df['Date'] = pd.to_datetime(merged_df['Date'])
    merged_df['Open'] = merged_df['Open'].astype(bool)
    merged_df['StateHoliday'] = merged_df['StateHoliday'].map({'0': 0, 'a': 1, 'b': 2, 'c': 3}).fillna(-1).astype(int)
    merged_df['SchoolHoliday'] = merged_df['SchoolHoliday'].astype(bool)
    merged_df['StoreType'] = merged_df['StoreType'].map({'a': 0, 'b': 1, 'c': 2, 'd': 3}).fillna(-1).astype(int)
    merged_df['Assortment'] = merged_df['Assortment'].map({'a': 0, 'b': 1, 'c': 2}).fillna(-1).astype(int)
    merged_df['Promo2'] = merged_df['Promo2'].astype(bool)

    # Split the data into training and validation sets
    X = merged_df.drop('Sales', axis=1)
    y = merged_df['Sales']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Combine the features and target variable back into dataframes
    df_train = pd.concat([X_train, y_train], axis=1)
    df_val = pd.concat([X_val, y_val], axis=1)

    # Save the training and validation sets to separate Parquet files
    current_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    train_output_file = os.path.join(output_folder_path, f'preprocessed_train_{current_timestamp}.parquet')
    val_output_file = os.path.join(output_folder_path, f'preprocessed_validation_{current_timestamp}.parquet')
    merged_output_file = os.path.join(output_folder_path, f'preprocessed_merged_{current_timestamp}.parquet')
    merged_df.to_parquet(merged_output_file, index=False)
    df_train.to_parquet(train_output_file, index=False)
    df_val.to_parquet(val_output_file, index=False)

    return df_train, df_val, merged_df