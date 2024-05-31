import pandas as pd
from sklearn.impute import KNNImputer


def handle_missing_promo2_since(df):
    """
    Handles missing values in the Promo2SinceWeek and Promo2SinceYear columns.
    
    Parameters:
    df (pandas.DataFrame): The input DataFrame.
    
    Returns:
    pandas.DataFrame: The updated DataFrame with new binary columns indicating missing values.
    """
    # Create a new column to indicate missing Promo2SinceWeek values
    df['Promo2SinceWeek_is_missing'] = df['Promo2SinceWeek'].isnull().astype(int)
    
    # Create a new column to indicate missing Promo2SinceYear values
    df['Promo2SinceYear_is_missing'] = df['Promo2SinceYear'].isnull().astype(int)
    
    return df




def impute_competition_open_since(df):
    """
    Imputes missing values in the CompetitionOpenSinceMonth and CompetitionOpenSinceYear columns using KNN imputation.
    
    Parameters:
    df (pandas.DataFrame): The input DataFrame.
    
    Returns:
    pandas.DataFrame: The updated DataFrame with imputed values.
    """
    # Create a copy of the DataFrame to avoid modifying the original
    df_imputed = df.copy()
    
    # Combine the columns to be imputed into a single feature matrix
    X = df_imputed[['CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear']].values
    
    # Create a KNNImputer and fit it to the data
    imputer = KNNImputer(n_neighbors=5)
    X_imputed = imputer.fit_transform(X)
    
    # Update the original columns with the imputed values
    df_imputed['CompetitionOpenSinceMonth'] = X_imputed[:, 0]
    df_imputed['CompetitionOpenSinceYear'] = X_imputed[:, 1]
    
    return df_imputed