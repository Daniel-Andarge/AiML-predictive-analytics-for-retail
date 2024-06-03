from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.engineer_features(X)

    def engineer_features(self, df):
        """
        Preprocess the datetime columns and extract additional features from the dataset.
        
        Parameters:
        df (pandas.DataFrame): The input DataFrame.
        
        Returns:
        pandas.DataFrame: The transformed DataFrame with additional features.
        """

        # Extract weekdays and weekends
        df['Weekday'] = df['DayOfWeek']  # Assuming 'DayOfWeek' is already numeric
        df['IsWeekend'] = df['Weekday'] >= 5  # True for Saturday and Sunday

        # Create a column for StateHoliday converted to datetime
        df['StateHolidayDate'] = df['Date'][df['StateHoliday'] != 0]

        # Function to calculate the number of days to the next holiday
        def days_to_holiday(row, holidays):
            future_holidays = holidays[holidays > row['Date']]
            if len(future_holidays) > 0:
                return (future_holidays.min() - row['Date']).days
            else:
                return np.nan

        # Function to calculate the number of days since the last holiday
        def days_since_holiday(row, holidays):
            past_holidays = holidays[holidays <= row['Date']]
            if len(past_holidays) > 0:
                return (row['Date'] - past_holidays.max()).days
            else:
                return np.nan

        # Extract unique holiday dates
        unique_holidays = df['StateHolidayDate'].dropna().unique()

        # Apply the functions to calculate days to and after holidays
        df['DaysToHoliday'] = df.apply(lambda row: days_to_holiday(row, unique_holidays), axis=1)
        df['DaysAfterHoliday'] = df.apply(lambda row: days_since_holiday(row, unique_holidays), axis=1)

        # Drop the intermediate StateHolidayDate column
        df.drop(columns=['StateHolidayDate'], inplace=True)

        # Beginning, Middle, and End of the Month
        df['DayOfMonth'] = df['Day']
        df['IsBeginningOfMonth'] = (df['DayOfMonth'] <= 10)
   

        # Additional Features
        df['Season'] = df['Month'].apply(lambda x: 0 if x in [12, 1, 2] else 1 if x in [3, 4, 5] else 2 if x in [6, 7, 8] else 3)
 
        return df