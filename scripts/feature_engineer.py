# feature_engineer.py
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

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
        
        # Ensure 'Date' column is datetime type
        if df['Date'].dtype == 'object':
            df['Date'] = pd.to_datetime(df['Date'])
        
        # Extract weekdays and weekends
        df['Weekday'] = df['Date'].dt.weekday  # Monday=0, Sunday=6
        df['IsWeekend'] = df['Weekday'] >= 5  # True for Saturday and Sunday
        
        # Create a column for StateHoliday converted to datetime
        df['StateHolidayDate'] = df['Date'][df['StateHoliday'] != 0]
        
        # Function to calculate the number of days to the next holiday
        def days_to_holiday(row, holidays):
            future_holidays = holidays[holidays > row['Date']]
            if len(future_holidays) > 0:
                return (future_holidays.min() - row['Date']).days
            else:
                return None

        # Function to calculate the number of days since the last holiday
        def days_since_holiday(row, holidays):
            past_holidays = holidays[holidays <= row['Date']]
            if len(past_holidays) > 0:
                return (row['Date'] - past_holidays.max()).days
            else:
                return None
        
        # Extract unique holiday dates
        unique_holidays = df['StateHolidayDate'].dropna().unique()
        
        # Apply the functions to calculate days to and after holidays
        df['DaysToHoliday'] = df.apply(lambda row: days_to_holiday(row, unique_holidays), axis=1)
        df['DaysAfterHoliday'] = df.apply(lambda row: days_since_holiday(row, unique_holidays), axis=1)
        
        # Drop the intermediate StateHolidayDate column
        df.drop(columns=['StateHolidayDate'], inplace=True)
        
        # Beginning, Middle, and End of the Month
        df['DayOfMonth'] = df['Date'].dt.day
        df['IsBeginningOfMonth'] = (df['DayOfMonth'] <= 10)
        df['IsMidMonth'] = (df['DayOfMonth'] > 10) & (df['DayOfMonth'] <= 20)
        df['IsEndOfMonth'] = (df['DayOfMonth'] > 20)
        
        # Additional Features
        df['Month'] = df['Date'].dt.month
        df['Quarter'] = df['Date'].dt.quarter
        df['Year'] = df['Date'].dt.year
        df['Season'] = df['Month'].apply(lambda x: 'Winter' if x in [12, 1, 2] else 'Spring' if x in [3, 4, 5] else 'Summer' if x in [6, 7, 8] else 'Fall')
        df['PromoDuration'] = (df['Promo2SinceYear'] * 52 + df['Promo2SinceWeek']) - df['Date'].dt.isocalendar().week
        df['CompetitionDuration'] = (df['Date'].dt.year - df['CompetitionOpenSinceYear'].astype(int)) * 12 + (df['Date'].dt.month - df['CompetitionOpenSinceMonth'].astype(int))

        # Convert boolean columns to float
        bool_cols = df.select_dtypes(include='bool').columns
        df[bool_cols] = df[bool_cols].astype(float)

        # Convert integer columns to float
        int_cols = df.select_dtypes(include=['int64', 'int32']).columns
        df[int_cols] = df[int_cols].astype(float)
        
        # Convert categorical (object) columns to category codes and then to float
        object_cols = df.select_dtypes(include='object').columns
        for col in object_cols:
            df[col] = df[col].astype('category').cat.codes.astype(float)
        
        return df
