import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
import datetime

class KerasRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, input_shape, epochs=30, batch_size=2):
        self.input_shape = input_shape
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None

    def fit(self, X, y):
        self.model = Sequential()
        self.model.add(LSTM(32, input_shape=self.input_shape))  
        self.model.add(Dense(1))  
        self.model.compile(loss='mse', optimizer='adam') 
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def save_model(self, filename):
        self.model.save(filename)

# Convert df_series to a NumPy array
df_series_array = df_series.values  

# Extract features and target variable from the NumPy array
X = df_series_array[:, :-1] 
y = df_series_array[:, -1]

# Create lagged target variable (adjust as needed)
y_shifted = np.roll(y, -6)  
y_shifted[-6:] = np.nan     
valid_indices = ~np.isnan(y_shifted)
X = X[valid_indices]
y = y_shifted[valid_indices]

# Reshape input
X = X.reshape(X.shape[0], 1, -1)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False) 

# Create the KerasRegressor estimator
estimator = KerasRegressor(input_shape=(X_train.shape[1], X_train.shape[2]))

# Create a pipeline (you can add other steps here if needed)
pipeline = Pipeline([
    ('estimator', estimator)
])

# Fit the pipeline (this will train the Keras model)
pipeline.fit(X_train, y_train) 

# Get current timestamp and save the model
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
model_filename = f'rossman_model_{timestamp}.h5'
pipeline['estimator'].save_model(model_filename)  # Access the estimator within the pipeline

print(f"Model saved as: {model_filename}")