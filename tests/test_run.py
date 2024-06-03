update the below code to Scale the data in the (-1, 1) range At the approprate place 


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import pandas as pd
from sklearn.model_selection import train_test_split

# Convert df_series to a NumPy array
df_series_array = df_series.values  

# Extract features and target variable from the NumPy array
X = df_series_array[:, :-1]  # Select all columns except the last one (assuming 'Sales' is the last column)
y = df_series_array[:, -1]   # Select the last column ('Sales')

# Create lagged target variable for supervised learning (6 weeks ahead)
# (Note: This part might need adjustments depending on how you want to handle the shift with a NumPy array)
y_shifted = np.roll(y, -6)  # Shift the target values by 6 positions
y_shifted[-6:] = np.nan     # Set the last 6 values to NaN (since they don't have corresponding targets)

# Remove NaN values (adjust this if you want to handle missing values differently)
valid_indices = ~np.isnan(y_shifted)
X = X[valid_indices]
y = y_shifted[valid_indices]

# Reshape input to be [samples, time steps, features]
X = X.reshape(X.shape[0], 1, -1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = Sequential()
model.add(LSTM(32, input_shape=(X_train.shape[1], X_train.shape[2])))  
model.add(Dense(1))  
model.compile(loss='mse', optimizer='adam') 


model.fit(X, y, epochs=50, batch_size=32, verbose=2)

y_pred = model.predict(X_test)

# Compare predicted values with actual values
for i in range(len(y_test)):
    print(f"Actual: {y_test[i]:.2f}, Predicted: {y_pred[i][0]:.2f}")


# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")