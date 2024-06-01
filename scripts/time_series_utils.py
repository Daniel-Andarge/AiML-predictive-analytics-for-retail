import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

def check_stationarity(data):
    result = adfuller(data)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    for key, value in result[4].items():
        print('Critical Values:')
        print(f'   {key}, {value}')
    return result[1]

def difference_data(data):
    return data.diff().dropna()

def plot_acf_pacf(data, lags=50):
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    sm.graphics.tsa.plot_acf(data, lags=lags, ax=ax[0])
    sm.graphics.tsa.plot_pacf(data, lags=lags, ax=ax[1])
    plt.show()

def transform_to_supervised(data, n_lags=1):
    df = pd.DataFrame(data)
    columns = []
    for i in range(n_lags, 0, -1):
        columns.append(df.shift(i))
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df.dropna(inplace=True)
    return df

def scale_data(data):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler
