#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
import arch
register_matplotlib_converters()
# %matplotlib inline
import warnings
import requests
warnings.filterwarnings("ignore")
import datetime, pytz
# import dataset
def dateparse (time_in_secs): 
    return pytz.utc.localize(datetime.datetime.fromtimestamp(float(time_in_secs)))
df = pd.read_csv('Data/bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv',parse_dates=[0], date_parser=dateparse) 

df['Timestamp'] = df['Timestamp'].dt.tz_localize(None)
df.rename(columns={"Volume_(Currency)": "Volume"},inplace=True)
#df = df.set_index('Timestamp')
df['Weighted_Price'].fillna(method='ffill', inplace=True)
df['Open'].fillna(method='ffill', inplace=True)
df['High'].fillna(method='ffill', inplace=True)
df['Low'].fillna(method='ffill', inplace=True)
df['Close'].fillna(method='ffill', inplace=True)

df = df.dropna()

# making copy for making changes
dataset_for_prediction = df.copy()
dataset_for_prediction['Actual']=dataset_for_prediction['Weighted_Price'].shift()
dataset_for_prediction=dataset_for_prediction.dropna()
dataset_for_prediction['Timestamp'] =pd.to_datetime(dataset_for_prediction['Timestamp'])
dataset_for_prediction.index= dataset_for_prediction['Timestamp']

# normalizing exogeneous variables
from sklearn.preprocessing import MinMaxScaler
sc_in = MinMaxScaler(feature_range=(0, 1))
scaled_input = sc_in.fit_transform(dataset_for_prediction[['Low', 'High', 'Open', 'Close', 'Volume', 'Weighted_Price']])
scaled_input = pd.DataFrame(scaled_input, index=dataset_for_prediction.index)
X= scaled_input
X.rename(columns={0:'Low', 1:'High', 2:'Open', 3:'Close', 4:'Volume', 5:'Weighted_Price'}, inplace=True)
print(X.head())


sc_out = MinMaxScaler(feature_range=(0, 1))
scaler_output = sc_out.fit_transform(dataset_for_prediction[['Actual']])
scaler_output =pd.DataFrame(scaler_output, index=dataset_for_prediction.index)
y=scaler_output
y.rename(columns={0:'BTC Price next day'}, inplace= True)
y.index=dataset_for_prediction.index
print(y.head())

# train-test split
train_size=int(len(df) *0.9)
test_size = int(len(df)) - train_size
train_X, train_y = X[:train_size].dropna(), y[:train_size].dropna()
test_X, test_y = X[train_size:].dropna(), y[train_size:].dropna()


from statsmodels.tsa.statespace.sarimax import SARIMAX

# Init the model
predic_garch =[]
for i in range(test_size):
    model= SARIMAX(pd.concat([train_y,test_y.iloc[:i+1]]),
    exog=pd.concat([train_X,test_X.iloc[:i+1]]),
    order=(0,1,1),
    seasonal_order =(0, 0, 1, 12),
    enforce_invertibility=False, enforce_stationarity=False)
    results= model.fit()
    garch = arch.arch_model(results.resid, p=1, q=1,vol='GARCH')
    garch_model = garch.fit(update_freq=1)
    garch_forecast = garch_model.forecast(start = train_size-1,horizon=1,method='simulation')
    predicted_et = garch_forecast.mean['h.1'].iloc[-1]
    predic_garch.append(predicted_et)
    print(predicted_et)


model= SARIMAX(train_y,
 exog=train_X,
 order=(0,1,1),
 seasonal_order =(0, 0, 1, 12),
 enforce_invertibility=False, enforce_stationarity=False)


# training the model
results= model.fit()

# plotting residuals
results.resid.plot()

# making preditions
predictions= results.predict(start =train_size, end=train_size+test_size-2,exog=test_X)
act= pd.DataFrame(scaler_output.iloc[train_size:, 0])
predictions=pd.DataFrame(predictions)
predictions.reset_index(drop=True, inplace=True)
predictions.index=test_X.index
predictions['Actual'] = act['BTC Price next day']
predictions.rename(columns={'predicted_mean':'Pred'}, inplace=True)
print(predictions)
for i in range(len(predictions)) : 
    predictions.iloc[i,0]= predictions.iloc[i,0]+predic_garch[i]

# plotting the results
trainPredict = sc_out.inverse_transform(predictions[['Pred']])
testPredict = sc_out.inverse_transform(predictions[['Actual']])

plt.figure(figsize=(20,10))
plt.plot(predictions.index, trainPredict, label='Pred', color='blue')
plt.plot(predictions.index, testPredict, label='Actual', color='red')
plt.legend()
plt.show()

from statsmodels.tools.eval_measures import rmse
error=rmse(trainPredict, testPredict)
print("RMSE:",error)

