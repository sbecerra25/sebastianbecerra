# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 20:32:46 2019

@author: sbece
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

df = sm.datasets.macrodata.load_pandas().data
df.head()
print(sm.datasets.macrodata.NOTE)

index = pd.Index(sm.tsa.datetools.dates_from_range('1959Q1' , '2009Q3'))
df.index = index

df.head()

##PLOT
df['realgdp'].plot()

##TRENDS
gdp_cycle, gdp_trend = sm.tsa.filters.hpfilter(df['realgdp'])
df['trend'] = gdp_trend
df[['realgdp','trend']]["2000-03-31":].plot()


##EMWA (EXPONENTIALLY WEIGHTED MOVING AVERAGES) MODELS
##SIMPLE MOVING AVERAGES (SMA)
airline = pd.read_csv('airline-passengers.csv', index_col = "Month")
airline.head()

airline.dropna(inplace=True)
airline.index = pd.to_datetime(airline.index)
airline.head()

airline.index
airline['6-month-SMA'] = airline['#Passengers'].rolling(window=6).mean()
airline['12-month-SMA'] = airline['#Passengers'].rolling(window=12).mean()

airline.plot(figsize=(10,8))

airline['EWMA-12'] = airline['#Passengers'].ewm(span=12).mean()
       
airline[["#Passengers","EWMA-12"]].plot()
         

##ETS (Error-Trend-Seasonality) MODELS
##Exponential Smoothing
##Trend Methods Models
##ETS Decomposition
airline.plot()

from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(airline['#Passengers'], model = 'multiplicative')
result.seasonal.plot()   
result.trend.plot()     

result.plot() 

##ARIMA MODELS     
##Step 1           
df = pd.read_csv('monthly-milk-production.csv')
df.head()   

df.columns = ["Month", "Milk in pounds per cow"]
df.head()

df.tail()
##to drop a row
##df.drop(168, axis=0, inplace=True)

df['Month'] = pd.to_datetime(df['Month'])

df.set_index('Month', inplace=True)
df.head()

df.describe()
df.describe().transpose()

##Step 2
df.plot()

time_series = df['Milk in pounds per cow']
type(time_series)

time_series.rolling(12).mean().plot(label = '12-Month Rolling Mean')
time_series.rolling(12).std().plot(label = '12-Month Rolling Std')
time_series.plot()
plt.legend()

from statsmodels.tsa.seasonal import seasonal_decompose
decomp = seasonal_decompose(time_series)
decomp.plot()

fig = decomp.plot()
fig.set_size_inches(15,8)

from statsmodels.tsa.stattools import adfuller
result = adfuller(df['Milk in pounds per cow'])

def adf_check(time_series):
    
    result = adfuller(time_series)
    print(" Augmented Dicky-Fuller Test")
    labels = ['ADF Test Statistic', 'p-value', '# of lags', 'Num of Observations used']
              
    for value, label in zip(result, labels):
        print (label+ ' : ' + str(value))
        
    if result[1] <= 0.05:
        print("Strong evidence against null hypothesis")
        print("Reject null hypothesis")
        print("Data has no unit root and is stationary")
    else:
        print("Weak evidence against null hypothesis")
        print("Failed to reject null hypothesis")
        print("Data has a unit root and is non-stationary")
        
adf_check(df['Milk in pounds per cow'])

df['First Difference'] = df['Milk in pounds per cow'] -df['Milk in pounds per cow'].shift(1)

df['First Difference'].plot()     

adf_check(df['First Difference'].dropna())   

##Si la primera diferencia es NO-estacionaria, tenemos que diferenciar tantas veces hasta
##que la sea
##A modo de ejemplo mostraremos la segunda diferencia

df['Second Difference'] = df['First Difference'] -df['First Difference'].shift(1)

df['Second Difference'].plot()     

adf_check(df['Second Difference'].dropna())   


#####Seasonal difference
df['Seasonal Difference'] = df['Milk in pounds per cow'] -df['Milk in pounds per cow'].shift(12)
df['Seasonal Difference'].plot()   
adf_check(df['Seasonal Difference'].dropna())

#####Seasonal difference - 1st difference
df['Seasonal First Difference'] = df['First Difference'] - df['First Difference'].shift(12)
df['Seasonal First Difference'].plot()     
adf_check(df['Seasonal First Difference'].dropna())   

##ACF and PACF
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
fig_first = plot_acf(df['First Difference'].dropna())
fig_seasonal_first = plot_acf(df['Seasonal First Difference'].dropna())

from pandas.plotting import autocorrelation_plot
autocorrelation_plot(df['Seasonal First Difference'].dropna())

result = plot_pacf(df['Seasonal First Difference'].dropna())

plot_acf(df['Seasonal First Difference'].dropna())
plot_pacf(df['Seasonal First Difference'].dropna())

##deploy an ARIMA model
from statsmodels.tsa.arima_model import ARIMA
model = sm.tsa.statespace.SARIMAX(df['Milk in pounds per cow'], order = (0,1,0), seasonal_order=(1,1,1,12))

results = model.fit()

print(results.summary())

results.resid #residuals
results.resid.plot()

results.resid.plot(kind='kde') #kernel density estimator

df['forecast'] = results.predict(start=150, end=168)
df[['Milk in pounds per cow', 'forecast']].plot()

##Predict new values
from pandas.tseries.offsets import DateOffset
future_dates = [df.index[-1] + DateOffset(months=x) for x in range(1,24)]

future_dates

future_df = pd.DataFrame(index=future_dates, columns=df.columns)
future_df

final_df = pd.concat([df, future_df])
final_df

final_df.tail()
final_df['forecast'] = results.predict(start = 168, end=192)

final_df['Milk in pounds per cow'].plot()
final_df['forecast'].plot()

final_df[['Milk in pounds per cow', 'forecast']].plot()