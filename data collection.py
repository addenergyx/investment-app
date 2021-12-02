# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 20:50:23 2021

@author: david
"""

import imaplib
import os
import email
from bs4 import BeautifulSoup
import pandas as pd
from dotenv import load_dotenv
import stockstats
import collections
from pandas_datareader import data as web
import smtplib, ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from prettytable import PrettyTable
import re
import yfinance as yf
from datetime import datetime, timedelta
import plotly.express as px
from plotly.offline import plot
import plotly.graph_objects as go
from pytrends.request import TrendReq
from sqlalchemy import create_engine
from pytrends import dailydata
from forex_python.converter import CurrencyRates
import glob
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from pandas.tseries.offsets import *
from plotly.subplots import make_subplots
from keras.models import Sequential

leaderboard = pd.read_csv('leaderboard.csv')
trades = pd.read_csv('Investment trades.csv')

# List of colours to prevent plotly repeating colours in plots
colours = px.colors.qualitative.G10 + px.colors.qualitative.Plotly + px.colors.qualitative.Light24 + px.colors.qualitative.Dark24 + px.colors.qualitative.Alphabet

grouped = leaderboard.groupby('Stock')

leads = pd.DataFrame(columns=leaderboard.columns)

for name, group in grouped:

    group['Date'] = pd.to_datetime(group['Date'], format='%d/%m/%Y', dayfirst=True)
    
    group.index = pd.to_datetime(group.Date)
    
    cal = calendar()
    holidays = cal.holidays(start='2020-11-11', end='2021-09-09') # Holidays to be removed
    
    c = group.reindex(pd.date_range('2020-11-11', '2021-09-09', freq=BDay())).drop(columns='Date').reset_index()
    
    df = c[~c['index'].isin(holidays)].rename(columns={'index':'Date'})
    
    df['Stock'] = name
    
    leads = leads.append(df, ignore_index = True)
    
# No missing data when scraped
    
leaderboard.isnull().sum()
    
# Number of missing days
# Script broke several times during project

tesla = leads[leads['Stock'] == 'Tesla'] # Chose tesla because I know it has always been in the top 100
tesla.isnull().sum() # 47 missing days

fig = px.line(leaderboard, x="Date", y="User_count", title='Leaderboard', color='Stock')
plot(fig)
    
ace = trades.groupby(['Ticker Symbol', 'Trading day']).sum().reset_index() # Sum of result
base = trades.groupby(['Ticker Symbol', 'Trading day']).count().reset_index() # Number of executions for a given day

base['Count'] = base['Result']
    
df = ace.merge(base[['Ticker Symbol', 'Trading day', 'Count']], on=['Ticker Symbol', 'Trading day'])

df = df.merge(trades[['Ticker Symbol', 'Trading day', 'Sector', 'Industry']], on=['Ticker Symbol', 'Trading day'])  
    
df.drop_duplicates(inplace=True)
    
# fig = px.scatter(df, x="gdpPercap", y="lifeExp", animation_frame="year", animation_group="country",
#            size="pop", color="continent", hover_name="country",
#            log_x=True, size_max=55, range_x=[100,100000], range_y=[25,90])    
    

tops = leads.groupby('Stock').sum().sort_values('User_count').reset_index()['Stock'][-30:][::-1]

# Change datetime to string for plotly
leads['Dates'] = leads['Date'].apply(lambda x: x.strftime('%d-%m-%Y'))

fig = px.bar(leads[leads['Stock'].isin(tops)], x="Stock", y="User_count", color="Stock", text='User_count',
  animation_frame="Dates", animation_group="Stock", color_discrete_sequence=colours, range_y=[0,200000]
  )
plot(fig)

# animated bar with filled values

grouped = leads.groupby('Stock')

bars = pd.DataFrame(columns=leads.columns)

for name, group in grouped:
        
    # interpolate data to fill gaps
    group.index = group['Date']
    group['User_count'].interpolate(method='time', inplace=True)
    group = group.reset_index(drop=True)
    
    bars = bars.append(group, ignore_index = True)

fig = px.bar(bars[bars['Stock'].isin(tops)], x="Stock", y="User_count", color="Stock", text='User_count',
  animation_frame="Dates", animation_group="Stock", color_discrete_sequence=colours, range_y=[0,200000]
  )
plot(fig)

## Trades for bubble plot

grouped = df.groupby('Ticker Symbol')

points = pd.DataFrame(columns=df.columns)

#group = df[df['Ticker Symbol'] == 'TSLA']

for name, group in grouped:
    
    group['Trading day'] = pd.to_datetime(group['Trading day'], format='%Y-%m-%d', dayfirst=True)
    
    group['Result_cumsum'] = group['Result'].cumsum()
    group['Count_cumsum'] = group['Count'].cumsum()
    group['Total_cost_cumsum'] = group['Total cost'].cumsum()
    
    group.index = pd.to_datetime(group['Trading day'])
    
    cal = calendar()
    holidays = cal.holidays(start='2020-03-20', end='2021-10-13') # Holidays to be removed
    
    #group = group.drop(columns='Trading day')
    
    c = group.reindex(pd.date_range('2020-03-20', '2021-10-13', freq=BDay())).drop(columns='Trading day').reset_index()
    
    df = c[~c['index'].isin(holidays)].rename(columns={'index':'Trading day'})
    
    df['Ticker Symbol'] = name
    
    df[['Sector', 'Industry']] =  df[['Sector', 'Industry']].bfill().ffill()
    
    # zero fill up till first trade then forward fill
    first_trade = df.notnull()['Shares'].idxmax()
    df[:first_trade] = df[:first_trade].fillna(0)
    
    # Forward fill missing dates
    df = df.ffill()
    
    points = points.append(df, ignore_index = True)
    
points.info()

points['Trading day'] = points['Trading day'].apply(lambda x: x.strftime('%d-%m-%Y'))

 
fig = px.scatter(points, x="Count_cumsum", y="Result_cumsum", animation_frame="Trading day", animation_group="Ticker Symbol",
        size="Count_cumsum", color="Sector", hover_name="Ticker Symbol", color_discrete_sequence=colours,
        log_y=True,
        size_max=100,
        range_x=[0,300],  #range_y=[-3500,3500]
        )
fig.update_xaxes(title_text="Number of executions")
fig.update_yaxes(title_text="Profit/Loss")
plot(fig) 

fig = px.scatter(points, x="Total_cost_cumsum", y="Result_cumsum", animation_frame="Trading day", animation_group="Ticker Symbol",
        size="Count_cumsum", color="Sector", hover_name="Ticker Symbol", color_discrete_sequence=colours,
        #log_y=True,
        size_max=100,
        range_x=[0,300],  #range_y=[-3500,3500]
        )
fig.update_xaxes(title_text="Number of executions")
fig.update_yaxes(title_text="Profit/Loss")
plot(fig) 

# Portfolio bubble plot
trades = pd.read_csv('Investment trades.csv')

ace = trades.groupby(['Ticker Symbol']).sum().reset_index() # Sum of result
base = trades.groupby(['Ticker Symbol']).count().reset_index() # Number of executions for a given day

base['Executions'] = base['Result']
    
df = ace.merge(base[['Ticker Symbol', 'Executions']], on=['Ticker Symbol'])

df = df.merge(trades[['Ticker Symbol', 'Sector', 'Industry']], on=['Ticker Symbol'])  
    
df.drop_duplicates(inplace=True)

fig = px.scatter(df, x="Total amount", y="Result", size="Executions", color="Sector", hover_name="Ticker Symbol", text="Ticker Symbol",
           log_x=True, log_y=True, size_max=55, color_discrete_sequence=colours#, range_x=[100,100000], range_y=[25,90]
           )

#fig.update_traces(textposition='top center')

plot(fig)

trades['Industry'].drop_duplicates()


# Longest stretch of missing data between 12/7 and 23/7 (10 working days)
tesla = leads[leads['Stock'] == 'Tesla']
tesla.isnull().sum()
#tesla = tesla.dropna()

# interpolate data to fill gaps
tesla.index = tesla['Date']
tesla['User_count'].interpolate(method='time', inplace=True)
tesla = tesla.reset_index(drop=True)


#ticker = tesla['Stock'].iloc[0]
end = tesla['Date'].iloc[-1]
start = tesla['Date'].iloc[0]

yf.pdr_override()
index = web.get_data_yahoo('TSLA', start=start, end=end)
index = index.reset_index()

df = index.merge(tesla, on=['Date'], how='left')

fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(
    go.Scatter(x=df['Date'], y=df['User_count'], name="Trading 212 User count"),
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(x=df['Date'], y=df['Close'], name="Closing price"),
    secondary_y=True,
)

fig.update_layout(
    title_text=f"Clsoing stock price vs User count of Tesla",
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
))

fig.update_xaxes(title_text="Date")
fig.update_yaxes(#range=[-0.25, 0.20], 
                 title_text="<b>Primary</b> Stock closing price", secondary_y=False)
fig.update_yaxes(#range=[-20, 20], 
                 title_text="<b>Secondary</b> Trading 212 User count", secondary_y=True)
    
plot(fig)


# df['Close'] = df['Close'].pct_change()
# df['User_count'] = df['User_count'].pct_change()

fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(
    go.Scatter(x=df['Date'], y=df['User_count'], name="Trading 212 User count"),
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(x=df['Date'], y=df['Close'], name="Closing price"),
    secondary_y=True,
)

fig.update_layout(
    title_text=f"Stock Price Percentage Change vs User count Percentage Change of Tesla",
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
))

fig.update_xaxes(title_text="Date")
fig.update_yaxes(#range=[-0.25, 0.20], 
                 title_text="<b>Primary</b> Stock closing price", secondary_y=False)
fig.update_yaxes(#range=[-20, 20], 
                 title_text="<b>Secondary</b> Trading 212 User count", secondary_y=True)
    
plot(fig)

# correlation between closing price and user count
from scipy.stats import pearsonr
corr, _ = pearsonr(df['Close'][1:], df['User_count'][1:])


pre_inter_tesla = leads[leads['Stock'] == 'Tesla']
tesla.isnull().sum()
#pre_inter_tesla = pre_inter_tesla.dropna()

fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(
    go.Scatter(x=pre_inter_tesla['Date'], y=pre_inter_tesla['User_count'], name="Trading 212 User count"),
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(x=df['Date'], y=df['Close'], name="Closing price"),
    secondary_y=True,
)

fig.update_layout(
    title_text=f"Clsoing stock price vs User count of Tesla",
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
))

fig.update_xaxes(title_text="Date")
fig.update_yaxes(#range=[-0.25, 0.20], 
                 title_text="<b>Primary</b> Stock closing price", secondary_y=False)
fig.update_yaxes(#range=[-20, 20], 
                 title_text="<b>Secondary</b> Trading 212 User count", secondary_y=True)
    
plot(fig)


def ml_model():

    leaderboard = pd.read_csv('leaderboard.csv')

    leads = pd.DataFrame(columns=leaderboard.columns)

    grouped = leaderboard.groupby('Stock')

    for name, group in grouped:
    
        group['Date'] = pd.to_datetime(group['Date'], format='%d/%m/%Y', dayfirst=True)
        
        group.index = pd.to_datetime(group.Date)
        
        cal = calendar()
        holidays = cal.holidays(start='2020-11-11', end='2021-09-09') # Holidays to be removed
        
        c = group.reindex(pd.date_range('2020-11-11', '2021-09-09', freq=BDay())).drop(columns='Date').reset_index()
        
        df = c[~c['index'].isin(holidays)].rename(columns={'index':'Date'})
        
        df['Stock'] = name
        
        leads = leads.append(df, ignore_index = True)
    
    tesla = leads[leads['Stock'] == 'Tesla']
    
    # interpolate data to fill gaps
    tesla.index = tesla['Date']
    tesla['User_count'].interpolate(method='time', inplace=True)
    tesla = tesla.reset_index(drop=True)
    
    #ticker = tesla['Stock'].iloc[0]
    end = tesla['Date'].iloc[-1]
    start = tesla['Date'].iloc[0]
    
    yf.pdr_override()
    index = web.get_data_yahoo('TSLA', start=start, end=end)
    index = index.reset_index()
    
    df = index.merge(tesla, on=['Date'], how='left')
    
    model_df = df[['Date', 'Open', 'Adj Close', 'Volume', 'User_count']]
    
    model_df['Adj Close'] = model_df['Adj Close'].pct_change() # Percentage change
    
    model_df.dropna(inplace=True)
        
    training_dataset = model_df[model_df['Date'] < datetime(2021, 7, 20)]
    test_data = model_df[model_df['Date'] >= datetime(2021, 7, 20)]
    
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np
    
    ## Using normalisation as will be using sigmoid function as activation functionn of output layer
    sc = MinMaxScaler()
    training_data = sc.fit_transform(training_dataset.drop(['Date'], axis=1))
    training_data.shape[0]
    
    window = 30
    
    x_train = [training_data[i-window:i] for i in range(60, training_data.shape[0])]
    
    # Open stock price
    y_train = [training_data[i, 0] for i in range(60, training_data.shape[0])]
    
    x_train, y_train = np.array(x_train ),  np.array(y_train)
    
    x_train.shape, y_train.shape
    
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Dropout
    
    ## Model architecture
    model = Sequential()
    
    ## Chose 50 nodes for high dimensionality
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    
    # Dropout Regularisation to pervent overfitting. 20% is a common choice
    model.add(Dropout(0.2))
    
    # Layers 2 - 3
    for x in range(2,4):
        print(f'Initalise layer {x}')
        model.add(LSTM(50, return_sequences=True))
        model.add(Dropout(0.2))
    
    # Final layer
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    
    # Output layer
    model.add(Dense(1))
    
    model.compile(loss="mean_squared_error", optimizer="adam") # Try RMWprop optimizer after
    
    model.summary()
    
    ## 32 recommended batch size
    model.fit(
        x_train, y_train, epochs=100, batch_size=32, verbose=1, validation_split=0.2 #, validation_data=(Xtest, ytest)
    ) # Loss progressively got better (lower)
    
    ## https://machinelearningmastery.com/how-to-use-the-timeseriesgenerator-for-time-series-forecasting-in-keras/
    
    ## Predictions ##
    
    #stock_test_data = model_df[model_df.index >= len(training_data)]
    #dataset = model_df.drop(['Date'], axis=1)
    
    # Adding last window days of training set to test set for LSTM
    total_test_data = pd.concat((training_dataset.tail(window), test_data), ignore_index = True).drop(['Date'], axis=1)
    
    scaled_test_data = sc.transform(total_test_data)
    
    #stock_test_data = test_data.drop(['Date'], axis=1)
    
    x_test = [scaled_test_data[i-window:i] for i in range(window, scaled_test_data.shape[0])]
    y_test = [scaled_test_data[i, 0] for i in range(window, scaled_test_data.shape[0])]
    
    x_test, y_test = np.array(x_test),  np.array(y_test)
    
    x_test.shape, y_test.shape
    
    y_pred = model.predict(x_test)
    
    ## How to use inverse_transform in MinMaxScaler for a column in a matrix
    ## https://stackoverflow.com/questions/49330195/how-to-use-inverse-transform-in-minmaxscaler-for-a-column-in-a-matrix
    # invert predictions
    # Original scaler variable (sc) won't work as it expects a 2D array instead of the 1D y_pred array we are trying to parse.
    scale = MinMaxScaler()
    scale.min_, scale.scale_ = sc.min_[0], sc.scale_[0]
    y_pred = scale.inverse_transform(y_pred)
    y_test = test_data['Open']
    
    y_pred = [x[0] for x in y_pred.tolist()]
    
    test_dates = pd.Series([training_dataset['Date'].iloc[-1]]).append(test_data['Date'], ignore_index=True)
    y_pred_graph =  [training_dataset['Open'].iloc[-1]] + y_pred
    y_test_graph = pd.Series([training_dataset['Open'].iloc[-1]]).append(y_test, ignore_index=True).tolist()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=training_dataset['Date'], y=training_dataset['Open'], name='Past Stock Price'))
    fig.add_trace(go.Scatter(x=test_dates, y=y_pred_graph, name='Predicted Stock Price'))
    fig.add_trace(go.Scatter(x=test_dates, y=y_test_graph, name='Actual Stock Price'))
    fig.update_layout(title="Predicted vs Actual Stock Price", xaxis_title="Date", yaxis_title="Opening Price")
    
    return fig

fig = ml_model()
plot(fig)


def ml_model():

    leaderboard = pd.read_csv('leaderboard.csv')

    leads = pd.DataFrame(columns=leaderboard.columns)

    grouped = leaderboard.groupby('Stock')

    for name, group in grouped:
    
        group['Date'] = pd.to_datetime(group['Date'], format='%d/%m/%Y', dayfirst=True)
        
        group.index = pd.to_datetime(group.Date)
        
        cal = calendar()
        holidays = cal.holidays(start='2020-11-11', end='2021-09-09') # Holidays to be removed
        
        c = group.reindex(pd.date_range('2020-11-11', '2021-09-09', freq=BDay())).drop(columns='Date').reset_index()
        
        df = c[~c['index'].isin(holidays)].rename(columns={'index':'Date'})
        
        df['Stock'] = name
        
        leads = leads.append(df, ignore_index = True)
    
    tesla = leads[leads['Stock'] == 'Tesla']
    
    # interpolate data to fill gaps
    tesla.index = tesla['Date']
    tesla['User_count'].interpolate(method='time', inplace=True)
    tesla = tesla.reset_index(drop=True)
    
    
    #ticker = tesla['Stock'].iloc[0]
    end = tesla['Date'].iloc[-1]
    start = tesla['Date'].iloc[0]
    
    yf.pdr_override()
    index = web.get_data_yahoo('TSLA', start=start, end=end)
    index = index.reset_index()
    
    df = index.merge(tesla, on=['Date'], how='left')
    
    
    
    model_df = df[['Date', 'Open', 'Adj Close', 'Volume', 'User_count']]
    
    training_dataset = model_df[model_df['Date'] < datetime(2021, 7, 20)]
    test_data = model_df[model_df['Date'] >= datetime(2021, 7, 20)]
    
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np
    
    ## Using normalisation as will be using sigmoid function as activation functionn of output layer
    sc = MinMaxScaler()
    training_data = sc.fit_transform(training_dataset.drop(['Date'], axis=1))
    training_data.shape[0]
    
    window = 30
    
    x_train = [training_data[i-window:i] for i in range(60, training_data.shape[0])]
    
    # Open stock price
    y_train = [training_data[i, 0] for i in range(60, training_data.shape[0])]
    
    x_train, y_train = np.array(x_train ),  np.array(y_train)
    
    x_train.shape, y_train.shape
    
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Dropout
    
    ## Model architecture
    model = Sequential()
    
    ## Chose 50 nodes for high dimensionality
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    
    # Dropout Regularisation to pervent overfitting. 20% is a common choice
    model.add(Dropout(0.2))
    
    # Layers 2 - 3
    for x in range(2,4):
        print(f'Initalise layer {x}')
        model.add(LSTM(50, return_sequences=True))
        model.add(Dropout(0.2))
    
    # Final layer
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    
    # Output layer
    model.add(Dense(1))
    
    model.compile(loss="mean_squared_error", optimizer="adam") # Try RMWprop optimizer after
    
    model.summary()
    
    ## 32 recommended batch size
    model.fit(
        x_train, y_train, epochs=100, batch_size=32, verbose=1, validation_split=0.2 #, validation_data=(Xtest, ytest)
    ) # Loss progressively got better (lower)
    
    ## https://machinelearningmastery.com/how-to-use-the-timeseriesgenerator-for-time-series-forecasting-in-keras/
    
    ## Predictions ##
    
    #stock_test_data = model_df[model_df.index >= len(training_data)]
    #dataset = model_df.drop(['Date'], axis=1)
    
    # Adding last window days of training set to test set for LSTM
    total_test_data = pd.concat((training_dataset.tail(window), test_data), ignore_index = True).drop(['Date'], axis=1)
    
    scaled_test_data = sc.transform(total_test_data)
    
    #stock_test_data = test_data.drop(['Date'], axis=1)
    
    x_test = [scaled_test_data[i-window:i] for i in range(window, scaled_test_data.shape[0])]
    y_test = [scaled_test_data[i, 0] for i in range(window, scaled_test_data.shape[0])]
    
    x_test, y_test = np.array(x_test),  np.array(y_test)
    
    x_test.shape, y_test.shape
    
    y_pred = model.predict(x_test)
    
    ## How to use inverse_transform in MinMaxScaler for a column in a matrix
    ## https://stackoverflow.com/questions/49330195/how-to-use-inverse-transform-in-minmaxscaler-for-a-column-in-a-matrix
    # invert predictions
    # Original scaler variable (sc) won't work as it expects a 2D array instead of the 1D y_pred array we are trying to parse.
    scale = MinMaxScaler()
    scale.min_, scale.scale_ = sc.min_[0], sc.scale_[0]
    y_pred = scale.inverse_transform(y_pred)
    y_test = test_data['Open']
    
    y_pred = [x[0] for x in y_pred.tolist()]
    
    test_dates = pd.Series([training_dataset['Date'].iloc[-1]]).append(test_data['Date'], ignore_index=True)
    y_pred_graph =  [training_dataset['Open'].iloc[-1]] + y_pred
    y_test_graph = pd.Series([training_dataset['Open'].iloc[-1]]).append(y_test, ignore_index=True).tolist()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=training_dataset['Date'], y=training_dataset['Open'], name='Past Stock Price'))
    fig.add_trace(go.Scatter(x=test_dates, y=y_pred_graph, name='Predicted Stock Price'))
    fig.add_trace(go.Scatter(x=test_dates, y=y_test_graph, name='Actual Stock Price'))
    fig.update_layout(title="Predicted vs Actual Stock Price", xaxis_title="Date", yaxis_title="Opening Price")
    
    return fig


trades = pd.read_sql_table("trades", con=engine, index_col='index', parse_dates=['Trading day'])

#base = trades.groupby(['Ticker Symbol', 'Type']).count().reset_index() # Number of executions for a given day

ace = trades.groupby(['Ticker Symbol']).sum().reset_index() # Sum of result
base = trades.groupby(['Ticker Symbol']).count().reset_index() # Number of executions for a given day

base['Count'] = base['Result']
    
df = ace.merge(base[['Ticker Symbol', 'Count']], on=['Ticker Symbol'])

df = df.merge(trades[['Ticker Symbol', 'Sector', 'Industry']], on=['Ticker Symbol'])  
    
df.drop_duplicates(inplace=True)



# aq = pd.DataFrame(columns=['ROE(%)','Beta'])
# bad_tickers = []

# holdings = pd.read_sql_table("charts", con=engine, index_col='index')

# for i in holdings['YF_TICKER']:
    
#     stock = yf.Ticker(i)
#     try:
#         ROE = stock.financials.loc['Net Income']/stock.balance_sheet.loc['Total Stockholder Equity']*100
#         Mean_ROE = pd.Series(ROE.mean())
#         Beta = pd.Series(stock.get_info()['beta'])

#         values_to_add = {'ROE(%)': Mean_ROE.values[0].round(2), 'Beta': Beta.values[0].round(2)}
#         row_to_add = pd.Series(values_to_add, name=i)
#         aq = aq.append(row_to_add)
#         print('Downloaded:',i)
#     except:
#         bad_tickers.append(i)

fig = px.scatter(df, y="Result", x="Count", hover_data=['Ticker Symbol'], log_y=True)
plot(fig)

fig = go.Figure()

for col in df[['Count', 'Result', 'Total cost']]:
  fig.add_trace(go.Box(y=df[col].values, name=df[col].name))
  
plot(fig)

df[['Shares', 'Count', 'Result', 'Total cost']].describe()

for x in ['Shares', 'Count', 'Result', 'Total cost']:

    Q1 = df[x].quantile(0.25)
    Q3 = df[x].quantile(0.75)
    
    IQR = Q3 - Q1
    
    lower_range = Q1 - 1.5 * IQR
    upper_range = Q3 + 1.5 * IQR
    
    print(df[(df[x]<lower_range) | (df[x]>upper_range)])

# Outliers
df = df[df['Ticker Symbol'] != '3LTS']
df = df[df['Ticker Symbol'] != 'TSLA']

copy = df[['Count', 'Result', 'Total cost', 'Shares']].copy()
copy.index = df['Ticker Symbol'].copy()

# scaling the data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_values = scaler.fit_transform(copy.values)

# printing pre-processed data
print(df_values)


from sklearn.cluster import KMeans
km_model = KMeans(n_clusters=2).fit(df_values)

clusters = km_model.labels_

copy['cluster'] = clusters
copy['cluster'] = copy['cluster'].astype(str)
copy = copy.reset_index()
copy

fig = px.scatter(copy, y="Result", x="Count", color="cluster", hover_data=['Ticker Symbol'], log_y=True)
plot(fig)


# calculating inertia (Sum squared error) for k-means models with different values of 'k'
inertia = []
k_range = range(1,10)
for k in k_range:
    model = KMeans(n_clusters=k)
    model.fit(df[['Count', 'Result', 'Total cost', 'Shares']])
    inertia.append(model.inertia_)
    print(inertia)
    
# plotting the 'elbow curve'
# plt.figure(figsize=(15,5))
# plt.xlabel('k value',fontsize='x-large')
# plt.ylabel('Model inertia',fontsize='x-large')
# plt.plot(k_range,inertia,color='r')
# plt.show()

# plotting the 'elbow curve'
# Used to determine the number of clusters to use
fig = px.line(x=k_range, y=inertia)
plot(fig)
    
    
# trades = pd.read_sql_table("trades", con=engine, index_col='index', parse_dates=['Trading day'])
# holdings = pd.read_sql_table("holdings", con=engine, index_col='index')

# lis = trades['Ticker Symbol'].value_counts().reset_index().rename(columns={'Ticker Symbol':'Executions', 'index':'Ticker Symbol' })

# holdings = holdings.merge(lis, on='Ticker Symbol', how='left')

# fig = px.scatter(holdings, y="Gains", x="Executions", hover_data=['Ticker Symbol'])
# plot(fig)

# holdings = holdings[['Gross Returns', 'Executions']]
# holdings = holdings.dropna() # Cannot provide missing values to kmeans

# df = holdings.copy()

# # scaling the data
# from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler()
# df_values = scaler.fit_transform(df.values)

# # printing pre-processed data
# print(df_values)


# from sklearn.cluster import KMeans
# km_model = KMeans(n_clusters=3).fit(df_values)

# clusters = km_model.labels_

# df['cluster'] = clusters
# df['cluster'] = df['cluster'].astype(str)
# df = df.reset_index().rename(columns={'index':'Ticker'})
# df

# fig = px.scatter(df, y="Gross Returns", x="Executions", color="cluster", hover_data=['Ticker'])
# plot(fig)
