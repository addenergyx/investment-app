# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 11:48:24 2021

@author: david
"""

import os
import pandas as pd
from dotenv import load_dotenv
from pandas_datareader import data as web
import yfinance as yf
from datetime import datetime, timedelta, time
import plotly.express as px
from plotly.offline import plot
import plotly.graph_objects as go
from pytrends.request import TrendReq
from sqlalchemy import create_engine
#from pytrends import dailydata
from helpers import get_buy_sell, get_yf_symbol, time_frame_returns
#from forex_python.converter import CurrencyRates
from plotly.subplots import make_subplots
from iexfinance.stocks import Stock
#from scraper import getPremarketChange, get_driver
#from fake_useragent import UserAgent
#import time as t
import calendar
import numpy as np

load_dotenv(verbose=True, override=True)

db_URI = os.getenv('AWS_DATABASE_URL')
engine = create_engine(db_URI)
#c = CurrencyRates()

# driver = get_driver(#headless=True
#                     )
#driver.implicitly_wait(10)
# driver.close()
# driver.quit()

def current_price(r):
    print(r['YF_TICKER'])
    if time(hour=9, minute=0) < datetime.now().time() < time(hour=14, minute=30) or time(hour=21) < datetime.now().time() < time(hour=22):
        if r['YF_TICKER'].find('.') == -1:
            try:
                # Use IEX, only works with US (NYSE) Stocks
                tickr = Stock(r['YF_TICKER']) 
                #tickr = Stock('TSLA')
                data = tickr.get_quote()
                
                if data['primaryExchange'].values[0].find('NASDAQ') == 0:
                    r['CURRENT_PRICE'] = getPremarketChange(r['YF_TICKER'], driver)
                else:             
                    # a = tickr.get_price_target()
                    # b = tickr.get_estimates()
                    latestPrice = data['latestPrice'].values[0]
                    extendedPrice = data['extendedPrice'].values[0]
                    #iexRealtimePrice = data['iexRealtimePrice'].values[0]
                    
                    # Only works with non nasdaq stocks due to new regulations 
                    #https://intercom.help/iexcloud/en/articles/3210401-how-do-i-get-nasdaq-listed-stock-data-utp-data-on-iex-cloud
                    r['CURRENT_PRICE'] = latestPrice if extendedPrice is None else extendedPrice
                return r

            except: 
                try:
                    r['CURRENT_PRICE'] = yf.download(tickers=r['YF_TICKER'], period='1m', progress=False)['Close'].values[0]
                    return r
                except:
                    r['CURRENT_PRICE'] = float('NaN')
                    return r
        else:
            try:
                r['CURRENT_PRICE'] = yf.download(tickers=r['YF_TICKER'], period='1m', progress=False)['Close'].values[0]
                return r
            except:
                r['CURRENT_PRICE'] = float('NaN')
                return r
    else:
        try:
            r['CURRENT_PRICE'] = yf.download(tickers=r['YF_TICKER'], period='1m', progress=False)['Close'].values[0]
        except:
            r['CURRENT_PRICE'] = float('NaN')
        return r

def get_holdings():
    holdings = pd.read_sql_table("portfolio", con=engine, index_col='index')
    
    # driver = get_driver(#headless=True
    #     )
    # driver.implicitly_wait(20)
    
    # holdings = holdings.apply(current_price, axis=1)
    # driver.close()
    # driver.quit()
    
    # holdings.dropna(axis=0, inplace=True)
    
    # Recent ticker change due to merger, Yahoo finance pulls wrong data, should be fixed later
    #holdings = holdings[holdings['Ticker'] != 'UWMC']
    
    holdings['PREV_CLOSE'] = holdings['PREV_CLOSE'].astype('float')
    #print('got holdings')
    return holdings

# holdings = get_holdings()
# holdings = holdings.apply(current_price, axis=1)
# holdings['CURRENT MARKET VALUE'] = ''
# holdings['OG MARKET VALUE'] = ''

def convert_to_gbp(r):
    tickr = yf.Ticker(r['YF_TICKER'])
    #tickr = yf.Ticker('TILS.L')
    # a = aapl.recommendations
    # b = aapl.calendar
    
    try:
        currency = tickr.info['currency']
    except:
        print('----------ERROR-------------')
        """
        Some need to be set manually, for example 3LTS is a leveraged ETF traded 
        on the LSE but uses $ instead of £
        """
        if r['YF_TICKER'] == '3LTS.L':
            currency = 'USD'
        elif r['YF_TICKER'].find('.L') != -1:
            currency = 'GBp' # Most UK stocks are in pence
        
    print(r['YF_TICKER'])
    print(currency)
        
    if currency == 'GBp':
        currency = 'GBP'
        r['CURRENT MARKET VALUE'] = (float('{:.2f}'.format(c.get_rate(currency, 'GBP'))) * r['CURRENT PRICE'] * r['QUANTITY']) / 100
        r['OG MARKET VALUE'] = (float('{:.2f}'.format(c.get_rate(currency, 'GBP'))) * r['PRICE'] * r['QUANTITY']) / 100
        # r['CURRENT MARKET VALUE'] = c.convert(currency, 'GBP', r['CURRENT PRICE']*r['QUANTITY']) / 100
        # r['OG MARKET VALUE'] = c.convert(currency, 'GBP', r['PRICE']*r['QUANTITY']) / 100
    else:
        #print(c.convert(currency, 'GBP', r['CURRENT PRICE']*r['QUANTITY']))
        r['CURRENT MARKET VALUE'] = (float('{:.2f}'.format(c.get_rate(currency, 'GBP'))) * r['CURRENT PRICE'] * r['QUANTITY'])
        r['OG MARKET VALUE'] = (float('{:.2f}'.format(c.get_rate(currency, 'GBP'))) * r['PRICE'] * r['QUANTITY'])
        # r['CURRENT MARKET VALUE'] = c.convert(currency, 'GBP', r['CURRENT PRICE']*r['QUANTITY'])
        # r['OG MARKET VALUE'] = c.convert(currency, 'GBP', r['PRICE']*r['QUANTITY'])
    return r

#c.convert('USD', 'GBP', 117.5*18)
#float('{:.2f}'.format(c.get_rate('GBP', 'GBP')))

# holdings = holdings.apply(convert_to_gbp, axis=1)

# floating_total = sum(holdings['CURRENT MARKET VALUE']) - sum(holdings['OG MARKET VALUE'])

# holdings['diff'] = holdings['CURRENT MARKET VALUE'] - holdings['OG MARKET VALUE']

# floating_profit = sum(holdings['diff'][holdings['diff'] > 0])
# floating_loss = sum(holdings['diff'][holdings['diff'] < 0])

def fig_layout(fig):
    fig.update_layout(#margin=dict(l=0, r=0, t=0, b=0),
                        #paper_bgcolor='rgba(0,0,0,0)',
                        #plot_bgcolor='rgba(0,0,0,0)',
                        hovermode="x unified",
                      )
    return fig


def day_treemap(colour='RdBu'):
    # 1 Day Performance
        
    holdings = pd.read_sql_table("charts", con=engine, index_col='index')
    
    fig = px.treemap(holdings, path=['Sector', 'Industry', 'Ticker'], values='CAPITAL', color='PCT',
                      color_continuous_scale=colour, color_continuous_midpoint=0, range_color=[-15,15], 
                      #hover_data=['Ticker', 'MARKET VALUE', 'PCT']
                      )
        
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        transition={
                            'duration': 500,
                            'easing': 'cubic-in-out',
                        }
                      )
    fig.update_layout(coloraxis_showscale=False)
    fig.data[0].hovertemplate = '%{label}<br>%{color}%<br>£%{value}'
    #plot(fig)
    #fig.data[0].textinfo = 'label+text+percent entry+percent parent+value'
    print('launch day map')
    return fig

# import time as t
# start = t.time()
# day_treemap()
# end = t.time()
# print(end - start)
#look at immr

def return_treemap(colour='RdBu'):
    
    holdings = pd.read_sql_table("charts", con=engine, index_col='index')
    
    fig = px.treemap(holdings, path=['Sector', 'Industry', 'Ticker'], values='CAPITAL', color='Ri',
                     color_continuous_scale=colour, color_continuous_midpoint=0, range_color=[-50,50])
    
    fig.update_layout(    
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    #https://plotly.com/python/colorscales/#setting-the-midpoint-of-a-color-range-for-a-diverging-color-scale
    fig.update_layout(coloraxis_showscale=False)
    fig.data[0].hovertemplate = '%{label}<br>%{color}%<br>£%{value}'
    print('launch returns map')
    # plot(fig)
    return fig

def chart(ticker):
    
    all_212_equities = pd.read_sql_table("equities", con=engine, index_col='index')

    market = all_212_equities[all_212_equities['INSTRUMENT'] == ticker]['MARKET NAME'].values[0] 
    
    buys, sells = get_buy_sell(ticker)
    
    start = datetime(2020, 2, 7)
    end = datetime.now()    
        
    yf_symbol = get_yf_symbol(market, ticker)   
    
    index = web.DataReader(yf_symbol, 'yahoo', start, end)
    index = index.reset_index()
    
    averages_df = averages[averages['Ticker Symbol'] == ticker]
    averages_df['ISIN'] = all_holdings[all_holdings['Ticker Symbol'] == ticker]['ISIN'].values[0]
    averages_df = averages_df.apply(avg_stock_split_adjustment, axis=1)

    # ## TODO: Allow user to switch between line and candlestick chart

    # # Add traces
    # fig.add_trace(go.Scatter(x=index['Date'], y=index['Adj Close'], 
    #                     mode='lines'))
    
    # # Buys
    # fig.add_trace(go.Scatter(x=buys['Trading day'], y=buys['dolla'],
    #                     mode='markers',
    #                     name='Buy point'
    #                     ))
    # # Sells
    # fig.add_trace(go.Scatter(x=sells['Trading day'], y=sells['dolla'],
    #                     mode='markers',
    #                     name='Sell point'
    #                     ))
    
    ## Candlestick Graph
        
    fig = go.Figure(data=[go.Candlestick(x=index['Date'],
                    open=index['Open'],
                    high=index['High'],
                    low=index['Low'],
                    close=index['Adj Close'],
                    name='Stock')])
    
    # Buys
    fig.add_trace(go.Scatter(x=sells['Trading day'], y=sells['dolla'],
                        mode='markers',
                        name='Sell point',
                        #marker=dict(color='#ff7f0e')
                        marker=dict(size=7,
                                    line=dict(width=2,
                                              color='DarkSlateGrey')),
                        ))
    
    # Sells
    fig.add_trace(go.Scatter(x=buys['Trading day'], y=buys['dolla'],
                        mode='markers',
                        name='Buy point',
                        #marker=dict(color='#1f77b4')
                        marker=dict(size=7,
                                    line=dict(width=2,
                                              color='DarkSlateGrey')),
                        ))
    
    # shapes = list()
    # for i in (20, 40, 60):
    #     shapes.append({'type': 'line',
    #                    'xref': 'x',
    #                    'yref': 'y',
    #                    'x0': ,
    #                    'y0': 0,
    #                    'x1': i,
    #                    'y1': 1})
    
    def hlines(r):
        fig.add_hline(y=r['Average']/r['Exchange rate'], line_width=3, line_dash="dash")

    averages_df[-5:-1].apply(hlines, axis=1)
    
    if index.iloc[-1]['Adj Close'] > averages_df.iloc[-1]['Average']/averages_df.iloc[-1]['Exchange rate']:
        fig.add_hline(y=averages_df.iloc[-1]['Average']/averages_df.iloc[-1]['Exchange rate'], line_width=3, line_dash="dash", line_color="green")
    else:
        fig.add_hline(y=averages_df.iloc[-1]['Average']/averages_df.iloc[-1]['Exchange rate'], line_width=3, line_dash="dash", line_color="red")

    fig.update_layout(hovermode="x unified", title=f'{ticker} Buy/Sell points') # Currently plotly doesn't support hover for overlapping points in same trace
    
    return fig

def performance_chart(ticker='TSLA'):

    #ticker = 'XOS'
    
    all_212_equities = pd.read_sql_table("equities", con=engine, index_col='index')
    
    try:
        market = all_212_equities[all_212_equities['INSTRUMENT'] == ticker]['MARKET NAME'].values[0] 
        yf_symbol = get_yf_symbol(market, ticker)   
    except:
        print("Can't find ticker")
        yf_symbol = ticker
    
    start = datetime(2020, 2, 7)
    end = datetime.now()    
    
    #index = web.DataReader(yf_symbol, start, end)
    
    yf.pdr_override()
    index = web.get_data_yahoo(yf_symbol, start=start, end=end)
    
    index = index.reset_index()
    
    # cache tesla data because function takes too long
    # Heroku has a 30sec timeout
    
    portfolio = pd.read_sql_table("trades", con=engine, index_col='index', parse_dates=['Trading day'])
                
    if len(portfolio[portfolio['Ticker Symbol'] == ticker]) > 150: 
        
        # data = pd.read_csv(f'cached_data/{ticker}.csv')
        data = pd.read_sql_table(f'{ticker}', con=engine, index_col='index')
        
        buys = data[data['Type']=='Buy']
        sells = data[data['Type']=='Sell']
        
        # a = buys.append(sells)
        # a.to_csv(f'cached_data/{ticker}.csv')
        # a.to_sql(f'{ticker}', engine, if_exists='replace')
        
    else:
    
        buys, sells = get_buy_sell(ticker) 
        
        index['Midpoint'] = (index['High'] + index['Low']) / 2
        
        buy_target = []
        sell_target = []
        
        for i, row in buys.iterrows():
            
            try:
                mid = index[index['Date'] == row['Trading day']]['Midpoint'].values[0]
                
                if row['Execution_Price'] < mid:
                    buy_target.append(1)
                else:
                    buy_target.append(0)
            except:   
                # Missing data from yahoo finance
                print(f'Missing data {row}')
        
        for i, row in sells.iterrows():
            
            try:
                mid = index[index['Date'] == row['Trading day']]['Midpoint'].values[0]
                
                if row['Execution_Price'] > mid:
                    sell_target.append(1)
                else:
                    sell_target.append(0)
            except:
                print(f'Missing data {row}')
    
        
        buys['Target'] = buy_target
        sells['Target'] = sell_target

    ## Discrete color graph
    
    main_fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig = go.Figure(data=[go.Candlestick(x=index['Date'],
                    open=index['Open'],
                    high=index['High'],
                    low=index['Low'],
                    close=index['Adj Close'],
                    name='Stock')])
    
    # fig = go.Candlestick(x=index['Date'],
    #             open=index['Open'],
    #             high=index['High'],
    #             low=index['Low'],
    #             close=index['Adj Close'],
    #             name='Stock')
    
    # Must be a string for plotly to interpret numeric values as a discrete value
    # https://plotly.com/python/discrete-color/
    sells['Target'] = sells['Target'].astype(str)
    buys['Target'] = buys['Target'].astype(str)
        
    if len(sells) > 0:
        
        fig1 = px.scatter(sells, x='Trading day', y='Execution_Price', color='Target')
    
        for x in fig1.data:
            if x['legendgroup'] == '1':
                x['marker'] =  {'color':'#E24C4F', 'line': {'color': 'yellow', 'width': 2}, 'size': 7, 'symbol': 'circle'}
                x['name'] = 'Successful Sell Point'
                fig.add_trace(x)
            elif x['legendgroup'] == '0':
                x['marker'] =  {'color':'#E24C4F', 'line': {'color': 'black', 'width': 2}, 'size': 7, 'symbol': 'circle'}
                x['name'] = 'Unsuccessful Sell Point'
                fig.add_trace(x)
    
    if len(buys) > 0:

        fig2 = px.scatter(buys, x='Trading day', y='Execution_Price', color='Target')
        #fig2.update_traces(marker=dict(color='blue'))
        #fig2.update_traces(marker=dict(color='#30C296', size=7, line=dict(width=2, color='DarkSlateGrey')))
        
        for x in fig2.data:
            if x['legendgroup'] == '1':
                x['marker'] =  {'color':'#3D9970', 'line': {'color': 'yellow', 'width': 2}, 'size': 7, 'symbol': 'circle'}
                x['name'] = 'Successful Buy Point'
                fig.add_trace(x)
            elif x['legendgroup'] == '0':
                x['marker'] =  {'color':'#3D9970','line': {'color': 'black', 'width': 2}, 'size': 7, 'symbol': 'circle'}
                x['name'] = 'Unsuccessful Buy Point'
                fig.add_trace(x)
    
    for x in range(len(fig.data)):
        main_fig.add_trace(fig.data[x], secondary_y=True)
    
    # include a go.Bar trace for volumes
    volume_fig = go.Figure(go.Bar(x=index['Date'], y=index['Volume'], name='Volume'))
    volume_fig.update_traces(marker_color='rgb(158,202,225)', opacity=0.6)
    
    main_fig.add_trace(volume_fig.data[0], secondary_y=False)
    
    main_fig.update_layout(hovermode="x unified", title=f'{ticker} Stock Graph', 
                    #   legend=dict(
                    #         yanchor="top",
                    #         y=0.99,
                    #         xanchor="left",
                    #         x=0.01
                    # )
                    # showlegend=False
                    )
    
    main_fig.layout.yaxis1.showgrid=False

    #plot(main_fig)
    
    # count = buys['Target'].value_counts().add(sells['Target'].value_counts(),fill_value=0)
    # percentage = count[1]/count.sum() *100
    # percentage = '{:.2f}'.format(percentage)
    
    return fig_layout(main_fig) #, percentage

# Monthly Returns and targets
def goal_chart():
    
    summary_df = pd.read_sql_table("summary", con=engine, index_col='index')
    
    fig = go.Figure(data=[
        go.Bar(name='Return', x=summary_df['Date'], y=summary_df['Returns']),
        go.Scatter(name='Goal', x=summary_df['Date'], y=summary_df['Goal']),
    ])
    
    # Change the bar mode
    fig.update_layout(barmode='overlay', title='Monthly Returns and Targets')
    
    return fig_layout(fig)

# Cumsum
def cumsum_chart():
    monthly_returns_df = pd.read_sql_table("summary", con=engine, index_col='index')
    monthly_returns_df['Rolling Returns'] = monthly_returns_df['Returns'].cumsum()
    fig = px.bar(monthly_returns_df, x='Date', y='Rolling Returns', title='Rolling Realised Returns')
    return fig_layout(fig)

def dividend_chart():
    # Dividends
    summary_df = pd.read_sql_table("summary", con=engine, index_col='index')
    fig = px.bar(summary_df, x='Date', y='Dividends', color='Date', title='Dividends')
    return fig_layout(fig)

def period_chart(time='M'):
    
    #time='D'
    
    timeframe_returns_df = time_frame_returns(time)
    timeframe_returns_df.reset_index(level=0, inplace=True)

    if time == 'W':
        timeframe_returns_df.Date = timeframe_returns_df.Date.astype(str) # Change type period to string
        timeframe_returns_df['Date'] = timeframe_returns_df['Date'].str.split('/', 1).str[1] # Week ending
        timeframe_returns_df['Date'] = pd.to_datetime(timeframe_returns_df['Date']) + timedelta(days=-2) # Last working day of week

    if time == 'M' or time == 'Q':
        timeframe_returns_df['Date'] = timeframe_returns_df['Date'].dt.strftime('%Y-%m')
    elif time == 'A-APR'or time == 'Y':
        timeframe_returns_df['Date'] = timeframe_returns_df['Date'].dt.strftime('%Y')
    else:
        timeframe_returns_df['Date'] = timeframe_returns_df['Date'].dt.strftime('%d-%m-%Y')

    fig = px.bar(timeframe_returns_df, x='Date', y='Returns', color='Date', title='Returns')
    #fig = px.line(timeframe_returns_df, x='Date', y='Returns', title='Returns')
    # fig = go.Figure(go.Scatter(x=timeframe_returns_df['Date'], y=timeframe_returns_df['Returns'],
    #                     mode='lines',
    #                     name='Returns',
    #                     line = {'color':'#6502C0', 'shape': 'spline', 'smoothing': 1},
    #                     fill='tozeroy',
    #                     ))
    
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        hovermode='x',
                        xaxis={    
                            'showgrid': False, # thin lines in the background
                            'zeroline': False, # thick line at x=0
                            #'visible': False,  # numbers below
                            #'tickmode':'linear',
                        },                                                
                        yaxis={    
                            'showgrid': False,
                            'zeroline': False,
                            #'visible': False,
                        },
                      )
    #plot(fig)

    return fig_layout(fig)

#period_chart()

def profit_loss_chart():
    # P/L
    monthly_returns_df = pd.read_sql_table("summary", con=engine, index_col='index')
    fig = go.Figure()
    fig.add_trace(go.Bar(x=monthly_returns_df['Date'], y=monthly_returns_df['Gains'],
                    marker_color='green',
                    name='Gains'))
    fig.add_trace(go.Bar(x=monthly_returns_df['Date'], y=monthly_returns_df['Losses'],
                    base=0,
                    marker_color='crimson',
                    name='Losses'
                    ))
    fig.update_layout(barmode='overlay')
    return fig_layout(fig)

# # Daily Returns
# fig = px.bar(daily_returns_df, x='Date', y='Returns', color='Date', title='Daily Returns')
# plot(fig)

# ## TODO: On click show all trades that day: daily_returns_df[daily_returns_df['Date'] == day_clicked]

# # Buy/Sell
# counts = trades['Type'].value_counts()       
# counts_df = counts.reset_index()
# counts_df.columns = ['Type', 'Count']
# fig = px.pie(counts_df, values='Count', names='Type')
# plot(fig)

# ## Stock activity - How many times I've bought/sold a stock         
# stocks = trades['Ticker Symbol'].value_counts()         
# stocks = stocks.reset_index()
# stocks.columns = ['Ticker Symbol', 'Count']           
# fig = px.pie(stocks, values='Count', names='Ticker Symbol', title='Portfolio Trading Activity')
# plot(fig)

def avg_stock_split_adjustment(r):
        
    market = get_market(r['ISIN'], r['Ticker Symbol'])[1] 
    
    ticker = get_yf_symbol(market, r['Ticker Symbol'])
    
    aapl = yf.Ticker(ticker)
    split_df = aapl.splits.reset_index()
    split = split_df[split_df['Date'] > r['Trading day']]['Stock Splits'].sum()
    
    if split > 0:
        r.Average = r.Average/split
    
    return r


# def ml_model():
#     pytrend = TrendReq()

#     keyword = 'tesla Stock'
    
#     end = datetime.now()
#     start = datetime(end.year - 5, end.month, end.day)
    
#     ss = start.strftime('%Y-%m-%d')
#     ee = end.strftime('%Y-%m-%d')
    
#     # 1 year follows price trend better than 5 year 
#     # This  may be because the values are calculated on a scale from 0 to 100, 
#     # where 100 is the timeframe with the most popularity as a fraction of total searches in the given period of time, 
#     # a value of 50 indicates a time which is half as popular. 
#     # A value of 0 indicates a location where there was not enough data for this term. 
#     # Source →Google Trends.
    
#     # For my hypothesis I feel 1 year is more accurate due to influx of new traders due to corona
#     # Old school traders rely on fundementals/technicals whereas newer trader trade on sentiment and momentum
    
#     pytrend.build_payload(kw_list=[keyword], timeframe=f'{ss} {ee}')
#     df = pytrend.interest_over_time() # Weekly data
#     df.reset_index(level=0, inplace=True)
    
#     df2 = dailydata.get_daily_data(keyword, start.year, start.month, end.year, end.month)
#     df2.reset_index(level=0, inplace=True)
#     df2 = df2.rename(columns={'date':'Date'})
    
#     index = web.DataReader('tsla', 'yahoo', start, end)
#     index = index.reset_index()

#     merged_df = pd.merge(index, df2, on="Date")

#     model_df = merged_df[['Date', 'Open', 'Adj Close', 'Volume', merged_df.filter(regex='_unscaled$').columns[0]]]

#     training_dataset = model_df[model_df['Date'] < datetime(2020, 11, 1)]
#     test_data = model_df[model_df['Date'] >= datetime(2020, 11, 1)]
    
#     from sklearn.preprocessing import MinMaxScaler
#     import numpy as np
    
#     ## Using normalisation as will be using sigmoid function as activation functionn of output layer
#     sc = MinMaxScaler()
#     training_data = sc.fit_transform(training_dataset.drop(['Date'], axis=1))
#     training_data.shape[0]
    
#     window = 30
    
#     x_train = [training_data[i-window:i] for i in range(60, training_data.shape[0])]
    
#     # Open stock price
#     y_train = [training_data[i, 0] for i in range(60, training_data.shape[0])]
    
#     x_train, y_train = np.array(x_train ),  np.array(y_train)
    
#     x_train.shape, y_train.shape
    
#     from keras.models import Sequential
#     from keras.layers import LSTM, Dense, Dropout
    
#     ## Model architecture
#     model = Sequential()
    
#     ## Chose 50 nodes for high dimensionality
#     model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    
#     # Dropout Regularisation to pervent overfitting. 20% is a common choice
#     model.add(Dropout(0.2))
    
#     # Layers 2 - 3
#     for x in range(2,4):
#         print(f'Initalise layer {x}')
#         model.add(LSTM(50, return_sequences=True))
#         model.add(Dropout(0.2))
    
#     # Final layer
#     model.add(LSTM(50))
#     model.add(Dropout(0.2))
    
#     # Output layer
#     model.add(Dense(1))
    
#     model.compile(loss="mean_squared_error", optimizer="adam") # Try RMWprop optimizer after
    
#     model.summary()
    
#     ## 32 recommended batch size
#     model.fit(
#         x_train, y_train, epochs=100, batch_size=32, verbose=1, validation_split=0.2 #, validation_data=(Xtest, ytest)
#     ) # Loss progressively got better (lower)
    
#     ## https://machinelearningmastery.com/how-to-use-the-timeseriesgenerator-for-time-series-forecasting-in-keras/
    
#     ## Predictions ##
    
#     #stock_test_data = model_df[model_df.index >= len(training_data)]
#     #dataset = model_df.drop(['Date'], axis=1)
    
#     # Adding last window days of training set to test set for LSTM
#     total_test_data = pd.concat((training_dataset.tail(window), test_data), ignore_index = True).drop(['Date'], axis=1)
    
#     scaled_test_data = sc.transform(total_test_data)
    
#     #stock_test_data = test_data.drop(['Date'], axis=1)
    
#     x_test = [scaled_test_data[i-window:i] for i in range(window, scaled_test_data.shape[0])]
#     y_test = [scaled_test_data[i, 0] for i in range(window, scaled_test_data.shape[0])]
    
#     x_test, y_test = np.array(x_test),  np.array(y_test)
    
#     x_test.shape, y_test.shape
    
#     y_pred = model.predict(x_test)
    
#     ## How to use inverse_transform in MinMaxScaler for a column in a matrix
#     ## https://stackoverflow.com/questions/49330195/how-to-use-inverse-transform-in-minmaxscaler-for-a-column-in-a-matrix
#     # invert predictions
#     # Original scaler variable (sc) won't work as it expects a 2D array instead of the 1D y_pred array we are trying to parse.
#     scale = MinMaxScaler()
#     scale.min_, scale.scale_ = sc.min_[0], sc.scale_[0]
#     y_pred = scale.inverse_transform(y_pred)
#     y_test = test_data['Open']
    
#     y_pred  = [x[0] for x in y_pred.tolist()]
    
#     test_dates = pd.Series([training_dataset['Date'].iloc[-1]]).append(test_data['Date'], ignore_index=True)
#     y_pred_graph =  [training_dataset['Open'].iloc[-1]] + y_pred
#     y_test_graph = pd.Series([training_dataset['Open'].iloc[-1]]).append(y_test, ignore_index=True).tolist()
    
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=training_dataset['Date'], y=training_dataset['Open'], name='Past Stock Price'))
#     fig.add_trace(go.Scatter(x=test_dates, y=y_pred_graph, name='Predicted Stock Price'))
#     fig.add_trace(go.Scatter(x=test_dates, y=y_test_graph, name='Actual Stock Price'))
#     fig.update_layout(title="Predicted vs Actual Stock Price", xaxis_title="Date", yaxis_title="Opening Price")
    
#     return fig


def trades_chart():

    trades = pd.read_sql_table("trades", con=engine, index_col='index', parse_dates=['Trading day'])
    
    #trades = pd.read_csv("https://raw.githubusercontent.com/addenergyx/datasets/main/trading%20data%20export%20with%20results.csv", parse_dates=['Trading day'], dayfirst=True)
    days = trades.groupby(['Trading day']).size().reset_index().rename(columns={0:'Count'})
    
    buys = trades[trades['Type'] == 'Buy'].groupby(['Trading day']).size().reset_index().rename(columns={0:'Buy Execution'})
    sells = trades[trades['Type'] == 'Sell'].groupby(['Trading day']).size().reset_index().rename(columns={0:'Sell Execution'})
    
    idx = pd.bdate_range(min(days['Trading day']), max(days['Trading day']))
    days.set_index('Trading day', inplace=True)
    buys.set_index('Trading day', inplace=True)
    sells.set_index('Trading day', inplace=True)
    
    days = days.reindex(idx, fill_value=0).reset_index().rename(columns={'index':'Trading day', 'Count':'Trading Activity'})
    buys = buys.reindex(idx, fill_value=0).reset_index().rename(columns={'index':'Trading day'})
    sells = sells.reindex(idx, fill_value=0).reset_index().rename(columns={'index':'Trading day'})
    
    days = days.merge(buys, on=['Trading day'], how='left').merge(sells, on=['Trading day'], how='left')
    
    #df = pd.read_csv("https://raw.githubusercontent.com/addenergyx/datasets/main/day_count.csv")
    # fig = px.bar(days, x='Trading day', y='Trading Activity')
    #plot(fig)
    
    fig = go.Figure(data=[
        go.Bar(name='Buy Executions', x=days['Trading day'], y=days['Buy Execution']),
        go.Bar(name='Sell Executions', x=days['Trading day'], y=days['Sell Execution'])
    ])
    # Change the bar mode
    fig.update_layout(barmode='stack', bargap=0, title='Trading volume over time')
    

    return fig


def vis4(colour='RdBu'):
    
    trades = pd.read_sql_table("trades", con=engine, index_col='index', parse_dates=['Trading day'])
    
    #trades = pd.read_csv("https://raw.githubusercontent.com/addenergyx/datasets/main/trading%20data%20export%20with%20results.csv", parse_dates=['Trading day'])
    
    trades['Result'] = trades['Result'].replace('', 0.0).astype(float)
        
    trades = trades.dropna(thresh=15)
    
    trades = trades.dropna(axis=1, thresh=400)
    
    a = trades.groupby(['Sector','Industry','Name']).count()
    b = trades.groupby(['Sector','Industry','Name']).sum()
    
    a = a.reset_index()
    b = b.reset_index()
    
    a['count'] = a['Result']
    
    a = a[['Sector','Industry', 'Name', 'count']]
    b = b[['Sector','Industry', 'Name','Result']]
    
    c = pd.concat([a, b['Result']], axis=1)
    
    #Weights sum to zero, can't be normalized
    b['Result'] = b['Result'].astype(int)
    b = b[b['Result']!=0]
    
    fig = px.treemap(c, path=['Sector', 'Industry', 'Name'], values='count', color='Result',
                          color_continuous_scale=colour, color_continuous_midpoint=0, range_color=[-1000,1000], 
                          #hover_data=['Ticker', 'MARKET VALUE', 'PCT']
                          )
            
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        transition={
                            'duration': 500,
                            'easing': 'cubic-in-out',
                        }
                      )
    #fig.update_layout(coloraxis_showscale=False)
    fig.data[0].hovertemplate = '%{label}<br><br>%{value} Trades<br>£%{color}'
    
    #fig.data[0].textinfo = 'label+text+percent entry+percent parent+value'
    return fig

# plot(vis4(colour='RdBu'))

# holdings = pd.read_sql_table("charts", con=engine, index_col='index')

# trades = pd.read_sql_table("trades", con=engine, index_col='index', parse_dates=['Trading day'])

# DF = pd.DataFrame(columns=['ROE(%)','Beta'])
# DFc = pd.DataFrame(columns=['ROE(%)','Beta'])
# bad_tickers = []


# for i in holdings['YF_TICKER']:
    
#     stock = yf.Ticker(i)
#     try:
#         ROE = stock.financials.loc['Net Income']/stock.balance_sheet.loc['Total Stockholder Equity']*100
#         Mean_ROE = pd.Series(ROE.mean())
#         Beta = pd.Series(stock.get_info()['beta'])

#         values_to_add = {'ROE(%)': Mean_ROE.values[0].round(2), 'Beta': Beta.values[0].round(2)}
#         row_to_add = pd.Series(values_to_add, name=i)
#         DF = DF.append(row_to_add)
#         print('Downloaded:',i)
#     except:
#         bad_tickers.append(i)


# # making a copy to work with
# df = DF.copy()

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

# fig = px.scatter(df, y="ROE(%)", x="Beta", color="cluster", hover_data=['Ticker'])
# plot(fig)


# # calculating inertia (Sum squared error) for k-means models with different values of 'k'
# inertia = []
# k_range = range(1,10)
# for k in k_range:
#     model = KMeans(n_clusters=k)
#     model.fit(df[["ROE(%)", "Beta"]])
#     inertia.append(model.inertia_)
#     print(inertia)
    
# # plotting the 'elbow curve'
# # plt.figure(figsize=(15,5))
# # plt.xlabel('k value',fontsize='x-large')
# # plt.ylabel('Model inertia',fontsize='x-large')
# # plt.plot(k_range,inertia,color='r')
# # plt.show()

# # plotting the 'elbow curve'
# # Used to determine the number of clusters to use
# fig = px.line(x=k_range, y=inertia)
# plot(fig)
    
    
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

def vis1(filters='Returns'):
    
    seasonality = pd.read_sql_table("trades", con=engine, index_col='index', parse_dates=['Trading day'])

    seasonality['weekday'] = seasonality['Trading day'].apply(lambda x :calendar.day_name[x.weekday()])
    
    seasonality['Trading_time'] = pd.to_timedelta(seasonality['Trading time'])
    seasonality['Trading_time'] = seasonality['Trading_time'].apply(lambda x : int(x.seconds /3600))
    
    s_buys = seasonality[seasonality['Type'] == 'Buy']
    s_sells = seasonality[seasonality['Type'] == 'Sell']
    
    s_buys['Trading_time'] = pd.to_timedelta(s_buys['Trading time'])
    s_buys['Trading_time'] = s_buys['Trading_time'].apply(lambda x : int(x.seconds /3600))
    s_buys = s_buys[['weekday','Trading_time']]
    s_buys = s_buys.groupby(['weekday','Trading_time']).size().reset_index().rename(columns={0:'Buy Count'})
    
    s_sells['Trading_time'] = pd.to_timedelta(s_sells['Trading time'])
    s_sells['Trading_time'] = s_sells['Trading_time'].apply(lambda x : int(x.seconds /3600))
    s_sells = s_sells[['weekday','Trading_time', 'Result']]
    s_sells = s_sells.groupby(['weekday','Trading_time']).agg(['sum','count']).reset_index()
    s_sells.columns = ['weekday', 'Trading_time', 'Returns', 'Sell Count']
    
    seasonality_df = seasonality.groupby(['weekday','Trading_time']).size().reset_index().rename(columns={0:'Count'})
    
    seasonality_df = seasonality_df.merge(s_buys, on=['weekday','Trading_time'], how='left').merge(s_sells, on=['weekday','Trading_time'], how='left')
    
    seasonality_df['pct_buy'] = seasonality_df['Buy Count'] / seasonality_df['Count']
    
    seasonality_df['Trading_time'] = [f'0{x}:00' if x < 10 else f'{x}:00' for x in seasonality_df['Trading_time']]
    
    seasonality_df['Returns'] = seasonality_df['Returns'].fillna(0)
    
    crange = [-100,100] if filters == 'Returns' else [0,1]
    mpoint = 0 if filters == 'Returns' else 0.5
    
    fig = px.scatter(seasonality_df, y="weekday", x="Trading_time",
	         size="Count", color=filters, color_continuous_scale='RdBu', color_continuous_midpoint=mpoint,
             size_max=40, range_color=crange
                 )

    fig.update_yaxes(title='Weekday', categoryorder='array', categoryarray= ['Friday', 'Thursday', 'Wednesday', 'Tuesday', 'Monday'])
    #fig.update_yaxes(categoryorder='array', categoryarray= ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])
    fig.update_xaxes(categoryorder='array', categoryarray= [f'0{x}:00' if x < 10 else f'{x}:00' for x in range(7,21)])
    fig.layout.yaxis.showgrid=False
    fig.layout.xaxis.showgrid=False
    
    return fig

def chart(ticker):

    #all_212_equities = pd.read_csv('all_212_equities.csv')
    
    all_212_equities = pd.read_sql_table("equities", con=engine, index_col='index')

    market = all_212_equities[all_212_equities['INSTRUMENT'] == ticker]['MARKET NAME'].values[0] 
    
    buys, sells = get_buy_sell(ticker)
    
    start = datetime(2020, 2, 7)
    end = datetime.now()    
        
    yf_symbol = get_yf_symbol(market, ticker)   
    
    index = web.DataReader(yf_symbol, 'yahoo', start, end)
    index = index.reset_index()
    
    ## Candlestick Graph
    fig = go.Figure(data=[go.Candlestick(x=index['Date'],
                    open=index['Open'],
                    high=index['High'],
                    low=index['Low'],
                    close=index['Adj Close'],
                    name='Stock')])
    
    # Buys
    fig.add_trace(go.Scatter(x=sells['Trading day'], y=sells['Execution_Price'],
                        mode='markers',
                        name='Sell point',
                        marker_symbol='triangle-down',
                        #marker=dict(color='#ff7f0e')
                        marker=dict(size=12,
                                    line=dict(width=2,
                                              color='DarkSlateGrey')),
                        ))
    
    # Sells
    fig.add_trace(go.Scatter(x=buys['Trading day'], y=buys['Execution_Price'],
                        mode='markers',
                        name='Buy point',
                        marker_symbol='triangle-up',
                        #marker=dict(color='#1f77b4')
                        marker=dict(size=12,
                                    line=dict(width=2,
                                              color='DarkSlateGrey')),
                        ))
    
    fig.update_layout(hovermode="x unified", title=f'{ticker} Buy/Sell points') # Currently plotly doesn't support hover for overlapping points in same trace
    
    plot(fig)
    
chart('AMC')

def line_chart(ticker):

    #all_212_equities = pd.read_csv('all_212_equities.csv')
    
    all_212_equities = pd.read_sql_table("equities", con=engine, index_col='index')

    market = all_212_equities[all_212_equities['INSTRUMENT'] == ticker]['MARKET NAME'].values[0] 
    
    buys, sells = get_buy_sell(ticker)
    
    start = datetime(2020, 2, 7)
    end = datetime.now()    
        
    yf_symbol = get_yf_symbol(market, ticker)   
    
    index = web.DataReader(yf_symbol, 'yahoo', start, end)
    index = index.reset_index()
    
    ## Candlestick Graph
    fig = go.Figure(data=[go.Scatter(x=index['Date'], y=index['Adj Close'], 
                        mode='lines', name='Closing price')])
    
    # Buys
    fig.add_trace(go.Scatter(x=sells['Trading day'], y=sells['Execution_Price'],
                        mode='markers',
                        name='Sell point',
                        #marker=dict(color='#ff7f0e')
                        marker=dict(size=7,
                                    line=dict(width=2,
                                              color='DarkSlateGrey')),
                        ))
    
    # Sells
    fig.add_trace(go.Scatter(x=buys['Trading day'], y=buys['Execution_Price'],
                        mode='markers',
                        name='Buy point',
                        #marker=dict(color='#1f77b4')
                        marker=dict(size=7,
                                    line=dict(width=2,
                                              color='DarkSlateGrey')),
                        ))
    
    fig.update_layout(hovermode="x unified", title=f'{ticker} Buy/Sell points') # Currently plotly doesn't support hover for overlapping points in same trace
    
    plot(fig)    


def performance_chart_shape(ticker='TSLA'):

    #ticker = 'XOS'
    
    all_212_equities = pd.read_sql_table("equities", con=engine, index_col='index')
    
    try:
        market = all_212_equities[all_212_equities['INSTRUMENT'] == ticker]['MARKET NAME'].values[0] 
        yf_symbol = get_yf_symbol(market, ticker)   
    except:
        print("Can't find ticker")
        yf_symbol = ticker
    
    start = datetime(2020, 2, 7)
    end = datetime.now()    
    
    #index = web.DataReader(yf_symbol, start, end)
    
    yf.pdr_override()
    index = web.get_data_yahoo(yf_symbol, start=start, end=end)
    
    index = index.reset_index()
    
    # cache tesla data because function takes too long
    # Heroku has a 30sec timeout
    
    portfolio = pd.read_sql_table("trades", con=engine, index_col='index', parse_dates=['Trading day'])
                
    if len(portfolio[portfolio['Ticker Symbol'] == ticker]) > 150: 
        
        # data = pd.read_csv(f'cached_data/{ticker}.csv')
        data = pd.read_sql_table(f'{ticker}', con=engine, index_col='index')
        
        buys = data[data['Type']=='Buy']
        sells = data[data['Type']=='Sell']
        
        # a = buys.append(sells)
        # a.to_csv(f'cached_data/{ticker}.csv')
        # a.to_sql(f'{ticker}', engine, if_exists='replace')
        
    else:
    
        buys, sells = get_buy_sell(ticker) 
        
        index['Midpoint'] = (index['High'] + index['Low']) / 2
        
        buy_target = []
        sell_target = []
        
        for i, row in buys.iterrows():
            
            try:
                mid = index[index['Date'] == row['Trading day']]['Midpoint'].values[0]
                
                if row['Execution_Price'] < mid:
                    buy_target.append(1)
                else:
                    buy_target.append(0)
            except:   
                # Missing data from yahoo finance
                print(f'Missing data {row}')
        
        for i, row in sells.iterrows():
            
            try:
                mid = index[index['Date'] == row['Trading day']]['Midpoint'].values[0]
                
                if row['Execution_Price'] > mid:
                    sell_target.append(1)
                else:
                    sell_target.append(0)
            except:
                print(f'Missing data {row}')
    
        
        buys['Target'] = buy_target
        sells['Target'] = sell_target

    ## Discrete color graph
    
    main_fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig = go.Figure(data=[go.Candlestick(x=index['Date'],
                    open=index['Open'],
                    high=index['High'],
                    low=index['Low'],
                    close=index['Adj Close'],
                    name='Stock')])
    
    # fig = go.Candlestick(x=index['Date'],
    #             open=index['Open'],
    #             high=index['High'],
    #             low=index['Low'],
    #             close=index['Adj Close'],
    #             name='Stock')
    
    # Must be a string for plotly to interpret numeric values as a discrete value
    # https://plotly.com/python/discrete-color/
    sells['Target'] = sells['Target'].astype(str)
    buys['Target'] = buys['Target'].astype(str)
        
    if len(sells) > 0:
        
        fig1 = px.scatter(sells, x='Trading day', y='Execution_Price', color='Target')
    
        for x in fig1.data:
            if x['legendgroup'] == '1':
                x['marker'] =  {'color':'#3D9970', 'line': {'width': 2},'size': 12, 'symbol': 'triangle-down'}
                x['name'] = 'Successful Sell Point'
                fig.add_trace(x)
            elif x['legendgroup'] == '0':
                x['marker'] =  {'color':'#E24C4F', 'line': {'width': 2}, 'size': 12, 'symbol': 'triangle-down'}
                x['name'] = 'Unsuccessful Sell Point'
                fig.add_trace(x)
    
    if len(buys) > 0:

        fig2 = px.scatter(buys, x='Trading day', y='Execution_Price', color='Target')
        #fig2.update_traces(marker=dict(color='blue'))
        #fig2.update_traces(marker=dict(color='#30C296', size=7, line=dict(width=2, color='DarkSlateGrey')))
        
        for x in fig2.data:
            if x['legendgroup'] == '1':
                x['marker'] =  {'color':'#3D9970', 'line': {'width': 2},'size': 12, 'symbol': 'triangle-up'}
                x['name'] = 'Successful Buy Point'
                fig.add_trace(x)
            elif x['legendgroup'] == '0':
                x['marker'] =  {'color':'#E24C4F', 'line': {'width': 2},'size': 12, 'symbol': 'triangle-up'}
                x['name'] = 'Unsuccessful Buy Point'
                fig.add_trace(x)
    
    for x in range(len(fig.data)):
        main_fig.add_trace(fig.data[x], secondary_y=True)
    
    # include a go.Bar trace for volumes
    volume_fig = go.Figure(go.Bar(x=index['Date'], y=index['Volume'], name='Volume'))
    volume_fig.update_traces(marker_color='rgb(158,202,225)', opacity=0.6)
    
    main_fig.add_trace(volume_fig.data[0], secondary_y=False)
    
    main_fig.update_layout(hovermode="x unified", title=f'{ticker} Stock Graph', 
                    #   legend=dict(
                    #         yanchor="top",
                    #         y=0.99,
                    #         xanchor="left",
                    #         x=0.01
                    # )
                    # showlegend=False
                    )
    
    main_fig.layout.yaxis1.showgrid=False

    #plot(main_fig)
    
    # count = buys['Target'].value_counts().add(sells['Target'].value_counts(),fill_value=0)
    # percentage = count[1]/count.sum() *100
    # percentage = '{:.2f}'.format(percentage)
    
    return fig_layout(main_fig) #, percentage














