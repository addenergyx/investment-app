# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 12:23:51 2021

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

load_dotenv(verbose=True, override=True)

#db_URI = os.getenv('AWS_DATABASE_URL')

db_URI = os.getenv('ElephantSQL_DATABASE_URL')

engine = create_engine(db_URI)

# ------------------------------------------------           
#
# Calculates total portfolio returns and individual stock returns
# Trading212 currently doesn't show total returns of individual stocks
# Averages are in GBP
#
# ------------------------------------------------

# This function broke on 10/2 daily statement email due to trading212 not sending
# daily statement for 28/1 :(

#df = trades[trades['Ticker Symbol'] == 'ROKU']

def returns():

    #trades = pd.read_sql_table("trades", con=engine, index_col='index', parse_dates=['Trading day'])
    
    trades = pd.read_sql_table("trades", con=engine, parse_dates=['Trading day']) # Want to parse index col for calculating results
    
    trades.sort_values(['Trading day','Trading time'], inplace=True, ascending=True)

    trades['Result'] = ''
    trades['Roling Average £'] = ''
        
    #trades = trades[trades['Trading day'] == '2021-02-11']
    #trades = trades[trades['Trading day'] == '2021-02-10']
    
    # all_holdings = trades['Ticker Symbol'].unique()
    
    ## Getting all tickers and isin from portfolio
    temp_df = trades.drop_duplicates('Ticker Symbol')
    
    all_holdings = temp_df[['Ticker Symbol', 'ISIN']] 
    
    holdings_dict = collections.defaultdict(dict) # Allows for nesting easily
    returns_dict = collections.defaultdict(dict)
    
    averages = pd.DataFrame(columns=['Trading day', 'Ticker Symbol', 'Average', 'Exchange rate'])
    
    for symbol in all_holdings['Ticker Symbol'].tolist():
        
        # symbol = 'BLNK'
        
        df = trades[trades['Ticker Symbol'] == symbol]
                        
        df = df.reset_index().drop('level_0', axis=1)
            
        last_sell = df[df.Type == 'Sell'].tail(1)
        
        # # if no sells
        # if len(a) == 0:
        #     a = df.tail(1)
            
        a = df.tail(1)
        
        print(f'-------{symbol}-------')
                    
        ## currently does not take into account fees 
        ## should use total cost column instead later
        
        temp_df = df.iloc[:int(a['index'])+1]
        
        # share_lis = df['Shares'][:int(a['index'])+1].tolist()
        # price_lis = df['Price'][:int(a['index'])+1].tolist()
        # type_lis = df['Type'][:int(a['index'])+1].tolist()
        # day_lis = df['Trading day'][:int(a['index'])+1].tolist()
        # fx_lis = df['Exchange rate'][:int(a['index'])+1].tolist()

        #fees_lis = df['Charges and fees'][:ii+1].tolist()
        aa=[]
        
        c = 0
        x = 0
        holdings = 0
        average = 0
        total_returns = 0
        
        # if symbol == 'VWDRY':
        #     data = yf.Ticker(symbol) # Get stock data
        #     split_df = data.splits.reset_index() # Extract stock split dates and values
        #     gain_loss = sell_row['Price'] - rolling_average[ii-1]/split_df.splits[0] 
        #     split = split_df[ (max(df['Trading day']) >= split_df['Date']) & (split_df['Date'] > row['Trading day'])]['Stock Splits'].sum()
                
        #for s, p, t, d, fx in list(zip(share_lis, price_lis, type_lis, day_lis, fx_lis)):
        for ii, row in temp_df.iterrows():
            
            if row['Type'] == 'Buy':
                c += row['Shares']*row['Price']
                holdings += row['Shares']
                average = c / holdings
                                    
                # print(f"Buy Order: {row['Shares']} @ £{row['Price']}")
                # print(f'New Holdings Average: {holdings} @ £{average}')
                
                row['Roling Average £'] = average
                trades[trades['index'] == row['index']] = row
            
            else:
                ## Selling stock
                
                ## if ii == len(share_lis): <- Doesn't work, This is probably because in the Python 3.x, 
                ## zip returns a generator object. This object is not a list
                ## https://stackoverflow.com/questions/31011631/python-2-3-object-of-type-zip-has-no-len/38045805
    
                if last_sell.index[0] == x:
                    
                    average = c / holdings
                    gain_loss = row['Price'] - average
                    total_profit = gain_loss * row['Shares']
                    print(f'Current Holdings Average: {holdings} @ {average}')
                    print(f"Final Sell Order: {row['Shares']} @ {row['Price']}")
                    print(f'Return on Investment: {round(total_profit, 2)}')
                    total_returns += round(total_profit, 2)
                    print(f'Total Return on Investment: {round(total_returns, 2)}')
                    row['Roling Average £'] = average
                    row['Result'] = round(total_profit, 2)
                    trades[trades['index'] == row['index']] = row
                                    
                else:
                    holdings -= row['Shares'] 
                    print(f"Sell Order: {row['Shares']} @ {row['Price']}")
                    
                    if holdings == 0:
                        ## Reset average after liquidating stock
                        gain_loss = row['Price'] - average
                        total_profit = gain_loss * row['Shares']
                        print(f'Return on Investment: {round(total_profit, 2)}')
                        total_returns += round(total_profit, 2)
                        average = 0
                        c = 0
                        row['Roling Average £'] = average
                        row['Result'] = round(total_profit, 2)
                        trades[trades['index'] == row['index']] = row
                        print('Sold all holdings')
                    else:
                        print(f'New Holdings Average: {holdings} @ {average}')
                        ## Take away shares from from holding average
                        ## However average stays the same
                        c -= row['Shares']*average
                        gain_loss = row['Price'] - average
                        total_profit = gain_loss * row['Shares']
                        print(f'Return on Investment: {round(total_profit, 2)}')
                        total_returns += round(total_profit, 2)
                        row['Roling Average £'] = average
                        row['Result'] = round(total_profit, 2)
                        trades[trades['index'] == row['index']] = row
                
                
                # if symbol in holdings_dict:
                #     #returns_dict[symbol]['Returns'] += total_profit
                    
                #     if total_profit > 0:
                #         holdings_dict[symbol]['Gains'] += total_profit
                #     else:
                #         holdings_dict[symbol]['Losses'] += total_profit
            
                # else:
                #     #returns_dict[symbol]['Returns'] = total_profit
                    
                #     if total_profit > 0:
                #         holdings_dict[symbol]['Gains'] = total_profit
                #         holdings_dict[symbol]['Losses'] = 0
                #     else:
                #         holdings_dict[symbol]['Losses'] = total_profit
                #         holdings_dict[symbol]['Gains'] = 0
                
                if symbol in holdings_dict:
                    # returns_dict[symbol] += total_profit
                    if total_profit > 0:
                        holdings_dict[symbol]['Gains'] += total_profit
                    else:
                        holdings_dict[symbol]['Losses'] += total_profit
                    
                    holdings_dict[symbol]['Gross Returns'] += total_profit

                else:
                    # returns_dict[symbol] = total_profit
                    
                    if total_profit > 0:
                        holdings_dict[symbol]['Gains'] = total_profit
                        holdings_dict[symbol]['Losses'] = 0
                    else:
                        holdings_dict[symbol]['Losses'] = total_profit
                        holdings_dict[symbol]['Gains'] = 0
                    
                    holdings_dict[symbol]['Gross Returns'] = total_profit

                             
                # if row['Trading day'] == datetime(2021, 11, 10):
                #     print(total_profit)
                #     print(total_returns)
                
                if row['Trading day'] in returns_dict:    
                    returns_dict[row['Trading day']]['Returns'] += total_profit
                    
                    if total_profit > 0:
                        returns_dict[row['Trading day']]['Gains'] += total_profit
                    else:
                        returns_dict[row['Trading day']]['Losses'] += total_profit
                    
                else:
                    returns_dict[row['Trading day']]['Returns'] = total_profit
                    
                    if total_profit > 0:
                        returns_dict[row['Trading day']]['Gains'] = total_profit
                        returns_dict[row['Trading day']]['Losses'] = 0
                    else:
                        returns_dict[row['Trading day']]['Losses'] = total_profit
                        returns_dict[row['Trading day']]['Gains'] = 0
                    
                    #print('-----------------')         
                    #break #Use break because don't care about orders after sell order
                
                
                print('----------------')
            
            
            aa.append(average)
            x += 1
    
    df['Roling average'] = aa
    
    #averages = averages.drop_duplicates(['Trading day', 'Ticker Symbol'], keep='last')
    
    print(f'Gross Returns: {total_returns}')
    net_returns = total_returns - trades['Charges and fees'].sum()
    print(f'Net Returns: {net_returns}')
    
    for symbol in all_holdings['Ticker Symbol'].tolist():
                
        df = trades[trades['Ticker Symbol'] == symbol]
        
        df = df.reset_index().drop('index', axis=1)
        
        # formatting float to resolve floating point Arithmetic Issue
        # https://www.codegrepper.com/code-examples/delphi/floating+point+precision+in+python+format
        # https://docs.python.org/3/tutorial/floatingpoint.html
        # could also use the math.isclose() function
        df['Shares'] = df['Shares'].apply(lambda x: float("{:.6f}".format(x))) # Trading 212 only allows fraction of shares up to 6dp 
    
        ## Watchlist
        if df.empty:
            holdings_dict[symbol]['Current Holdings'] = 0
            holdings_dict[symbol]['Current Average'] = 0.0
        
        else:
            print(f'------- {symbol} History -------')
            
            for ii, row in df.iterrows():
                
                ## currently does not take into account fees 
                ## should use total cost column instead later
                ## trading212 doesn't include fees in returns per stock
                
                share_lis = df['Shares'][:ii+1].tolist()
                price_lis = df['Price'][:ii+1].tolist()
                type_lis = df['Type'][:ii+1].tolist()
            
                c = holdings = average = 0 
                #= x
                        
                for s, p, t, in list(zip(share_lis, price_lis, type_lis)):
                    
                    if t == 'Buy':
                        c += s*p
                        holdings += s
                        average = c / holdings
                        print(f'Buy Order: {s} @ {p}')
                        print(f'New Holdings Average: {holdings} @ {average}')
                    
                    else:
        
                        holdings -= s 
                        print(f'Sell Order: {s} @ {p}')
                        
                        if holdings == 0:
                            ## Reset average after liquidating stock
                            average = 0
                            c = 0
                            print('Sold all holdings')
                        else:
                            print(f'New Holdings Average: {holdings} @ {average}')
                            ## Take away shares from from holding average
                            ## However average stays the same
                            c -= s*average
              
            holdings_dict[symbol]['Current Holdings'] = holdings #formatting(holdings)
            holdings_dict[symbol]['Current Average'] = average #formatting(average)
                
            print(f'Holdings Average: {holdings} @ {average}')  
    
    holdings_df = pd.DataFrame.from_dict(holdings_dict, orient='index').reset_index().rename(columns={'index':'Ticker Symbol'})
    returns_df = pd.DataFrame.from_dict(returns_dict, orient='index').reset_index().rename(columns={'index':'Date'})

    returns_df['Date'] = pd.to_datetime(returns_df['Date'], format='%d-%m-%Y')
    
    #returns_df = pd.DataFrame(holdings_dict).transpose().reset_index(level=0).rename(columns={'index':'Ticker Symbol'})
    returns_df.to_sql('returns', engine, if_exists='replace')
    holdings_df.to_sql('holdinigs', engine, if_exists='replace')
    
    trades.drop(columns=['index'], inplace=True)
    trades.sort_values(['Trading day','Trading time'], inplace=True, ascending=True)
    trades['Result'].replace('', 0.0, inplace=True)
    trades['Result'] = trades['Result'].astype(float)

    print(sum(returns_df['Returns']))

    trades.to_sql('trades', engine, if_exists='replace') # Added results column
    trades.to_csv('Investment trades.csv', index=False )
    
    return trades
    
returns()



















