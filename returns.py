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

# db_URI = os.getenv('AWS_DATABASE_URL')

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

def stock_split_adjustment(r):

    try:
        data = yf.Ticker(r['YF_TICKER']) # Get stock data
        split_df = data.splits.reset_index() # Extract stock split dates and values
        split = split_df[split_df['Date'] >= r['Trading day']]['Stock Splits'].sum()  
    except:
        # Assume split adjustment not needed
        # Stock may be delisted like HGM.L, WORK & CDM.L
        return r
    
    # r.Stock_split = split
    
    if split > 0:
        r.post_split_shares = float(r.Shares * split)
        
    return r

# def post_stock_split_adjustment(r):

#     print(r['YF_TICKER'].to_string())
    
#     data = yf.Ticker(r['YF_TICKER']) # Get stock data
#     split_df = data.splits.reset_index() # Extract stock split dates and values
#     split = split_df[split_df['Date'] <= r['Trading day']]['Stock Splits'].sum()  
    
#     r.Stock_split = split
        
#     return r

def returns():

    #r = pd.read_sql_table("holdings", con=engine, index_col='index', parse_dates=['Trading day'])
        
    trades = pd.read_sql_table("trades", con=engine, parse_dates=['Trading day']) # Want to parse index col for calculating results
        
    trades.sort_values(['Trading day','Trading time'], inplace=True, ascending=True)

    trades['Result'] = ''
    
    # Helper columns
    trades['post_split_shares'] = ''
    trades['rolling_average'] = ''
    trades['Stock_split'] = ''
    trades['current_holdings'] = ''
    trades['pre_split_rolling_average'] = ''
    trades['adjusted_split_shares'] = ''
    trades['adjusted_split_price'] = ''
        
    #trades = trades[trades['Trading day'] == '2021-02-11']
    #trades = trades[trades['Trading day'] == '2021-02-10']
    
    # all_holdings = trades['Ticker Symbol'].unique()
    
    ## Getting all tickers and isin from portfolio
    temp_df = trades.drop_duplicates('Ticker Symbol')
    
    all_holdings = temp_df[['Ticker Symbol', 'ISIN']] 

    total_returns = 0
    
    holdings_dict = collections.defaultdict(dict) # Allows for nesting easily
    returns_dict = collections.defaultdict(dict)
    
    dunno_averages = pd.DataFrame(columns=['Trading day', 'Ticker Symbol', 'Average', 'Exchange rate'])
    
    for symbol in all_holdings['Ticker Symbol'].tolist():
        
        symbol = 'BLNK' #'XSPA' #'CDM.L' #'TSLA' #'AML' #LTC #VWDRY
        
        print(f'-------{symbol}-------')
        
        df = trades[trades['Ticker Symbol'] == symbol]
                        
        df = df.reset_index().drop('level_0', axis=1)
        
        # Try once before runniong apply
        # try:
            # data_df = yf.Ticker('NIU').splits.reset_index() 
            # data_df = data_df[data_df['Date'] >= min(df['Trading day'])]
            
        #     if len(data_df.index) > 0:
        #         # TODO: make this quicker by only sending part of dataframe that needs split adjustment
        #         df = df.apply(stock_split_adjustment, axis=1)
            
        # except AttributeError:
        #     data_df = pd.DataFrame()
        
        rolling_average = []
        quantities = []
        averages = []
        holdings_lis = []
        aas = []
        
        holdings = average = 0
        
        # change all iterrows to be like below
        for i, row in df.iterrows():
                
            temp_df = df.iloc[:i+1]
            # print(temp_df)
            
            if row['Type'] == 'Buy':
                
                if not row['post_split_shares']:
                    holdings += row['Shares']
                    quantities.append(row['Shares'])
                else: 
                    holdings += row['post_split_shares']    
                    quantities.append(row['post_split_shares'])

                #print(holdings)
                
                # The average cost price is calculated by reference to purchases only; it makes no reference to sales.
                            
                averages.append(row['Price'])

                sums = [a*b for a,b in zip(averages, quantities)]
    
                # if len(quantities) == 0:
                #     average = 0
                # else:
                average = sum(sums) / sum(quantities)
            
                rolling_average.append(average)
              
            # Selling
            else:
                
                if not row['post_split_shares']:
                    holdings -= float("{:.6f}".format(row['Shares']))  
                else: 
                    holdings -= float("{:.6f}".format(row['post_split_shares']))
                
                # Floating Point Arithmetic: Issues and Limitations
                holdings = float("{:.6f}".format(holdings))
                
                #print(holdings)
                
                quantities.append(float("{:.6f}".format(holdings)))
                
                if holdings == 0:
                    ## Reset average after liquidating stock
                    average = 0
                    rolling_average.append(0)
                    averages = []
                    quantities = []
                else:
                    
                    #print(f'New Holdings Average: {holdings} @ {average}')
                    
                    ## Take away shares from from holding average
                    ## However average stays the same
                    rolling_average.append(average) # should equal previous average, sales don't change average
                    averages = []
                    quantities = []
            
            holdings_lis.append(holdings)
            aas.append(average)
            # print(holdings)

        df['rolling_average'] = rolling_average
        df['current_holdings'] = holdings_lis
        
        pre_split_rolling_average = []
        adjusted_split_shares = []
        adjusted_split_price = []
                
        # if len(data_df.index) > 0:
        #     for ii, row in df.iterrows():
        #         data = yf.Ticker(row['YF_TICKER']) # Get stock data
        #         split_df = data.splits.reset_index() # Extract stock split dates and values

        #         split = split_df[ (max(df['Trading day']) >= split_df['Date']) & (split_df['Date'] > row['Trading day'])]['Stock Splits'].sum()
        #         # print(row['rolling_average'])
        #         # print(split)
        #         if split == 0:
        #             pre_split_rolling_average.append(row['rolling_average'])
        #             adjusted_split_shares.append(row['Shares'])
        #             adjusted_split_price.append(row['Price'])
        #         else:
        #             b = row['rolling_average']/split
        #             c = row['Shares']*split
        #             d = row['Price']/split
        #             pre_split_rolling_average.append(b)
        #             adjusted_split_shares.append(c)
        #             adjusted_split_price.append(d)
        # else:
        #     for ii, row in df.iterrows():
        #         pre_split_rolling_average.append(row['rolling_average'])
        #         adjusted_split_shares.append(row['Shares'])
        #         adjusted_split_price.append(row['Price'])
        
        # df['pre_split_rolling_average'] = pre_split_rolling_average
        # df['adjusted_split_shares'] = adjusted_split_shares
        # df['adjusted_split_price'] = adjusted_split_price
        
        a = df[df.Type == 'Sell']
            
        for ii, sell_row in a.iterrows():
                        
            # sell_row = a.iloc[1]
            
            # if rolling_average[ii] == 0:
            #     gain_loss = sell_row['Price'] - rolling_average[ii-1]/data.splits[0] # Fix later, use stock_split_adjustment()
            #     total_profit = gain_loss
            # else:
            #     gain_loss = sell_row['Price'] - rolling_average[ii-1]
            #     total_profit = (gain_loss * sell_row['Shares']) - sell_row['Total cost']
            
            #symbol = 'CDM.L'
            
            # if len(data_df.index) > 0:                
            #     data = yf.Ticker(sell_row['YF_TICKER']) # Get stock data
            #     split_df = data.splits.reset_index() # Extract stock split dates and values
            #     split = split_df[ (min(df['Trading day']) <= split_df['Date']) & (split_df['Date'] <= row['Trading day'])]['Stock Splits'].sum()
                
            #     if split == 0:
            #         gain_loss = sell_row['Price'] - rolling_average[ii-1]
            #     else:
            #         gain_loss = sell_row['Price'] - rolling_average[ii-1]/split
            # else:             
            #     gain_loss = sell_row['Price'] - rolling_average[ii-1]
            
               
            gain_loss = sell_row['Price'] - rolling_average[ii-1]
            
            # sold all stock
            # if sell_row['current_holdings'] == 0:
            #     gain_loss = sell_row['Price'] - rolling_average[ii-1]
            
            total_profit = gain_loss * sell_row['Shares'] #- sell_row['Char']
            
            #print(total_profit)
            
            total_returns += round(total_profit, 2)
                        
            sell_row['Result'] = round(total_profit, 2)
            
            trades[trades['index'] == sell_row['index']] = sell_row
        
            print(f'Sell Order: {sell_row["Shares"]} @ £{sell_row["Price"]}')
            print(f'Total Return on Investment: {round(total_profit, 2)}')
            print('--------------')
            total_returns += round(total_profit, 2)
            
            dunno_averages.loc[len(averages)] = [sell_row['Trading day'], symbol, average, sell_row['Exchange rate']]
        
            if symbol in holdings_dict:
                                
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
        
        
        
        ## old #############################################################
        
    #     averages = pd.DataFrame(columns=['Trading day', 'Ticker Symbol', 'Average', 'Exchange rate'])
        
    #     for ii, sell_row in a.iterrows():
            
    #         ## currently does not take into account fees  
    #         ## should use total cost column instead later
            
    #         share_lis = df['Shares'][:ii+1].tolist()
    #         price_lis = df['Price'][:ii+1].tolist()
    #         type_lis = df['Type'][:ii+1].tolist()
    #         day_lis = df['Trading day'][:ii+1].tolist()
    #         fx_lis = df['Exchange rate'][:ii+1].tolist()
    #         split_lis = df['post_split_shares'][:ii+1].tolist()
    #         avg_lis = df['rolling_average'][:ii+1].tolist()

    #         #fees_lis = df['Charges and fees'][:ii+1].tolist()
        
    #         c = 0 # Total spent on stock
    #         x = 0
    #         holdings = 0
    #         average = 0
            
    #         #averages = []
    #         quantities = []
    #         averages_gbp = []
            
    #         for s, p, t, d, fx, split, avg in list(zip(share_lis, price_lis, type_lis, day_lis, fx_lis, split_lis, avg_lis)):
                
    #             if t == 'Buy':
    #                 c += s*p
                    
    #                 if not split:
    #                     holdings += s  
    #                 else: 
    #                     holdings += split
                                        
    #                 # if fx != 0.01:
    #                 #     average = p * (1/fx)
    #                 # else:
    #                 #     average = c / holdings

    #                 # averages.loc[len(averages)] = [d, symbol, average, fx]
                    
    #                 # print(f'Buy Order: {s} @ {p}')
    #                 # print(f'New Holdings Average: {holdings} @ {average}')
                        
    #                 average_gbp = p

    #                 if not split:
    #                     quantities.append(s) 
    #                 else: 
    #                     quantities.append(split)
                    
    #                 averages_gbp.append(average_gbp)

    #                 sums_gbp = [a*b for a,b in zip(averages_gbp, quantities)]
        
    #                 if len(quantities) == 0:
    #                     average = 0
    #                     averages_gbp = 0
    #                 else:
    #                     average = sum(sums_gbp) / sum(quantities) # could also use holdings
                
    #             else:
    #                 ## Selling stock
                    
    #                 ## if ii == len(share_lis): <- Doesn't work, This is probably because in the Python 3.x, 
    #                 ## zip returns a generator object. This object is not a list
    #                 ## https://stackoverflow.com/questions/31011631/python-2-3-object-of-type-zip-has-no-len/38045805
        
    #                 if ii == x:
                        
    #                     #average = sum(sums_gbp) / sum(quantities)
                        
    #                     gain_loss = p - average
    #                     # total_profit = gain_loss * s if not split else gain_loss * split
                        
    #                     diff = holdings - s
                        
    #                     total_profit = gain_loss * s #if diff != 0 else gain_loss
                        
    #                     # print(f'Current Holdings Average: {holdings} @ {average}')
    #                     print(f'Sell Order: {s} @ {p}')
    #                     print(f'Total Return on Invesatment: {round(total_profit, 2)}')
    #                     print('--------------')
    #                     total_returns += round(total_profit, 2)
                        
    #                     sell_row['Result'] = round(total_profit, 2)
    #                     trades[trades['index'] == sell_row['index']] = sell_row
                        
    #                     if symbol in holdings_dict:
    #                         # returns_dict[symbol] += total_profit
                            
    #                         if total_profit > 0:
    #                             holdings_dict[symbol]['Gains'] += total_profit
    #                         else:
    #                             holdings_dict[symbol]['Losses'] += total_profit
                            
    #                         holdings_dict[symbol]['Gross Returns'] += total_profit
                            
    #                     else:
    #                         # returns_dict[symbol] = total_profit
                            
    #                         if total_profit > 0:
    #                             holdings_dict[symbol]['Gains'] = total_profit
    #                             holdings_dict[symbol]['Losses'] = 0
    #                         else:
    #                             holdings_dict[symbol]['Losses'] = total_profit
    #                             holdings_dict[symbol]['Gains'] = 0
                            
    #                         holdings_dict[symbol]['Gross Returns'] = total_profit
                                                  
    #                     if d in returns_dict:    
    #                         returns_dict[d]['Returns'] += total_profit
                            
    #                         if total_profit > 0:
    #                             returns_dict[d]['Gains'] += total_profit
    #                         else:
    #                             returns_dict[d]['Losses'] += total_profit
                            
    #                     else:
    #                         returns_dict[d]['Returns'] = total_profit
                            
    #                         if total_profit > 0:
    #                             returns_dict[d]['Gains'] = total_profit
    #                             returns_dict[d]['Losses'] = 0
    #                         else:
    #                             returns_dict[d]['Losses'] = total_profit
    #                             returns_dict[d]['Gains'] = 0
                        
    #                     #print('-----------------')         
    #                     break #Use break because don't care about orders after sell order
                    
    #                 else:
    #                     holdings -= s 
    #                     # print(f'Sell Order: {s} @ {p}')
                        
    #                     if holdings == 0:
    #                         ## Reset average after liquidating stock
    #                         average = 0
    #                         c = 0
    #                         # print('Sold all holdings')
    #                     else:
    #                         # print(f'New Holdings Average: {holdings} @ {average}')
    #                         ## Take away shares from from holding average
    #                         ## However average stays the same
    #                         c -= s*average
                    
    #             x += 1
    
    # averages = averages.drop_duplicates(['Trading day', 'Ticker Symbol'], keep='last')
    
    # print(f'Gross Returns: {total_returns}')
    # net_returns = total_returns - trades['Charges and fees'].sum()
    # print(f'Net Returns: {net_returns}')
    
    for symbol in all_holdings['Ticker Symbol'].tolist():
        
        #symbol = '3LTS'
                
        df = trades[trades['Ticker Symbol'] == symbol]
        
        df = df.reset_index().drop('index', axis=1)
        
        # formatting float to resolve floating point Arithmetic Issue
        # https://www.codegrepper.com/code-examples/delphi/floating+point+precision+in+python+format
        # https://docs.python.org/3/tutorial/floatingpoint.html
        # could also use the math.isclose() function
        df['Shares'] = df['Shares'].apply(lambda x: float("{:.6f}".format(x))) # Trading 212 only allows fraction of shares up to 6dp 
    
        averages = []
        quantities = []

        ## Depricated: don't have a watchlist anymore
        if df.empty:
            holdings_dict[symbol]['Current Holdings'] = 0
            holdings_dict[symbol]['Current Average'] = 0.0
        
        else:
            print(f'------- {symbol} History -------')
            
            # for ii, row in df.iterrows():
                
            ## currently does not take into account fees 
            ## should use total cost column instead later
            ## trading212 doesn't include fees in returns per stock
            
            # share_lis = df['Shares'][:ii+1].tolist()
            # price_lis = df['Price'][:ii+1].tolist()
            # type_lis = df['Type'][:ii+1].tolist()
            # fx_lis = df['Exchange rate'][:ii+1].tolist()
            # ex_lis = df['Execution_Price'][:ii+1].tolist()
            
            # Price list is GBP, Execution_Price is the stock's currency
            
            share_lis = df['Shares'].tolist()
            price_lis = df['Price'].tolist()
            type_lis = df['Type'].tolist()
            fx_lis = df['Exchange rate'].tolist()
            ex_lis = df['Execution_Price'].tolist()
        
            c = holdings = average = 0
            #= x
                    
            # below basically same as 'for ii, row in df.iterrows():'
                        
            for s, p, t, fx, ex in list(zip(share_lis, price_lis, type_lis, fx_lis, ex_lis)):
                
                if t == 'Buy':
                    c += s*p
                    holdings += s
                
                    # The average cost price is calculated by reference to purchases only; it makes no reference to sales.
                    average = ex
                    
                    # if fx != 0.01:
                    #     average = p * (1/fx)
                    # else:
                    #     average = c / holdings

                    averages.append(average)
                    quantities.append(s)

                    sums = [a*b for a,b in zip(averages, quantities)]
        
                    if len(quantities) == 0:
                        average = 0
                    else:
                        average = sum(sums) / sum(quantities) # could aslo use holdings
                
                    print(f'Buy Order: {s} @ {ex}')
                    print(f'New Holdings Average: {holdings} @ {average}')
                    print('--------------')
                
                else:
    
                    holdings -= s 
                    print(f'Sell Order: {s} @ {ex}')
                    
                    if holdings == 0:
                        ## Reset average after liquidating stock
                        average = 0
                        c = 0
                        
                        averages = []
                        quantities = []
                        
                        print('Sold all holdings')
                    else:
                        
                        print(f'New Holdings Average: {holdings} @ {average}')
                        
                        ## Take away shares from from holding average
                        ## However average stays the same
                        
                        averages = []
                        quantities = []
                        #averages_gbp = []

                        averages.append(average)
                        quantities.append(holdings)

                        c -= s*average
                    
                    print('--------------')
                
        # sums = [a*b for a,b in zip(averages, quantities)]
        
        # if len(quantities) == 0:
        #     average = 0
        # else:
        #     average = sum(sums) / sum(quantities) # could aslo use holdings
        
        holdings_dict[symbol]['Current Holdings'] = holdings #formatting(holdings)
        holdings_dict[symbol]['Current Average'] = average #formatting(average)
            
        print(f'Current Holdings Average: {holdings} @ {average}')  
    
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

    trades.to_sql('trades', engine, if_exists='replace') # Added results column
    trades.to_csv('Investment trades.csv', index=False )

    print(sum(returns_df['Returns']))
    
    return trades
    
#returns()


# a = trades[trades['Ticker Symbol'] == 'MRO']
















