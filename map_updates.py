# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 21:58:24 2021

@author: david
"""

import os
from dotenv import load_dotenv
import yfinance as yf
from datetime import datetime, time
from sqlalchemy import create_engine
from iexfinance.stocks import Stock
from scraper import getPremarketChange, get_driver
import pandas as pd

load_dotenv(verbose=True, override=True)

# db_URI = os.getenv('AWS_DATABASE_URL')

#db_URI = os.getenv('HEROKU_DATABASE_URL')

db_URI = os.getenv('DATABASE_URL')

engine = create_engine(db_URI)

holdings = pd.read_sql_table("portfolio", con=engine, index_col='index')

def get_holdings():
    holdings = pd.read_sql_table("portfolio", con=engine, index_col='index')
      
    # Recent ticker change due to merger, Yahoo finance pulls wrong data, should be fixed later
    #holdings = holdings[holdings['Ticker'] != 'UWMC']
    
    holdings['PREV_CLOSE'] = holdings['PREV_CLOSE'].astype('float')
    return holdings

def return_map():
       
    holdings = pd.read_sql_table("portfolio", con=engine, index_col='index')
    
    def uk_current_price(r):
        #print(r['YF_TICKER'])
        if r['YF_TICKER'].find('.L') != -1:
            r['CURRENT_PRICE'] = yf.download(tickers=r['YF_TICKER'], period='1m', progress=False)['Close'].values[0]
            return r
        else:
            return r
        
    # Before 9am do nothing, should use after hours 
    # if time(hour=8, minute=59) > datetime.now().time():
    #     holdings = pd.read_sql_table("charts", con=engine, index_col='index')
    #     holdings = holdings.apply(uk_current_price, axis=1)
    #     #print('done')
    #     return holdings.to_sql('charts', engine, if_exists='replace')
    
    # if time(hour=9, minute=0) < datetime.now().time() < time(hour=14, minute=30) or time(hour=21) < datetime.now().time() < time(hour=22):
    #     driver = get_driver(#
    #         headless=True
    #         #proxy=True
    #         )
    
    # holdings = get_holdings()
    
    # tickr = Stock('AMC') 
    # #tickr = Stock('TSLA')
    # data = tickr.get_quote()
    # index = yf.download(tickers='AMC', period='1m', progress=False)
    # index2 = yf.download(tickers='AMC', period='2d')['Open'].values[0]

    # close = 36.77
    # pnow = index['Close'][0]

    # ((pnow - close) / close ) * 100

    def current_price(r):
        #print(r['YF_TICKER'])
        ## US Stocks open different time to UK 
        if time(hour=9, minute=0) < datetime.now().time() < time(hour=14, minute=30) or time(hour=21) < datetime.now().time() < time(hour=22):
            if r['YF_TICKER'].find('.') == -1:
                try:
                    # Use IEX, only works with US (NYSE) Stocks
                    # Note 15min delay on price
                    
                    tickr = Stock(r['YF_TICKER']) 
                    #tickr = Stock('TSLA')
                    data = tickr.get_quote()
                    
                    if data['primaryExchange'].values[0].find('NASDAQ') == 0:
                        #r['CURRENT_PRICE'] = getPremarketChange(r['YF_TICKER'], driver)
                        print('getPremarketChange')
                        
                        latestPrice = data['latestPrice'].values[0]
                        extendedPrice = data['extendedPrice'].values[0]
                        #iexRealtimePrice = data['iexRealtimePrice'].values[0]
                        
                        # Only works with non nasdaq stocks due to new regulations 
                        #https://intercom.help/iexcloud/en/articles/3210401-how-do-i-get-nasdaq-listed-stock-data-utp-data-on-iex-cloud
                        r['CURRENT_PRICE'] = latestPrice if extendedPrice is None else extendedPrice
                        
                    else:             
                        # a = tickr.get_price_target()
                        # b = tickr.get_estimates()
                        print('getmarketChange')
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
    
    holdings = holdings.apply(current_price, axis=1)
    
    # if 'driver' in locals():
    #     driver.close()
    #     driver.quit()
    
    def prev_price(r):
        print(r['YF_TICKER'])
        try:
            # look at date, must not be today us stocks work with values[-1] others need values[0]
            if r['YF_TICKER'].find('.') != -1:
                # VFEM is broken on yfinance
                r['PREV_CLOSE'] = yf.download(tickers=r['YF_TICKER'], period='2d')['Adj Close'].values[0]
            else:
                #print('US')
                if time(hour=14, minute=30) < datetime.now().time():
                    r['PREV_CLOSE'] = yf.download(tickers=r['YF_TICKER'], period='2d')['Open'].values[0] # works while market is open
                else:
                    r['PREV_CLOSE'] = yf.download(tickers=r['YF_TICKER'], period='2d')['Adj Close'].values[-1] # previous close
        except IndexError:
            print(f"Can't find previous price for {r['YF_TICKER']}")
            r['PREV_CLOSE'] = float('NaN')
            pass
        return r
    
    holdings = holdings.apply(prev_price, axis=1)
    
    holdings.dropna(axis=0, inplace=True)
    
    # Return chart
    holdings['Ri'] = (holdings['CURRENT_PRICE'] - holdings['PRICE']) / abs(holdings['PRICE']) *100 # May be slightly off due to fx
    
    # Day chart
    holdings['PCT'] = (holdings['CURRENT_PRICE'] - holdings['PREV_CLOSE']) / abs(holdings['PREV_CLOSE']) *100
    holdings['PCT'] = holdings['PCT'].round(2)    
    
    # Yahoo finance pulls wrong data, should be fixed later
    holdings = holdings[holdings['Ticker'] != 'IITU']
    holdings = holdings[holdings['Ticker'] != '3CRM']
    
    holdings.to_sql('charts', engine, if_exists='replace')

    print('done')
    
return_map()















