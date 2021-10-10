# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 17:52:14 2021

@author: david
"""
import pandas as pd
import os
from dotenv import load_dotenv
import yfinance as yf
from datetime import datetime, time
from sqlalchemy import create_engine
from iexfinance.stocks import Stock
from scraper import getPremarketChange, get_driver
import time as t
import schedule

load_dotenv(verbose=True, override=True)

db_URI = os.getenv('AWS_DATABASE_URL')
engine = create_engine(db_URI)

def get_holdings():
    holdings = pd.read_sql_table("portfolio", con=engine, index_col='index')
         
    # Recent ticker change due to merger, Yahoo finance pulls wrong data, should be fixed later
    #holdings = holdings[holdings['Ticker'] != 'UWMC']
    
    holdings['PREV_CLOSE'] = holdings['PREV_CLOSE'].astype('float')
    return holdings

def day_chart():
        
#    print('start')
#    start_time = t.time()
    
    def uk_current_price(r):
        #print(r['YF_TICKER'])
        if r['YF_TICKER'].find('.L') != -1:
            r['CURRENT_PRICE'] = yf.download(tickers=r['YF_TICKER'], period='1m', progress=False)['Close'].values[0]
            return r
        else:
            return r
        
    # Before 9am do nothing, should use after hours 
    if time(hour=8, minute=59) > datetime.now().time():
        holdings = pd.read_sql_table("day_chart", con=engine, index_col='index')
        holdings = holdings.apply(uk_current_price, axis=1)
        #print('done')
        return holdings.to_sql('day_chart', engine, if_exists='replace')
    
    # if time(hour=9) < datetime.now().time() < time(hour=14, minute=30) or time(hour=21) < datetime.now().time() < time(hour=22, minute=30):
    #     driver = get_driver(
    #         headless=True
    #         #proxy=True
    #         )
    
    def current_price(r):
        #print(r['YF_TICKER'])
        if time(hour=9) < datetime.now().time() < time(hour=14, minute=30) or time(hour=21) < datetime.now().time() < time(hour=22, minute=30):
            if r['YF_TICKER'].find('.') == -1:
                try:
                    # Use IEX, only works with US (NYSE) Stocks
                    tickr = Stock(r['YF_TICKER']) 
                    #tickr = Stock('TSLA')
                    data = tickr.get_quote()
                    
                    if data['primaryExchange'].values[0].find('NASDAQ') == 0:
                        #r['CURRENT_PRICE'] = getPremarketChange(r['YF_TICKER'], driver) # Too many calls to site get blocked
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
        
    holdings = get_holdings()
    holdings = holdings.apply(current_price, axis=1)
        
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
                    r['PREV_CLOSE'] = yf.download(tickers=r['YF_TICKER'], period='2d')['Adj Close'].values[0] # works while market is open
                else:
                    r['PREV_CLOSE'] = yf.download(tickers=r['YF_TICKER'], period='2d')['Adj Close'].values[-1] # previous close
        except IndexError:
            print(f"Can't find previous price for {r['YF_TICKER']}")
            r['PREV_CLOSE'] = float('NaN')
            pass
        return r
    
    holdings = holdings.apply(prev_price, axis=1)
    # if 'driver' in locals():
    #     driver.close()
    #     driver.quit()
        
    holdings.dropna(axis=0, inplace=True)
    holdings['PCT'] = (holdings['CURRENT_PRICE'] - holdings['PREV_CLOSE']) / abs(holdings['PREV_CLOSE']) *100
    holdings['PCT'] = holdings['PCT'].round(2)    
    
    # Yahoo finance pulls wrong data, should be fixed later
    holdings = holdings[holdings['Ticker'] != 'IITU']
    holdings = holdings[holdings['Ticker'] != '3CRM']
    
    holdings.to_sql('day_chart', engine, if_exists='replace')
  
day_chart()

#    end_time = t.time()
#    print(end_time-start_time)
#    print('end')

#def job():
#    day_chart()
#                
#schedule.every(30).seconds.do(job)
#
#while True:
#    if datetime.now().time() < time(hour=20, minute=40):
#        schedule.run_pending()
#        t.sleep(1)
#    else:
#        break
    
    
    
    
    
    
    
    