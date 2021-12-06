# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 13:05:52 2020

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
from sqlalchemy.pool import NullPool

load_dotenv(verbose=True, override=True)


# ENV VARS

email_user = os.getenv('GMAIL')
email_pass = os.getenv('GMAIL_PASS') # Make sure 'Less secure app access' is turned on

# db_URI = os.getenv('AWS_DATABASE_URL')

db_URI = os.getenv('ElephantSQL_DATABASE_URL')

db_URI = os.getenv('HEROKU_DATABASE_URL')

port = 993

SMTP_SERVER = "imap.gmail.com"

mail = imaplib.IMAP4_SSL(SMTP_SERVER)

# mail.login(email_user, email_pass)

engine = create_engine(db_URI, poolclass=NullPool)

# c = CurrencyRates()
# date = '28/01/2021 14:42'
# date_time_obj = datetime.strptime(date, '%d/%m/%Y %H:%M')
# rate = c.get_rate('USD', 'GBP', date_time_obj)

# def get_ex_rate(r):
#     date_time_obj = datetime.strptime(date, '%d/%m/%Y %H:%M')
#     rate = c.get_rate(r['Currency (Price / share)'], 'GBP', date_time_obj)
#     r['Exchange rate'] = rate
    
# missing_data.apply(get_rate, axis=1)

def get_holdings():
    holdings = pd.read_sql_table("portfolio", con=engine, index_col='index')
    engine.dispose() 
    # Recent ticker change due to merger, Yahoo finance pulls wrong data, should be fixed later
    #holdings = holdings[holdings['Ticker'] != 'UWMC']
    
    holdings['PREV_CLOSE'] = holdings['PREV_CLOSE'].astype('float')
    return holdings

def get_mailbox_list(folder):
    
    mail.select(folder)

    status, mailbox = mail.search(None, 'ALL')
        
    mailbox_list = mailbox[0].split()
    
    return mailbox_list

## For the yahoo finance api, stocks outside of the US have trailing symbols to state which market they are from
def get_yf_symbol(market, symbol):
    if market == 'London Stock Exchange' or market == 'LSE AIM':
        symbol = symbol.rstrip('.')
        yf_symbol = f'{symbol}.L'
    elif market == 'Deutsche Börse Xetra':
        yf_symbol = f'{symbol}.DE'
    elif market == 'Bolsa de Madrid':
        yf_symbol = f'{symbol}.MC'
    elif market == 'Euronext Netherlands':
        yf_symbol = f'{symbol}.AS'
    elif market == 'SIX Swiss':
        yf_symbol = f'{symbol}.SW'
    elif market == 'Euronext Paris':
        yf_symbol = f'{symbol}.PA'
    elif symbol == 'IAG':
        """
        Works assuming I never buy IAMGold Corporation (IAG)
        International Consolidated Airlines Group SA in equities df 
        whereas IAG SA in holdings so string match doesn't work
        """
        yf_symbol = 'IAG.L'  
    else:
        yf_symbol = symbol
    return yf_symbol

# all_212_equities = pd.read_sql_table("equities", con=engine, index_col='index')
# engine.dispose()

all_212_equities = pd.read_csv('https://raw.githubusercontent.com/addenergyx/investment-app/main/equities.csv')

# all_212_equities.to_sql('equities', engine, if_exists='replace')
# all_212_equities.to_csv('equities.csv', index=False)

def get_market(isin, symbol, old_symbol=''):
    
    ## When tickers change (due to mergers or company moving market) 212 removes the old ticker from the equities table
    ## As 212 doesn't provide the company name in the daily statement there is no way for me to link old tickers with the new one
    ## so will manually replace tickers here
    ## Preventing this in the future by saving all old tickers in a csv

    if symbol == 'AO.':
        old_symbol = symbol
        symbol = symbol.rstrip('.')
    elif symbol == 'DTG':
        old_symbol = symbol
        symbol = 'JET2'
    elif symbol == 'FMCI':
        old_symbol = symbol
        symbol = 'TTCF'
    elif symbol == 'SHLL':
        old_symbol = symbol
        symbol = 'HYLN'

    markets = all_212_equities.query('ISIN==@isin and INSTRUMENT==@symbol')['MARKET NAME']
    
    if len(markets) == 0:
        try:
            # Backup approach to find market
            market = all_212_equities[all_212_equities['INSTRUMENT'] == symbol]['MARKET NAME'].item() 
        except:
            # Can't find market 
            market = '-'
    else:
        market = markets.values[0]
        #if len(market) == 0: print(len(market)) 
    return symbol, market

def get_portfolio():
    
    print('Updating Orders')
    
    mailbox_list = get_mailbox_list('investing')
    
    status, mailbox = mail.search(None, 'ALL')
    
    data = []
    column_headers = ['Order ID', 'Ticker Symbol', 'Type', 
                      'Shares', 'Price', 'Total amount', 'Trading day', 
                      'Trading time', 'Commission', 'Charges and fees', 'Order Type', 
                      'Execution venue', 'Exchange rate', 'Total cost']
    
    mailbox_list = mailbox[0].split()
        
    for item in mailbox_list:
        
    # for num in data[0].split():
        status, body = mail.fetch(item, '(RFC822)')
        email_msg = body[0][1]
    
        #raw_email = email_msg.decode('utf-8')
    
        email_message = email.message_from_bytes(email_msg)
    
        counter = 1
        for part in email_message.walk():
            if part.get_content_maintype() == "multipart":
                continue
            filename = part.get_filename()
            if not filename:
                ext = '.html'
                filename = 'msg-part%08d%s' %(counter, ext)
            
            counter += 1
            
            content_type = part.get_content_type()
            # print(content_type)
            
            if "html" in content_type:
                html_ = part.get_payload()
                soup = BeautifulSoup(html_, 'html.parser')
                
                # only want trades from invest account: 1718757
                #if soup.find_all("td", string="1718757"):
                    
                # Extracting trades table
                inv = soup.select('table[class*="report"]')
                
                for table in inv:
                    rows = table.findChildren('tr')
                    for row in rows:
                        row_list = []
                        # cells = row.find_all(['th', 'td'], recursive=False)
                        cells = row.find_all('td', recursive=False)
                        for cell in cells:
                            value = cell.string                
                            if value:
                                row_list.append(value.strip())
                                #print(value.strip())
                        if row_list:
                            data.append(row_list[1:])
        
    trades = pd.DataFrame(data, columns=column_headers)
    
    #trades.to_csv('order_executions.csv')
        
    float_values = ['Shares', 'Price', 'Total amount','Commission', 'Charges and fees','Total cost', 'Exchange rate']

    for column in float_values:
        trades[column] = trades[column].str.rstrip('GBP').astype(float)
    
    ## Split ISIN and stock, need ISIN because some companies have the same ticker symbol, ISIN is the uid
    trades[['Ticker Symbol', 'ISIN']] = trades['Ticker Symbol'].str.split('/', expand=True)
    
    trades['Trading day'] = pd.to_datetime(trades['Trading day'], format='%d-%m-%Y', dayfirst=True) #pd.to_datetime(trades["Trading day"]).dt.strftime('%m-%d-%Y')
    
    ## For getting ROI, Dataframe needs to be ordered in ascending order and grouped by Ticker Symbol
    trades.sort_values(['Ticker Symbol','Trading day','Trading time'], inplace=True, ascending=True)
    
    trades['Execution_Price'] = trades['Price'] / trades['Exchange rate']
    
    ## ------------------------- Add missing data from csv ------------------------- ##
    
    data = pd.DataFrame()
    for file_name in glob.glob('missing_data/'+'*.csv'):
        x = pd.read_csv(file_name, low_memory=False)
        print(file_name)
        data = pd.concat([data,x],axis=0)

    # data = pd.read_csv('trading data export.csv')
    # data = data[data['Notes'].isnull()]
    # data = data[data['Exchange rate'] != 'Not available']

    data['Exchange rate'] = data['Exchange rate'].astype(float)

    data['Execution_Price'] = data['Price / share']
    data[['Trading day', 'Trading time']] = data['Time'].str.split(' ', expand=True)
    data['Type'] = data['Action'].str.split(' ').str[-1].str.capitalize()
    data['Trading day'] = pd.to_datetime(data['Trading day'], dayfirst=True)
    data['Price'] = data['Price / share'] / data['Exchange rate']
    
    # Change exchane rate to match email 
    data['Exchange rate'] = 1/data['Exchange rate']
            
    # time missing seconds
    data['Trading time'] = pd.to_datetime(data['Trading time']) 
    data['Trading time'] = [time.time() for time in data['Trading time']]
    data['Trading time'] = [time.strftime("%H:%M:%S") for time in data['Trading time']]

    #email
    data.drop(['Time', 'Action', 'Finra fee (GBP)', 'Currency (Price / share)', 'Result (GBP)', 'Price / share', 'Name' ], inplace=True, axis=1)  
    
    #export
    #data.drop(['Time', 'Action', 'Finra fee (GBP)', 'Currency (Price / share)', 'Price / share' ], inplace=True, axis=1)  
    
    data.rename(columns={'Ticker':'Ticker Symbol', 'No. of shares':'Shares', 'Total (GBP)':'Total amount', 'ID':'Order ID',
                                  'Finra fee (GBP)':'Charges and fees', 'Currency (Price / share)': 'Currency Price / share',
                                  'Result (GBP)':'Result'}, inplace=True)
    #trades = data.copy()
    
    trades = pd.concat([trades, data], ignore_index=True)
    
    ## TODO: Temp fix by changing DTG to Jet2, Dartgroup changed their ticker symbol to JET2
    trades['Ticker Symbol'].replace('DTG','JET2', inplace=True)
    
    # SPAC Meger name changes
    trades['Ticker Symbol'].replace('STPK','STEM', inplace=True)
    trades['Ticker Symbol'].replace('NPA','ASTS', inplace=True)
    trades['Ticker Symbol'].replace('NGAC','XOS', inplace=True)
    trades['Ticker Symbol'].replace('SBE','CHPT', inplace=True)
    trades['Ticker Symbol'].replace('CCIV','LCID', inplace=True)
    trades['Ticker Symbol'].replace('FMCI','TTCF', inplace=True)
    trades['Ticker Symbol'].replace('SHLL','HYLN', inplace=True)
    trades['Ticker Symbol'].replace('FIII','ELMS', inplace=True)
    trades['Ticker Symbol'].replace('CFII','VIEW', inplace=True)
    trades['Ticker Symbol'].replace('BFT','PSFE', inplace=True) # Fixes wrong returns for 11/2/2021, note trading212 auto changes tickers in csv
    trades['Ticker Symbol'].replace('BPC','CEG.L', inplace=True)
    trades['Ticker Symbol'].replace('ACTC','PTRA', inplace=True)
    trades['Ticker Symbol'].replace('FTR','FYBR', inplace=True)
    trades['Ticker Symbol'].replace('GHIV','UWMC', inplace=True)
    trades['Ticker Symbol'].replace('FTOC','PAYO', inplace=True)
    trades['Ticker Symbol'].replace('TBA','IS', inplace=True)
    trades['Ticker Symbol'].replace('PDAC','LICY', inplace=True)
    trades['Ticker Symbol'].replace('THCB','MVST', inplace=True)
    trades['Ticker Symbol'].replace('HOL','ASTR', inplace=True)
    trades['Ticker Symbol'].replace('IPV','AEVA', inplace=True)
    trades['Ticker Symbol'].replace('RAAC','BGRY', inplace=True)
    trades['Ticker Symbol'].replace('LGVW','BFLY', inplace=True)
    trades['Ticker Symbol'].replace('GIK','ZEV', inplace=True)
    trades['Ticker Symbol'].replace('PCPL','ETWO', inplace=True)
    trades['Ticker Symbol'].replace('INVU','ETWO', inplace=True)
    trades['Ticker Symbol'].replace('GHVI','MTTR ', inplace=True)
    trades['Ticker Symbol'].replace('VGAC','ME', inplace=True)
    trades['Ticker Symbol'].replace('IPOE','SOFI', inplace=True)
    trades['Ticker Symbol'].replace('FUSE','ML', inplace=True)
    trades['Ticker Symbol'].replace('PACE','NRDY', inplace=True) 

    trades['Ticker Symbol'].replace('AO.','AO', inplace=True)
    
    ## Airbus changed their ticker symbol
    trades['Ticker Symbol'].replace('AIRp', 'AIR', inplace=True)
    
    # Fixes wrong returns for 11/2/2021, note trading212 auto changes tickers in csv
    trades['Ticker Symbol'].replace('SHLL', 'HYLN', inplace=True)
    
    trades.sort_values(['Trading day','Trading time'], inplace=True, ascending=True) # Sort for returns.py
    
    #data.to_csv('trading data export with results.csv', index=False)
    
    # equities = pd.read_sql_table("equities", con=engine, index_col='index')

    trades.rename(columns={'Ticker Symbol':'Ticker'}, inplace=True)

    trades['Sector'] = ''
    trades['Industry'] = ''
    trades['YF_TICKER'] = ''
    trades['Name'] = ''
    
    import time
    start_time = time.time()
    
    def get_sector(r):
        
        print(r.Ticker)

        _, market = get_market(r.ISIN, r.Ticker, old_symbol='')
        
        ticker = get_yf_symbol(market, r['Ticker'])
        r['YF_TICKER'] = ticker
        
        return r
    
    trades = trades.apply(get_sector, axis=1)
    #A = trades[trades['Ticker'] == 'LCID']
    
    categories = {}
    
    errors = []
    
    for x in trades['YF_TICKER'].drop_duplicates():
        
        dic = yf.Ticker(x).info
        
        name = dic['longName'] if 'longName' in dic else x
        
        sector = dic['sector'] if 'sector' in dic else 'Uncategorised'
            
        industry = dic['industry'] if 'industry' in dic else 'Uncategorised'

        print(name) if name is not x or None else print(f'Error: {x}')
        
        if name is x:
            errors.append(name)
        
        categories[x] = {}
        categories[x]['Sector'] = sector
        categories[x]['Industry'] = industry
        categories[x]['Name'] = name
    
    print(errors)
    
    def get_sectors(r):
        
        r.Sector = categories[r.YF_TICKER]['Sector']
        r.Industry = categories[r.YF_TICKER]['Industry']
        r.Name = categories[r.YF_TICKER]['Name']
        print(r.name)
        
        return r
        
    trades = trades.apply(get_sectors, axis=1)
    
    print("--- %s seconds ---" % (time.time() - start_time))
    
    """
    Old script. Code below took 3hrs to run. 
    Refactored to the code above takes 25min
    """
    
    # def get_sector(r):

    #     # market = get_market(equities[equities['INSTRUMENT'] == r['Ticker']]['ISIN'].values[0], r['Ticker'])[1]
    #     #names = difflib.get_close_matches(r['INSTRUMENT'], companies, n=3, cutoff=0.3)
    #     #names = difflib.get_close_matches('IAG SA', companies, n=3, cutoff=0.3)
    #     #market = equities[equities['COMPANY'] == name and equities['INSTRUMENT'] == 'TSLA']  ['MARKET NAME'].values[0]
        
    #     #print(r['INSTRUMENT']) 
    #     #ticker = 'NG'
        
    #     print(r.Ticker)
    #     print(r.name)
        
    #     #market = equities.query('INSTRUMENT==@r.Ticker')['MARKET NAME'].values[0]
        
    #     # try:
    #     #     market = equities.query('ISIN==@r.ISIN and INSTRUMENT==@r.Ticker')['MARKET NAME'].values[0]
    #     # except IndexError:
    #     #     market = equities.query('ISIN==@r.ISIN')['MARKET NAME'].values[0] # Assuming no duplicate tickers
        
    #     _, market = get_market(r.ISIN, r.Ticker, old_symbol='')
        
    #     ticker = get_yf_symbol(market, r['Ticker'])
    #     r['YF_TICKER'] = ticker
        
    #     try:
            
    #         dic = yf.Ticker(ticker).info
            
    #         r.Sector = dic['sector']
                        
    #         r.Industry = dic['industry']
            
    #         print(dic['sector'])
            
    #     except:
    #         r.Sector = 'Uncategorised'
    #         r.Industry = 'Uncategorised'
            
    #         print('Error:')
            
    #         return r
    #     return r
        
    # trades = trades.apply(get_sector, axis=1)
    
    print('done')

    trades.rename(columns={'Ticker':'Ticker Symbol'}, inplace=True)

    #a = trades[trades['Ticker Symbol'] == 'ROKU']
    
    # trades.to_csv('trading data export with results.csv', index=False)

    trades.to_csv('Investment trades.csv', index=False )
    trades.to_sql('trades', engine, if_exists='replace')
    
    print('Orders Updated')
    
    return trades

#t = get_portfolio()
    
def stock_split_adjustment(r):
        
    market = get_market(r['ISIN'], r['Ticker Symbol'])[1] 
    
    ticker = get_yf_symbol(market, r['Ticker Symbol'])
    
    data = yf.Ticker(ticker) # Get stock data
    split_df = data.splits.reset_index() # Extract stock split dates and values
    split = split_df[split_df['Date'] >= r['Trading day']]['Stock Splits'].sum() # 
    
    if split > 0:
        r.Execution_Price = r.Execution_Price/split
    
    return r

def time_frame_returns(timeframe='M'):
    
    returns_df = pd.read_sql_table("returns", con=engine, index_col='index', parse_dates=['Dates'])
    engine.dispose() 

    # Fill missing business days
    idx = pd.bdate_range(min(returns_df.Date), max(returns_df.Date))
    returns_df.set_index('Date', inplace=True)
    #s.index = pd.DatetimeIndex(s.index)
    daily_returns_df = returns_df.reindex(idx, fill_value=0).reset_index().rename(columns={'index':'Date'})
    
    """ 
    Time Frames
    # Yearly Returnsn "Y"

    # Quaterly Returns "Q"
    
    # Tax year "A-APR" https://stackoverflow.com/questions/35339139/where-is-the-documentation-on-pandas-freq-tags
    
    # Monthly Returns "M"
    
    # Weekly Returns "W"
    """
    
    period = daily_returns_df.Date.dt.to_period(timeframe)
    g = daily_returns_df.groupby(period)
    timeframe_returns_df = g.sum()

    return timeframe_returns_df

def get_summary():

    mailbox_list = get_mailbox_list('dividends')
    
    data = []
    
    for item in mailbox_list:
        
    # for num in data[0].split():
        status, body = mail.fetch(item, '(RFC822)')
        email_msg = body[0][1]
    
        #raw_email = email_msg.decode('utf-8')
    
        email_message = email.message_from_bytes(email_msg)
    
        counter = 1
        for part in email_message.walk():
            if part.get_content_maintype() == "multipart":
                continue
            filename = part.get_filename()
            if not filename:
                ext = '.html'
                filename = 'msg-part%08d%s' %(counter, ext)
            
            counter += 1
            
            content_type = part.get_content_type()
            # print(content_type)
            
            summary = ['Date', 'Dividends', 'Opening balance', 'Closing balance']
            
            if "html" in content_type:
                html_ = part.get_payload()
                soup = BeautifulSoup(html_, 'html.parser')
            
                # Doesn't work, half info box missing
                # inv = soup.select('table[class*="info"]')[1] # Use inspect tool in outlook not gmail because classnames don't appear in gmail
                # rows = inv.findChildren(['th', 'tr'])
                
                row_list = []
                
                month = soup.find(text=re.compile('Closed transactions'))
                month = re.sub('[^0-9-]','', month)
                
                row_list.append(month)
                
                for x in summary[1:]:
                    value = float(re.sub('[^0-9.]', '', soup.find(text=re.compile(x)).findNext('td').text))
                    row_list.append(value)
                
                data.append(row_list)
    
    summary_df = pd.DataFrame(data, columns=summary)
    
    now = datetime.now()

    summary_df.loc[len(summary_df)] = [f"{now.year}-{now.strftime('%m')}" , float('NaN'), summary_df.loc[len(summary_df)-1]['Closing balance'], float('NaN')]
     
    summary_df['Target'] = summary_df['Opening balance'] * .05 # Aim for 5% returns a month
    
    summary_df['Goal'] = summary_df['Opening balance'] * .10
    
    monthly_returns_df = time_frame_returns()
    
    monthly_returns_df.index = monthly_returns_df.index.strftime('%Y-%m')
    monthly_returns_df.reset_index(level=0, inplace=True)
    
    #smmary_df = summary_df.merge(monthly_returns_df, on='Date')
    
    summary_df = summary_df.merge(monthly_returns_df, on='Date', how="outer")
    summary_df = summary_df.sort_values(by=['Date'], ignore_index=True)
        
    month_count = pd.to_datetime(summary_df['Date'], errors='coerce').dt.year.value_counts()
    
    summary_df['House Goal'] = [float('NaN') for x in range(month_count.values[0])] + [1000 for x in range(month_count.values[1:].sum())]  
    summary_df['Minimum Goal'] = [100 for x in range(month_count.values[0])] + [200 for x in range(month_count.values[1:].sum())]
    
    summary_df[['Returns', 'Gains', 'Losses']] = summary_df[['Returns', 'Gains', 'Losses']].fillna(0)
            
    summary_df.to_sql('summary', engine, if_exists='replace')
    summary_df.to_csv('Monthly Summary.csv', index=False )
    
    print('Summary done')
    
    return summary_df

# a = get_summary()

def get_buy_sell(ticker):
    
    portfolio = pd.read_sql_table("trades", con=engine, index_col='index', parse_dates=['Trading day'])
    engine.dispose() 

    df = portfolio[portfolio['Ticker Symbol'] == ticker]

    #df['Execution_Price'] = df['Price / share'] # Convert price to original currency
    # df['Execution_Price'] = df['Price'] / df['Exchange rate'] # for emails instead of csv
    
    df['Trading day'] = pd.to_datetime(df['Trading day']) # Match index date format
    
    buys = df[df['Type']=='Buy']
    sells = df[df['Type']=='Sell']
    
    buys = buys.apply(stock_split_adjustment, axis=1)
    sells = sells.apply(stock_split_adjustment, axis=1)
    
    return buys, sells

def formatting(num):
    return round(num, 2)

def get_capital():
    
    mailbox_list = get_mailbox_list('deposits')
    
    total = 0
    
    for item in mailbox_list:
    
    # for num in data[0].split():
        status, body = mail.fetch(item, '(RFC822)')
        email_msg = body[0][1]
    
        #raw_email = email_msg.decode('utf-8')
    
        email_message = email.message_from_bytes(email_msg)
    
        counter = 1
        for part in email_message.walk():
            if part.get_content_maintype() == "multipart":
                continue
            filename = part.get_filename()
            if not filename:
                ext = '.html'
                filename = 'msg-part%08d%s' %(counter, ext)
            
            counter += 1
            
            content_type = part.get_content_type()
            # print(content_type)
            
            if "html" in content_type:
                html_ = part.get_payload()
                soup = BeautifulSoup(html_, 'html.parser')
                
                #a = soup.find("strong").string
    
                table = soup.find_all('table')
    
                a = pd.read_html(str(table))[2][1][0]
                
                a = a.strip('GBP ')
                
                pattern = re.compile(r'\s+')
                a = re.sub(pattern, '', a)
                #print(a)
                total += float(a)
    
    # (9.2 from free shares, 85.2 from rbg rights issue)
    # Add this to match T212 total
    deposits = total + 94.4 
    
    total = 0
    
    mailbox_list = get_mailbox_list('withdraws')
    
    for item in mailbox_list:
    
    # for num in data[0].split():
        status, body = mail.fetch(item, '(RFC822)')
        email_msg = body[0][1]
    
        #raw_email = email_msg.decode('utf-8')
    
        email_message = email.message_from_bytes(email_msg)
    
        counter = 1
        for part in email_message.walk():
            if part.get_content_maintype() == "multipart":
                continue
            filename = part.get_filename()
            if not filename:
                ext = '.html'
                filename = 'msg-part%08d%s' %(counter, ext)
            
            counter += 1
            
            content_type = part.get_content_type()
            # print(content_type)
            
            if "html" in content_type:
                html_ = part.get_payload()
                soup = BeautifulSoup(html_, 'html.parser')
                
                a = soup.find("span").string
                
                a = a.strip('GBP ')
                
                pattern = re.compile(r'\s+')
                a = re.sub(pattern, '', a)
                #print(a)
                total += float(a)
                
    #Add 85.2 for rbg rights issue to match T212
    withdraws = total + 85.2
    
    capital = round(deposits - withdraws, 2)
    
    return capital






