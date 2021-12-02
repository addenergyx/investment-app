# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 20:56:03 2020

@author: david
"""

# import requests
# from bs4 import BeautifulSoup
import pandas as pd
#from selenium.common.exceptions import NoSuchElementException
import time
from datetime import date, datetime
import re
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from scraper import get_driver

load_dotenv(verbose=True, override=True)

## Trading 212 hot list
## On 2nd November Trading 212 added a popularity tracker to their site
## Can You Predict Stock Price Movements With Users’ Behaviour data?
## T212 doesn't store historical data so will be scrapping site daily and storing the results in a csv
## Trading 212 uses javascript to load the data so have to use selenium instead of beautifulsoup

timestamp = date.today().strftime('%d-%m-%Y')

db_URI = os.getenv('ElephantSQL_DATABASE_URL')

# db_URI = os.getenv('AWS_DATABASE_URL')
engine = create_engine(db_URI)

## Google authentication

gauth = GoogleAuth()
#gauth.LocalWebserverAuth()
gauth.LoadCredentialsFile("mycreds.txt")
if gauth.credentials is None:
    # Authenticate if they're not there
    gauth.LocalWebserverAuth()
elif gauth.access_token_expired:
    # Refresh them if expired
    gauth.Refresh()
else:
    # Initialize the saved creds
    gauth.Authorize()
gauth.SaveCredentialsFile("mycreds.txt")

drive = GoogleDrive(gauth) 

## Using Long table as it's more flexible for this dataset
## Improve: use database instead of csv
# leaderboard = pd.read_csv('leaderboard.csv', parse_dates=['Date', 'Last_updated'], dayfirst=True) # Date format changes for some observations when reading csv unsure why
# risers = pd.read_csv('risers-03-09-2021.csv', parse_dates=['Date', 'Last_updated'], dayfirst=True)
# fallers = pd.read_csv('fallers-03-09-2021.csv', parse_dates=['Date', 'Last_updated'], dayfirst=True)

# Bad practice to dynamically create variables
leaderboard = pd.read_sql_table("leaderboard", con=engine, index_col='index', parse_dates=['Last_updated'])
risers = pd.read_sql_table("risers", con=engine, index_col='index', parse_dates=['Last_updated'])
fallers = pd.read_sql_table("fallers", con=engine, index_col='index', parse_dates=['Last_updated'])

for df in [leaderboard, risers, fallers]:
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

columns = ['Stock', 'Position', 'Start', 'End', 'Date', 'User_change', 'Percentage_change', 'Last_updated']

## For webapp ##
overall_leaderboard = leaderboard.drop_duplicates('Stock', keep='first').reset_index(drop=True)
overall_leaderboard['Position'] = list(overall_leaderboard.index+1)

## ------------------------- Selenium Setup ------------------------- ##

# def get_driver():
#     options = Options()
#     ua = UserAgent()
#     userAgent = ua.random
#     options.add_argument('user-agent={}'.format(userAgent))
     
#     if WORKING_ENV == 'development':
#         return webdriver.Chrome(ChromeDriverManager().install(), options=options) # automatically use the correct chromedriver by using the webdrive-manager
#     else:
        
#         ## Headless browser - doesn't pop up
#         ## A headless browser is a web browser without a graphical user interface.
#         #options.add_argument("--headless") 
        
#         # https://ivanderevianko.com/2020/01/selenium-chromedriver-for-raspberrypi
#         # https://www.reddit.com/r/selenium/comments/7341wt/success_how_to_run_selenium_chrome_webdriver_on/
#         # https://askubuntu.com/questions/1090142/cronjob-unable-to-find-module-pydub
#         # https://stackoverflow.com/questions/23908319/run-selenium-with-crontab-python
#         return webdriver.Chrome('/usr/lib/chromium-browser/chromedriver', options=options) 

driver = get_driver()

## ------------------------- Leaderboard ------------------------- ##

daily_hotlist = []
    
driver.implicitly_wait(20)        

driver.get('https://www.trading212.com/en/hotlist')

elements = driver.find_elements_by_class_name("pt-popularity-content-item")

def get_last_update(update):
    match = re.search(r'\d{2}/\d{2}/\d{4}, \d{2}:\d{2}:\d{2}', update)
    last_update = datetime.strptime(match.group(), '%d/%m/%Y, %H:%M:%S').strftime('%d-%m-%Y %H:%M:%S')
    return last_update

def upload_to_google_drive(filename):
    fileList = drive.ListFile({'q': "'14Xi6xKvy8T0NnsbQGJipOVsjskBpwbsb' in parents and trashed=false"}).GetList()
    
    for file in fileList:
        if file['title'] == filename:
            file1 = drive.CreateFile({'id': file['id']})
            file1.Trash()
    
    f = drive.CreateFile({'parents': [{'id': '14Xi6xKvy8T0NnsbQGJipOVsjskBpwbsb'}]}) 
    f.SetContentFile(filename) 
    f.Upload() 
    f = None

# Get date from string
update = driver.find_element_by_class_name("pt-footer-notice").text
last_update = get_last_update(update)

for stock in elements:
    print(stock.find_element_by_class_name('pt-name').text)
    daily_hotlist.append([stock.find_element_by_class_name('pt-name').text, 
            stock.find_element_by_class_name('pt-number').text, 
            stock.find_element_by_class_name('pt-holders-count').text, 
            timestamp,
            last_update])

# elements = driver.find_elements_by_class_name("pt-popularity-content-results")
# elements[0].text
# df = pd.read_html(elements[0].text)

## Direct correlation between position and user count so should remove position for model
data = pd.DataFrame(daily_hotlist, columns=['Stock', 'Position', 'User_count', 'Date', 'Last_updated'])
data = data[data['User_count'] != 'USERS']
data['User_count'] = data['User_count'].str.replace(',', '').astype(float)

# data[['Date','Last_updated']] = data[['Date','Last_updated']].apply(pd.to_datetime)
data['Last_updated'] =  pd.to_datetime(data['Last_updated'], dayfirst=True)
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)

df = pd.concat([data, leaderboard], ignore_index=True)

## This script will run several times a day to get as much data as possible. 
## Because positions throughout the day will keep changing. For example when US or UK markets open stocks in their
## region would naturally climb up the table.

#To drop duplicates based on multiple columns:
#https://stackoverflow.com/questions/12497402/python-pandas-remove-duplicates-by-columns-a-keeping-the-row-with-the-highest
complete_df = df.sort_values('User_count', ascending=False).drop_duplicates(['Stock','Date'], keep='first').reset_index(drop=True) #.sort_index()

## Fixing positions column

#complete_df['Last_updated'] = pd.to_datetime(complete_df.Last_updated)
complete_df['Date'] = pd.to_datetime(complete_df.Date)
complete_df = complete_df.sort_values(['Date', 'User_count'], ascending=[True, False])

comp = pd.DataFrame()

for d in complete_df['Date'].unique():
    temp_df = complete_df[complete_df['Date'] == d].reset_index(drop=True)
    temp_df['Position'] = list(temp_df.index+1)
    comp = comp.append(temp_df, ignore_index=True)

complete_df = comp.copy()
complete_df['Date'] = complete_df['Date'].dt.strftime('%d/%m/%Y')

complete_df.to_csv(f'leaderboard-{timestamp}.csv', index=False)

complete_df.to_sql('leaderboard', engine, if_exists='replace')

upload_to_google_drive(f'leaderboard-{timestamp}.csv')

## Position in dataset
# complete_df = complete_df.sort_values('User_count', ascending=False)
## Reset positions, use list() so it's int instead of string
# complete_df['Position'] = list(complete_df.index+1)

## ------------------------- Daily Risers/Fallers ------------------------- ##

def user_data(xpath, file, historical_df):
    
    daily = []
    
    driver.find_element_by_xpath(xpath).click()
    
    # 1D chart stopped working so using 8H chart
    #driver.find_element_by_xpath('/html/body/div[1]/section[2]/div/div/div[2]/div[2]/div[3]').click()

    time.sleep(5) # Pause for page to load
    
    update = driver.find_element_by_class_name("pt-footer-notice").text
    last_update = get_last_update(update)
    
    elements = driver.find_elements_by_class_name("pt-popularity-content-item")
    
    for stock in elements[1:]:
        print(stock.find_element_by_class_name('pt-name').text)
        daily.append([stock.find_element_by_class_name('pt-name').text, 
                stock.find_element_by_class_name('pt-number').text, 
                stock.find_element_by_class_name('pt-change').text,
                stock.find_element_by_class_name('pt-start').text,
                stock.find_element_by_class_name('pt-end').text,
                timestamp,
                last_update])
    
    data = pd.DataFrame(daily, columns=['Stock', 'Position', 'Change','Start', 'End', 'Date', 'Last_updated'])
    
    data[['User_change','Percentage_change']] = data['Change'].str.split(' ', expand=True)
    data = data.drop('Change', 1) # inplace=True not working
    #data = data[data['User_count'] != 'USERS']
    data['Percentage_change'] = data['Percentage_change'].str.strip('()').str.strip('%').astype(float) # Remove brackets
    data['User_change'] = data['User_change'].str.replace(',', '').astype(float)
    data['Start'] = data['Start'].str.replace(',', '').astype(float)
    data['End'] = data['End'].str.replace(',', '').astype(float)
    #data[['Date','Last_updated']] = data[['Date','Last_updated']].apply(pd.to_datetime)
    data['Last_updated'] =  pd.to_datetime(data['Last_updated'], dayfirst=True)
    data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
    
    data = data[columns]
    
    df = pd.concat([data, historical_df], ignore_index=True)
    
    ## To fix fallers position column make column abs then times by -1 after reorder
    
    complete_df = df.sort_values('User_change', ascending=False).drop_duplicates(['Stock','Date'], keep='first').reset_index(drop=True)
    
    complete_df['User_change'] = complete_df['User_change'].abs()
    
    #complete_df['Date'] = pd.to_datetime(complete_df.Date)
    
    #complete_df[['Date','Last_updated']] = complete_df[['Date','Last_updated']].apply(pd.to_datetime)

    complete_df = complete_df.sort_values(['Date', 'User_change'], ascending=[True, False])

    comp = pd.DataFrame()
    
    for d in complete_df['Date'].unique():
        temp_df = complete_df[complete_df['Date'] == d].reset_index(drop=True)
        temp_df['Position'] = list(temp_df.index+1)
        comp = comp.append(temp_df, ignore_index=True)
    
    complete_df = comp.copy()
    
    #complete_df['User_change'] = complete_df['User_change'] * -1
    complete_df['Date'] = complete_df['Date'].dt.strftime('%d/%m/%Y')
    complete_df.to_csv(file, index=False)
    
    upload_to_google_drive(file)

    return complete_df

risers_df = user_data('//*[@id="__next"]/main/section/div/div/div/div[1]/div/div/div[2]/div[2]/div', f'risers-{timestamp}.csv', risers)
fallers_df = user_data('//*[@id="__next"]/main/section/div/div/div/div[1]/div/div/div[2]/div[3]/div', f'fallers-{timestamp}.csv', fallers)

risers_df.to_sql('risers', engine, if_exists='replace')
fallers_df.to_sql('fallers', engine, if_exists='replace')

## ------------------------- Selenium Shutdown ------------------------- ##

driver.close()
driver.quit()
