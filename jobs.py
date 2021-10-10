# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 10:28:50 2021

@author: david
"""

from returns import returns
from live_portfolio import get_live_portfolio
from helpers import get_portfolio, get_summary
#from csv_trades import upload

def updates():
    get_portfolio()
    returns()
    #upload()
    get_summary()
    #get_live_portfolio()

#updates()
