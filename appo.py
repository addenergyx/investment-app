# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 11:12:58 2021

@author: david
"""

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import dash_daq as daq
import plotly.graph_objs as go
import plotly.express as px
import yfinance as yf
from dash.dependencies import Input, Output
import pandas as pd
from datetime import datetime
from pandas_datareader import data as web
from plotly.subplots import make_subplots
import calendar
#pd.options.mode.chained_assignment = None  # default='warn'

def vis1(filters='Returns'):
    seasonality = pd.read_csv("https://raw.githubusercontent.com/addenergyx/datasets/main/trading%20data%20export%20with%20results.csv", parse_dates=['Trading day'], dayfirst=True)

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

def vis2():
    seasonality_df = pd.read_csv("https://raw.githubusercontent.com/addenergyx/datasets/main/animated.csv")
    
    fig = px.scatter(seasonality_df, x="Trading_time", y="Trading day",
    	         size="Count", color="pct_buy", color_continuous_scale='RdBu', color_continuous_midpoint=0.5,
                     hover_name="Count", animation_frame='week',
                 size_max=50
                     )
    fig.update_yaxes(title='Weekday', categoryorder='array', categoryarray= ['Friday', 'Thursday', 'Wednesday', 'Tuesday', 'Monday'])
    fig.update_xaxes(categoryorder='array', categoryarray= [f'0{x}:00' if x < 10 else f'{x}:00' for x in range(7,21)])
    #fig.update_layout(transition = {'duration': 50000}, margin=dict(l=0,r=0,b=0,t=0))
    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 1000
    
    return fig
    
def vis3():
    
    trades = pd.read_csv("https://raw.githubusercontent.com/addenergyx/datasets/main/trading%20data%20export%20with%20results.csv", parse_dates=['Trading day'], dayfirst=True)
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
    fig = px.bar(days, x='Trading day', y='Trading Activity')
    #plot(fig)
    
    fig = go.Figure(data=[
        go.Bar(name='Buy Executions', x=days['Trading day'], y=days['Buy Execution']),
        go.Bar(name='Sell Executions', x=days['Trading day'], y=days['Sell Execution'])
    ])
    # Change the bar mode
    fig.update_layout(barmode='stack', bargap=0)
    
    return fig
    
def vis4(colour='RdBu'):
    trades = pd.read_csv("https://raw.githubusercontent.com/addenergyx/datasets/main/trading%20data%20export%20with%20results.csv", parse_dates=['Trading day'])
    
    trades['Result'] = trades['Result'].fillna(0.0)
    
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
    
def stock_split_adjustment(r):
    
    #portfolio = pd.read_csv("https://raw.githubusercontent.com/addenergyx/datasets/main/trading%20data%20export%20with%20results.csv", parse_dates=['Trading day'], dayfirst=True)
    portfolio = pd.read_csv("trading data export with results.csv", parse_dates=['Trading day'], dayfirst=True)

    ticker = portfolio[portfolio['Ticker'] == r.Ticker]['YF_TICKER'].values[0]    

    aapl = yf.Ticker(ticker)
    split_df = aapl.splits.reset_index()
    split = split_df[split_df['Date'] > r['Trading day']]['Stock Splits'].sum()
    
    if split > 0:
        r.Execution_Price = r.Execution_Price/split
    
    return r

def get_buy_sell(ticker):
    
    #portfolio = pd.read_csv("https://raw.githubusercontent.com/addenergyx/datasets/main/trading%20data%20export%20with%20results.csv", parse_dates=['Trading day'], dayfirst=True)
    portfolio = pd.read_csv("trading data export with results.csv", parse_dates=['Trading day'], dayfirst=True)

    df = portfolio[portfolio['Ticker'] == ticker]

    #df['Execution_Price'] = df['Price / share'] # Convert price to original currency
    # df['Execution_Price'] = df['Price'] / df['Exchange rate'] # for emails instead of csv
    
    df['Trading day'] = pd.to_datetime(df['Trading day']) # Match index date format
    
    buys = df[df['Type']=='Buy']
    sells = df[df['Type']=='Sell']
    
    buys = buys.apply(stock_split_adjustment, axis=1)
    sells = sells.apply(stock_split_adjustment, axis=1)
    
    return buys, sells

def performance_chart(ticker='TSLA'):

    #ticker = 'NG'
    portfolio = pd.read_csv("trading data export with results.csv", parse_dates=['Trading day'], dayfirst=True)
    #portfolio = pd.read_csv("https://raw.githubusercontent.com/addenergyx/datasets/main/trading%20data%20export%20with%20results.csv", parse_dates=['Trading day'], dayfirst=True)

    
    if ticker=='TSLA':
        buys = pd.read_csv("buys.csv")
        sells = pd.read_csv("sells.csv")
        index = pd.read_csv("tesla.csv")  
           
    else:
        buys, sells = get_buy_sell(ticker) #Need to make this quicker
    
        start = datetime(2020, 2, 7)
        end = datetime.now()    
    
        yf_symbol = portfolio[portfolio['Ticker'] == ticker]['YF_TICKER'].values[0]    
    
        index = web.DataReader(yf_symbol, 'yahoo', start, end)
        index = index.reset_index()
    
        index['Midpoint'] = (index['High'] + index['Low']) / 2
    
    buy_target = []
    sell_target = []

    for i, row in buys.iterrows():
        mid = index[index['Date'] == row['Trading day']]['Midpoint'].values[0]

        if row['Execution_Price'] < mid:
            buy_target.append(1)
        else:
            buy_target.append(0)

    for i, row in sells.iterrows():
        mid = index[index['Date'] == row['Trading day']]['Midpoint'].values[0]

        if row['Execution_Price'] > mid:
            sell_target.append(1)
        else:
            sell_target.append(0)

    buys['Target'] = buy_target
    sells['Target'] = sell_target

    ## Discrete color graph

    main_fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig = go.Figure(data=[go.Candlestick(x=index['Date'],
                    open=index['Open'],
                    high=index['High'],
                    low=index['Low'],
                    close=index['Adj Close'],
                    name='Stock',
                    #increasing_line_color= 'blue', decreasing_line_color= 'red'
                    )])

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

    #main_fig.show()
    
    return main_fig #, percentage

portfolio = pd.read_csv("https://raw.githubusercontent.com/addenergyx/datasets/main/trading%20data%20export%20with%20results.csv", parse_dates=['Trading day'], dayfirst=True)
portfolio['count'] = portfolio.groupby('Name')['Name'].transform('count')
portfolio.sort_values(by='count', ascending=False, inplace=True)

#portfolio.sort_values(by='Trading day', ascending=False, inplace=True)

def company(x):
#     try:
    company = portfolio[portfolio['Ticker'] == x]['Name'].values[0]
    dic = {'label': f'{company} ({x})', 'value': x}
#     except:
#         dic = {'label': str(x), 'value': x}
    return dic

tickers = [company(x) for x in portfolio['Ticker'].drop_duplicates()]

# Build App
app = dash.Dash(__name__)

server = app.server

app.layout = html.Div([
    
    html.H1("Visualisation 1 (V1)"),
    html.H3("Trade volumes by day and hour of week"),
    dcc.RadioItems(
    options=[
        {'label': 'Profit/Loss', 'value': 'Returns'},
        {'label': 'Buy/Sell', 'value': 'pct_buy'},
    ],
    value='Returns',
    labelStyle={'display': 'inline-block'},
    id='filters'
    ),
    dcc.Graph(id='vis1', figure=vis1()),
    
    html.H1("Visualisation 2 (V2)"),
    html.H3('Trade volumes by day and hour of week aggregated by week (interactive)'),
    dcc.Loading(
        dcc.Graph(id='vis2', figure=vis2()),
    ),
    
    html.H1("Visualisation 3 (V3)"),
    html.H3('Trade volumes over the year'),
    dcc.Graph(id='vis3', figure=vis3()),
    
    html.H1("Visualisation 4 (V4)"),
    html.H3('Treemap of trades aggregated by Sector and Industry'),
    dcc.RadioItems(
    options=[
        {'label': 'Red-Blue (Colourblind safe)', 'value': 'RdBu'},
        {'label': 'Red-Green', 'value': 'RdYlGn'},
    ],
    value='RdBu',
    labelStyle={'display': 'inline-block'},
    id='colours'
    ),
    html.Div(style={'margin':'20px'}),
    dcc.Graph(id='vis4', figure=vis4()),
    
    html.H1("Visualisation 5 (V5)"),
    html.H3('OHLC + Volume and successful/unsuccessful trades (interactive).'),
    html.H6('Please note can take a while to load'),
    dcc.Loading(
        dcc.Graph(id='graphy'),
    ),
        
    html.Div(
            [
          dcc.Dropdown(
            id='ticker-dropdown',
            options=tickers,
            value=tickers[0]['value'],
            searchable=True,
            style={'margin-bottom':'50px'}
          ),
      ]),
    html.Div(style={'margin':'20px'}),
    
    ])
# Define callback to update graph

@app.callback(Output('graphy','figure'), 
              [Input("ticker-dropdown", "value")])
def event_a(ticker):
    return performance_chart(ticker)

@app.callback(Output('vis4','figure'), 
              [Input("colours", "value")])
def event_b(colour):
    return vis4(colour)

@app.callback(Output('vis1','figure'), 
              [Input("filters", "value")])
def event_c(choice):
    return vis1(choice)

# Run app and display result inline in the notebook
if __name__ == '__main__':
    app.run_server() 



































