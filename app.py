# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 07:02:46 2020

@author: david
"""

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
#import dash_daq as daq
from dash.dependencies import Input, Output, State
#import plotly.graph_objs as go
from visuals import *
#performance_chart, ml_model, period_chart, goal_chart, profit_loss_chart, cumsum_chart, dividend_chart, return_treemap, convert_to_gbp, get_holdings
from components import Fab
import os
from sqlalchemy import create_engine
from jobs import updates
import plotly.express as px
#from live_portfolio import get_live_portfolio
from helpers import get_capital
from visuals import day_treemap, return_treemap
from datetime import time
import time as t

db_URI = os.getenv('AWS_DATABASE_URL')
engine = create_engine(db_URI)

external_stylesheets =['https://codepen.io/IvanNieto/pen/bRPJyb.css', dbc.themes.BOOTSTRAP,
                       'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.13.0/css/all.min.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets, assets_folder='./assets/investment_assets',
            )

server = app.server

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        {%css%}
        {%favicon%}
    </head>
    <body>
        <div></div>
        {%app_entry%}
        <footer> 
          {%config%} 
          {%scripts%} 
          {%renderer%}
        </footer>
    </body>
</html>
'''

portfolio = pd.read_sql_table("trades", con=engine, index_col='index', parse_dates=['Trading day']).sort_values(['Trading day','Trading time'], ascending=False)
equities = pd.read_sql_table("equities", con=engine, index_col='index')

def stats():
    summary_df = pd.read_sql_table("summary", con=engine, index_col='index')

    # balance = summary_df['Closing balance'].iloc[-2]
    # balance = "{:.2f}".format(round(balance, 2))
    balance = "{:.2f}".format(get_capital())
    month_profit = round(summary_df['Returns'].iloc[-1], 2)
    total_profit = "{:.2f}".format(round(summary_df['Returns'].cumsum().iloc[-1], 2))
    
    if month_profit > 0:
        style = {'color': 'green'}
        month = f'£{"{:.2f}".format(month_profit)}'
    elif month_profit < 0:
        style = {'color': 'red'}
        month = f'-£{"{:.2f}".format(abs(month_profit))}'
    else:
        style = {'color': 'white'}
        month = f'£{"{:.2f}".format(month_profit)}'
    
    return f'£{balance}', f'£{total_profit}', f'{month}', style

summary = stats()

# if time(hour=9, minute=0) < datetime.now().time() < time(hour=14, minute=30) or time(hour=21) < datetime.now().time() < time(hour=22):
#     interval = 240000
# else:
#     interval = 120000

# def update_news(ticker):
#     # Init
#     newsapi = NewsApiClient(api_key=os.getenv('NEWS_API_KEY'))
    
#     # /v2/top-headlines
#     top_headlines = newsapi.get_top_headlines(q=ticker,
#                                               #sources='google-news',
#                                               language='en',
#                                               #country='gb'
#                                               )
    
#     articles = top_headlines['articles']
    
#     titles = []
#     urls = []
#     for a in articles:
#         titles.append(a['title'])
#         urls.append(a['url'])
    
#     d = {'Title':titles,'Url':urls}
    
#     news_df = pd.DataFrame(d)

#     return news_df

# df = update_news('BFT')

app.title = 'Investments'

colours = {
            'remaining':'#1FDBB5',
            'principal':'#F54784',
            'interest':'#FFAC51'
          }

graph_card = [
                html.Div([            
                    dcc.Loading(
                        dcc.Graph(id='graphy'#, figure=performance_chart('TSLA')
                                  )
                    )
                ], id='graph-block', hidden=False)
             ]

profit_card = [
                html.Div([            
                    dcc.Loading(
                        dcc.Graph(id='profit-graph')
                    )
                ], id='profit-block', hidden=False)
             ]

map_card = [
                html.Div([            
                   #dcc.Loading(
                        dcc.Graph(figure=day_treemap(), id='treemap-graph'#, figure=day_treemap()
                                  )
                    #)
                ], id='treemap-block', hidden=False)
             ]

map_cardb = [
                html.Div([            
                   #dcc.Loading(
                        dcc.Graph(figure=return_treemap(), id='returnmap-graph'#, figure=day_treemap()
                                  )
                    #)
                ], id='returnmap-block', hidden=False)
             ]

stats_card = [
                dbc.CardBody(
                    [
                        html.H2('Results'),
                        html.P("Capital"),
                        html.Strong(html.P(summary[0], id='Capital', className='result')),
                        html.P('Total Realised P/L'),
                        html.Strong(html.P(summary[1], id='total-profit', className='result')),
                        html.P("This Month's Realised P/L"),
                        html.Strong(html.P(summary[2], id='monthly-profit', className='result', style=summary[3])),
                        # html.P('Floating P/L'),
                        # html.Strong(html.P(id='floats', className='result'))
                    ], className='stats'
                )
             ]

trades_card = [
                html.Div([            
                    dcc.Loading(
                        dcc.Graph(figure=trades_chart(), id='trades-graph')
                    )
                ], id='trades-block', hidden=False)
             ]

performance_main = [
                  dbc.Row(
                      [
                          dbc.Col(html.Div(profit_card), width=12),
                      ], className = 'data-row'
                  ),
                      
                  dbc.Row(
                      [
                          dcc.RadioItems(
                           options=[
                               {'label': 'Red-Blue (Colourblind safe)', 'value': 'RdBu'},
                               {'label': 'Red-Green', 'value': 'RdYlGn'},
                           ],
                           value='RdBu',
                           labelStyle={'display': 'inline-block'},
                           style={'color':'white'},
                           id='colours'
                           ),
                          dbc.Col(html.Div(map_card), width=12),
                      ], className = 'data-row'
                  ),
                  
                   dbc.Row(
                       [
                           dbc.Col(html.Div(map_cardb), width=12),
                       ], className = 'data-row'
                   ),
                                                 
                  dbc.Row(
                      [
                          html.H6('Please note can take a while to load'),
                          dbc.Col(html.Div(graph_card), width=12),
                      ], className = 'data-row'
                  ),
                  
                  dbc.Row(
                      [
                          dbc.Col(dbc.Card(id='full-data-card', className='summary-card'), width=12, lg=8),
                          dbc.Col(dbc.Card( 
                              # children=[                 
                              #     dcc.Loading(
                              #         dcc.Graph(id='ml-graph'#, figure=ml_model()
                              #                   )
                              #     )
                              # ],
                              #html.P(id='advice'), 
                              #id='side-data-card', 
                              className='summary-card justify-content-center align-self-center'), width=12, lg=4),
                          # dbc.Col(dbc.Card(stats_card, id='side-data-card', className='summary-card'), width=12, lg=4),
                          # #dbc.Col(dbc.Card(aaa, id='test', className='summary-card'), width=12, lg=4),
                      ], className = 'data-row'
                  ),
                                       
                  dbc.Row(
                      [
                          dbc.Col(width=12),
                      ], className = 'data-row'
                  ),
                                       
                  dcc.Interval(id="stats-interval", n_intervals=0, interval=600000),
                  dcc.Interval(id="map-interval", n_intervals=0, interval=60000), # TODO: change interval based on time of day, 240s in pre market, 120 in market hours 
                  dcc.Interval(id="map-intervalb", n_intervals=0, interval=60000),
                  dcc.Interval(id="dropdown-interval", n_intervals=0, interval=720000),
                  html.Div(id='container-button-basic', hidden=True)
                 
              ]

insight_main = [
                  dbc.Row(
                      [
                          dbc.Col(html.Div(trades_card), width=12),
                      ], className = 'data-row'
                  ),
                  
                  dbc.Row(
                      [
                            html.H4('Treemap of trades/returns aggregated by Sector and Industry', style={'color':'white'},),
                            html.Br(),
                            dcc.RadioItems(
                            options=[
                                {'label': 'Red-Blue (Colourblind safe)', 'value': 'RdBu'},
                                {'label': 'Red-Green', 'value': 'RdYlGn'},
                            ],
                            value='RdBu',
                            labelStyle={'display': 'inline-block'},
                            style={'color':'white'},
                            id='colours'
                            ),
                            html.Div(style={'margin':'20px'}),
                            dbc.Col(dcc.Graph(id='vis4', figure=vis4()), width=12),
                      ], className = 'data-row'
                  ),
                  
            ]

ml_main = []

# Returns Top cell bar for header area
# def get_top_bar_cell(cellTitle, cellValue):
#     return html.Div(
#         #className="two-col",
#         children=[
#             html.P(className="p-top-bar", children=cellTitle),
#             html.P(id=cellTitle, className="display-none", children=cellValue),
#             html.P(children=human_format(cellValue)),
#         ],
#     )


# def rows(row, df):
#     pp = []
#     for x in range(0, len(df.columns)):
#         pp.append(html.Td(
#             html.P(row.values[x], className='balances'), **{'data-label': 'Month'}
#         ))
#     return pp
        
# def build_table(df):
#     return html.Div(
#     [
#         #Header
#         html.Table([html.Tr([html.Th(col) for col in df.columns])]
#         +
#         #body
#         [
#             html.Tr(
#                 [
#                     rows(row, df)
  
#                 ]) for i, row in df.iterrows()], className="hover-table amortization-table"
#         ), 
#     ], className='table-block', #style={"height": "100%", "overflowY": "scroll", "overflowX": "hidden"}, #className='large-2'
#     )

def build_table(df):
    return html.Div(
    [
        #Header
        html.Table([html.Tr([html.Th(col) for col in df.columns[:7]])]
        +
        #body
        [
            html.Tr(
                [
                    html.Td(
                        html.P(row.values[0], className='balances'), **{'data-label': 'Month'}
                    ),
                    html.Td(
                        html.P(row.values[1], className='balances'), **{'data-label': 'Payment'} 
                    ),
                    html.Td(
                        html.P(row.values[2], className='balances'), className='amount', **{'data-label': 'Principal'}
                    ),
                    html.Td(
                        html.P((row.values[3]), className='balances'), className='amount', **{'data-label': 'Interest'}
                    ),
                    html.Td(
                        html.P((row.values[4]), className='balances'), className='amount', **{'data-label': 'Total Interest'} 
                    ),
                    html.Td(
                        html.P((row.values[5].strftime("%d/%m/%Y")), className='balances'), className='amount', **{'data-label': 'Balance'} 
                    ),
                ]) for i, row in df.iterrows()], className="hover-table amortization-table"
        ), 
    ], className='table-block', #style={"height": "100%", "overflowY": "scroll", "overflowX": "hidden"}, #className='large-2'
    )

def build_card(title, colour):
    return html.Div(
        [
            dbc.Row(
                [
                    #dbc.Col(html.Span(className='fas fa-money-bill-wave icon', style={'color':colour}), className='d-flex justify-content-center icon-container', width=3), 
                    dbc.Col(
                        [
                            html.P(title.capitalize(), className='money')
                        ], className='d-flex justify-content-center text-center')
                ]
            ),
            dbc.Row(
                [
                    #dbc.Col(width=3),
                    dbc.Col(html.P(id=f'{title}-value'), className='d-flex justify-content-center text-center')
                ]
            ),
        ]
    )

app.config.suppress_callback_exceptions = True

def company(x):
    try:
        company = equities[equities['INSTRUMENT'] == x]['COMPANY'].values[0]
        dic = {'label': f'{company} ({x})', 'value': x}
    except:
        dic = {'label': str(x), 'value': x}
    return dic

tickers = [company(x) for x in portfolio['Ticker Symbol'].drop_duplicates()]

# tickers = []
# for x in portfolio['Ticker Symbol'].drop_duplicates():
#     try:
#         company = equities[equities['INSTRUMENT'] == x]['COMPANY'].values[0]
#         tickers.append({'label': f'{company} ({x})', 'value': x})
#     except:
#         tickers.append({'label': str(x), 'value': x})

#tickers = [{'label':str(x), 'value': x} for x in portfolio['Ticker Symbol'].drop_duplicates()]

charts = [{'label':str(x), 'value': x} for x in ['Goals', 'Monthly', 'Dividends', 'Cumulative', 'Profit/Loss', 'Daily', 'Weekly', 'Yearly', 'Quarterly', 'Fiscal Year']]
maps = [{'label':str(x), 'value': x} for x in ['Day', 'Portfolio']]

body = html.Div(
            [
              dbc.Row(
                    [
                        ## Side Panel
                        dbc.Col(
                           [
                               html.H1('Investments', style={'text-align':'center'}),
                              
                               html.Div(
                                   [
                                        html.Div(dbc.Row([ dbc.Col(dbc.Card(stats_card, className='summary-card stats-card',  ))]), hidden=False, id='statss' )
                                        
                                   ], id='user-inputs'
                               ),
                              
                              #html.Div(style={'margin':'200px'}),
                              
                             html.Div(
                                   [
                                  dcc.Dropdown(
                                    id='chart-dropdown',
                                    options=charts,
                                    value='Goals',
                                    clearable=False,
                                    style={'margin-top':'100px'}
                                  ),
                              ]),
                            
                              html.Div(
                                    [
                                   dcc.Dropdown(
                                     id='map-dropdown',
                                     options=maps,
                                     value='Day',
                                     clearable=False,
                                     style={'margin-top':'50px'}
                                   ),
                               ]), 
                            
                              html.Div(
                                   [
                                  dcc.Dropdown(
                                    id='ticker-dropdown',
                                    options=tickers,
                                    value=tickers[0]['value'],
                                    searchable=True,
                                    style={'margin-top':'50px'}
                                  ),
                              ]),
                              
                              # html.Div(
                              #      [
                              #     html.Button(
                              #       'Update All',
                              #       id='update-all-btn',
                              #       style={'margin-top':'50px'}
                              #     ),
                              #     html.Button(
                              #       'Update Portfolio',
                              #       id='update-portfolio-btn',
                              #       style={'margin-top':'50px'}
                              #     ),
                              # ]),
                              
                           ], id='side-panel', width=12, lg=2
                        ),
                      
                     ## Main panel
                     dbc.Col(
                         [
                             dbc.Row(
                                  [
                                      dbc.Col(html.Button('Performance', id='Performance-btn', className = 'button button1'), className = 'center-stuff', width=4),
                                      dbc.Col(html.Button('Insight', id='Insight-btn', className = 'button button2'), className = 'center-stuff', width=4),
                                      dbc.Col(html.Button('Machine Learning', id='Machine-Learning-btn', className = 'button button3'), className = 'center-stuff', width=4),
                                  ], className = 'data-row'
                             ),
                          
                            html.Div(performance_main, id='main-panel-content')
                            
                         ], width=12, lg=9, id='main-panel'),
                    
         
                ], no_gutters=True),
             ])

@app.callback([Output("Capital", "children"), Output("total-profit", "children"), 
                #Output("total-interest", "children"),  
                Output("monthly-profit", "children"), Output("monthly-profit", "style")], 
              [Input("stats-interval", "n_intervals")])
def event_cb(data):
    return stats()

# @app.callback(
#     Output('container-button-basic', 'children'),
#     [Input('update-portfolio-btn', 'n_clicks'), Input('update-all-btn', 'n_clicks')])
# def update_output(btn1, btn2):
    
#     changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
#     print([p['prop_id'] for p in dash.callback_context.triggered])
    
#     if 'update-portfolio-btn' in changed_id:
#         return get_live_portfolio()
#     elif 'update-all-btn' in changed_id:
#         return updates()
#     return ''

@app.callback(
    Output('main-panel-content', 'children'),
    [Input('Performance-btn', 'n_clicks'), Input('Insight-btn', 'n_clicks'), Input('Machine-Learning-btn', 'n_clicks')])
def update_output(btn1, btn2, btn3):
    
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    print([p['prop_id'] for p in dash.callback_context.triggered])
    
    if 'Performance-btn' in changed_id:
        return performance_main
    elif 'Insight-btn' in changed_id:
        return insight_main
    elif 'Machine-Learning-btn' in changed_id:
        return ml_main
    return performance_main

@app.callback(
    [Output('ticker-dropdown', 'options'), Output('full-data-card','children')],
    [Input('dropdown-interval', 'n_intervals')])
def update_tickers(n_clicks):
    
    portfolio = pd.read_sql_table("trades", con=engine, index_col='index', parse_dates=['Trading day']).sort_values(['Trading day','Trading time'], ascending=False)
    tickers = [company(x) for x in portfolio['Ticker Symbol'].drop_duplicates()]
    
    # portfolio = portfolio[['Ticker Symbol', 'Type', 'Shares', 'Price', 'Total amount', 'Trading day']]
    portfolio = portfolio[['Ticker Symbol', 'Type', 'Shares', 'Price', 'Total cost', 'Trading day']]

    return tickers, build_table(portfolio)
    
@app.callback(Output('graphy','figure'), 
              [Input("ticker-dropdown", "value")])
def event_a(ticker):
    return performance_chart(ticker)

# @app.callback(Output('treemap-graph','figure'), 
#               [Input("colours", "value")])
# def event_colour(colour):
#     return day_treemap(colour), return_treemap(colour)

@app.callback(
    [Output('treemap-graph','figure'), Output('returnmap-graph','figure')],
    [Input("map-interval", "n_intervals"), Input("colours", "value")])
def event_o(ticks, colour):
    return day_treemap(colour), return_treemap(colour)

@app.callback(Output('profit-graph','figure'), 
              [Input("chart-dropdown", "value")])
def event_b(chart):
        
    options = {'Goals' : goal_chart,
           'Monthly' : period_chart,
           'Dividends' : dividend_chart,
           'Cumulative' : cumsum_chart,
           'Profit/Loss' : profit_loss_chart,
           'Daily' : period_chart,
           'Weekly' : period_chart,
           'Yearly': period_chart,
           'Quarterly' : period_chart,
           'Fiscal Year': period_chart,
    }
    
    # Seperated from options dict because running all functions takes time
    if chart == 'Monthly':
        param = 'M'
        return options[chart](param)
    elif chart == 'Daily':
        param = 'D'
        return options[chart](param)
    elif chart == 'Weekly':
        param = 'W'
        return options[chart](param)
    elif chart == 'Yearly':
        param = 'Y'
        return options[chart](param)
    elif chart == 'Quarterly':
        param = 'Q'
        return options[chart](param)
    elif chart == 'Fiscal Year':
        param = 'A-APR'
        return options[chart](param)
        
    fig = options[chart]()
    
    return fig

@app.callback(Output('vis4','figure'), 
              [Input("colours", "value")])
def event_g(colour):
    return vis4(colour)

def Homepage():
    return html.Div([
            body,
            #button()
            #html.Div(Fab()),
        ], id='background')

"""
Set layout to be a function so that for each new page load                                                                                                       
the layout is re-created with the current data, otherwise they will see                                                                                                     
data that was generated when the Dash app was first initialised
"""     

#app.scripts.config.serve_locally=True
app.layout = Homepage()

if __name__ == '__main__':
    #app.run_server(debug=True, threaded=True, use_reloader=False) 
    #app.run_server(debug=True, use_reloader=False, processes=4) # https://community.plotly.com/t/keep-updating-redrawing-graph-while-function-runs/8744
    app.run_server()

















