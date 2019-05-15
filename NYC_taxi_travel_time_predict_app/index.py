#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ericyuan
"""
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from app import app

import datetime as dt

from module.feature_eng import allfe
import xgboost as xgb
import pandas as pd
from sklearn.externals import joblib

import plotly.graph_objs as go

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

logopath = 'https://upload.wikimedia.org/wikipedia/en/thumb/f/f1/Columbia_University_shield.svg/1200px-Columbia_University_shield.svg.png'
index_page = html.Div([
    # Navbar
    dbc.Navbar([html.A(dbc.Row(
                    [dbc.Col(html.Img(src=logopath, height="30px")),
                     dbc.Col(dbc.NavbarBrand("NYC Taxi Travel Time Prediction", className="ml-2"))],
                     align="center",
                     no_gutters=True),
                     href="index")],
                color="dark",
                dark=True),
    # Input
    html.Div([
            html.P('Pickup time', style = {'display':'inline-block',\
                                           'padding': '0px 5px 0px 5px',\
                                           'font-weight': 'bold'}),\
            dcc.Input(id='input_1', type='text', value='2016-01-01 15:12',\
                      style = {'padding': '0px 0px 0px 0px',\
                               "max-width": "150px"}),\
            html.P('Pickup latitude', style = {'display':'inline-block',\
                                               'padding': '0px 10px 0px 10px',\
                                               'font-weight': 'bold'}),\
            dcc.Input(id='input_2', type='text', value='',\
                      style = {'padding': '0px 0px 0px 0px',\
                               "max-width": "70px"}),\
            html.P('Pickup longtitude', style = {'display':'inline-block', \
                                                 'padding': '0px 10px 0px 10px',\
                                                 'font-weight': 'bold'}),\
            dcc.Input(id='input_3', type='text', value='',\
                      style = {'padding': '0px 0px 0px 0px',\
                               "max-width": "70px"}),\
            html.P('Dropoff latitude', style = {'display':'inline-block', \
                                                 'padding': '0px 10px 0px 10px',\
                                                 'font-weight': 'bold'}),\
            dcc.Input(id='input_4', type='text', value='',\
                      style = {'padding': '0px 0px 0px 0px',\
                               "max-width": "70px"}),
            html.P('Dropoff longtitude', style = {'display':'inline-block', \
                                                 'padding': '0px 10px 0px 10px',\
                                                 'font-weight': 'bold'}),\
            dcc.Input(id='input_5', type='text', value='',\
                      style = {'padding': '0px 0px 0px 0px',\
                               "max-width": "70px"}),\
            html.Button("Predict", \
                        id = 'btn_1', \
                        n_clicks_timestamp = 0, \
                        style = {'margin': '0px 20px 0px 30px'}),\
            html.P(),\
            html.P('Result:', style = {'display':'inline-block', \
                                     'padding': '0px 5px 0px 5px',\
                                     'font-weight': 'bold'}),\
            html.Div(id = 'output-container-button', \
                     children = 'Enter your information and press predict',\
                     style={'display':'inline-block'})
            ], style={'display':'inline-block',
                      'margin': '20px 30px 30px 30px'}),
#    # map + result
#    html.Div([
#            # left map
#            html.Div([dcc.Graph(id = 'map')], \
#            style = {'margin': '10px 0', \
#                     'height': '80%', \
#                     'width': '100%'})
#            ], style={}),
])

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    return index_page

@app.callback(Output('output-container-button', 'children'),
              [Input("btn_1", 'n_clicks_timestamp'),
               Input('input_1', 'value'),
               Input('input_2', 'value'),
               Input('input_3', 'value'),
               Input('input_4', 'value'),
               Input('input_5', 'value')])
def predict(btn_1, input_1, input_2, input_3, input_4, input_5):
    print(dt.datetime.now().timestamp() - 1)
    btn_1 = float(btn_1)/1000
    print(btn_1)
    if btn_1 == 0:
        return('Enter your information and press predict')
    elif btn_1 < (dt.datetime.now().timestamp() - 1):
        return('Enter your information and press predict')
    elif btn_1 > (dt.datetime.now().timestamp() - 1):
        try:
            pickuptime = dt.datetime.strptime(input_1, "%Y-%m-%d %H:%M")
            pickup_lat = float(input_2)
            pickup_long = float(input_3)
            dropoff_lat = float(input_4)
            dropoff_long = float(input_5)
        except:
            return("Wrong Inputs")
        print("running ... ...")
        test_df = pd.DataFrame({"pickup_longitude": [pickup_long],\
                                "pickup_latitude": [pickup_lat], \
                                "pickup_datetime": [pickuptime],\
                                "dropoff_longitude": [dropoff_long],\
                                "dropoff_latitude": [dropoff_lat]})
        print("feature engineering ... ...")
        # feature engineering
        test_df = allfe(test_df, weather = False, with_fare = False, predict = True)
        print("feature selecting ... ...")
        # feature selection
        featureList = ['distance', 'minute_oftheday', 'degree', 'dropoff_latitude',\
                       'dropoff_longitude', 'pickup_longitude', 'pickup_latitude', 'weekday',\
                       'day', 'hour']
        print("preparing data ... ...")
        test_df = test_df[featureList]
        test_df_xgb = xgb.DMatrix(test_df)
        # stacking features
        stack_X = pd.DataFrame({"xgb": bst.predict(test_df_xgb), \
                                "rf": rf_model.predict(test_df), \
                                "gbdt": gbdt_model.predict(test_df)})
        print("predicting ... ...")
        # ridge
        stack_pred = ridge_stack.predict(stack_X)
        return str(round(stack_pred[0], 3))
    
@app.callback(Output('map', 'figure'),
              [Input("btn_1", 'n_clicks_timestamp'),
               Input('input_2', 'value'),
               Input('input_3', 'value'),
               Input('input_4', 'value'),
               Input('input_5', 'value')])
def map_plot(btn_1, input_2, input_3, input_4, input_5):
    btn_1 = float(btn_1)/1000
    if btn_1 > (dt.datetime.now().timestamp() - 1):
        try:
            pickup_lat = float(input_2)
            pickup_long = float(input_3)
            dropoff_lat = float(input_4)
            dropoff_long = float(input_5)
        except:
            return None
        # prepare for plotly data
        geo_data = [go.Scattergeo(
            lat = [pickup_lat, dropoff_lat],
            lon = [pickup_long, dropoff_long],
            mode = 'lines',
            line = go.scattergeo.Line(
                width = 2,
                color = 'black',
            ),
        )]
        layout = go.Layout(
            showlegend = False,
            geo = go.layout.Geo(
                resolution = 50,
                showland = True,
                showlakes = True,
                landcolor = 'rgb(243, 243, 243)',
                projection = go.layout.geo.Projection(
                    type = 'equirectangular'
                ),
                coastlinewidth = 2,
                lataxis = go.layout.geo.Lataxis(
                    range = [39, 43],
                    showgrid = True,
                    dtick = 10
                ),
                lonaxis = go.layout.geo.Lonaxis(
                    range = [-75, -70],
                    showgrid = True,
                    dtick = 20
                ),
            ),
            margin = go.layout.Margin(pad=0)
        )
        fig = {
            'data': geo_data,
            'layout': layout
        }
        return fig
    else:
        return None


if __name__ == '__main__':
    # load xgboost model
    bst = xgb.Booster()
    bst.load_model("ml_models/model_5.model")
    # load rf model
    rf_model = joblib.load('ml_models/rf_model2.pkl')
    # load gbdt model
    gbdt_model = joblib.load('ml_models/gbdt_model2.pkl')
    # stack model
    ridge_stack = joblib.load('ml_models/ridge_stack2.pkl')
    app.run_server(debug = True)