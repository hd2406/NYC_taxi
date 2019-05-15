#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Quan Yuan
"""
# feature engineering module
# the is a module for feature enginnering

import datetime as dt
import pandas as pd
import numpy as np

import module.fare_predict as fp

# transform latitude and longtitude to zipcode
class Geotransform(object):
    def __init__(self, mapdata):
        self.mapdata = mapdata
    # transform one point longtitude, latitude to one zipcode
    def transform(self, longtitude, latitude):
        # object function: Euler distance
        loss = abs(self.mapdata["Latitude"] - latitude)**2 + \
               abs(self.mapdata["Longitude"] - longtitude)**2
        result = self.mapdata.iloc[loss.idxmin()]
        result = result.loc[['Zip', 'City', 'State', 'Timezone']]
        return result.to_dict()
    # transform series to zipcode sereis
    def zipcode_series(self, longtitude, latitude):
        tmp = longtitude.map(lambda x: str(round(x, 4)) + " ") + \
              latitude.map(lambda x: str(round(x, 4)))
        zipcode = tmp.map(lambda x: x.split()).map(lambda x: \
                         self.transform(float(x[0]), float(x[1]))['Zip'])
        return zipcode

# change time frequency to daily
def second_to_day(x):
    year = x.year
    month = x.month
    day = x.day
    return dt.datetime(year, month, day)

# claculate the haversine distance
def ft_haversine_distance(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371 #km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h

# calculate the direction
def ft_degree(lat1, lng1, lat2, lng2):
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))

# data cleaning
def clean_data(taxi_data):
    # we only use trip distance to drop outliers
    # the speed of fastest car in the word and we believe NYC taxi cannot surpass it
    fastest = 268/60/60
    taxi_data = taxi_data[taxi_data['trip_distance']/taxi_data['travel_time'] < fastest]
    # trip distance should > 0
    taxi_data = taxi_data[taxi_data['trip_distance'] > 0]
    
    # drop outliers
    # fix a range for latitude and longtitude
    taxi_data = taxi_data[(taxi_data['dropoff_longitude'] > -75) &\
                          (taxi_data['dropoff_longitude'] < -72)]
    taxi_data = taxi_data[(taxi_data['pickup_longitude'] > -75) &\
                          (taxi_data['pickup_longitude'] < -72)]
    
    # if no valid data after drop na, continue
    taxi_data = taxi_data.dropna()
    taxi_data = taxi_data.sort_values(by = "pickup_datetime")
    # drop latitude = 0 and longitude = 0
    taxi_data = taxi_data[(taxi_data['pickup_longitude'] != 0) &\
                      (taxi_data['pickup_latitude'] != 0) & \
                      (taxi_data['dropoff_longitude'] != 0) & \
                      (taxi_data['dropoff_latitude'] != 0)]
    return taxi_data

# all feature engineering works
def allfe(taxi_data, weather = True, with_fare = False, predict = False):
    # if we include the weather feature (in 2016 we have hourly weather data)
    if weather:
        # join with hourly weather data
        features = ["pickup_datetime", "tempi", "dewpti", "hum", "wspdm", "wdird", \
                "wdire", "vism", "pressurem", "precipm", "conds", "icon", \
                "fog", "rain", "snow", "hail", "thunder", "tornado"]
        weather_data = pd.read_csv("data/hourly_weather.csv", usecols = features)
        weather_data['pickup_datetime'] = weather_data['pickup_datetime'].map(lambda x: \
                                      dt.datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
        weather_data = weather_data.sort_values(by = "pickup_datetime")
        taxi_data = taxi_data.sort_values(by = "pickup_datetime")
        # join them together
        taxi_data = pd.merge_asof(taxi_data, weather_data, on = "pickup_datetime")
        # fillna for weather data
        taxi_data["precipm"] = taxi_data["precipm"].fillna(0)
        taxi_data = taxi_data.fillna(method = "ffill")
        
        # One Hot encoding for categorical weather features
        taxi_data = pd.concat([taxi_data, pd.get_dummies(taxi_data['wdire'])], axis = 1)
        taxi_data = pd.concat([taxi_data, pd.get_dummies(taxi_data['conds'])], axis = 1)
        taxi_data = pd.concat([taxi_data, pd.get_dummies(taxi_data['icon'])], axis = 1)
        taxi_data.drop(['wdire'], axis = 1, inplace = True)
        taxi_data.drop(['conds'], axis = 1, inplace = True)
        taxi_data.drop(['icon'], axis = 1, inplace = True)
    
    # if we want to train the model and we have travel_time, we will do log transformation
    # but if we use this function for predicting, we don't have travel_time
    if not predict:
        # log transformation
        taxi_data["travel_time"] = taxi_data["travel_time"].map(np.log)
    
    # calculate distance
    dist2 = ft_haversine_distance(taxi_data['pickup_latitude'].values,
                              taxi_data['pickup_longitude'].values,
                              taxi_data['dropoff_latitude'].values,
                              taxi_data['dropoff_longitude'].values)
    taxi_data['distance'] = dist2
    
    # calculate degree
    degree = ft_degree(taxi_data['pickup_latitude'].values,
               taxi_data['pickup_longitude'].values,
               taxi_data['dropoff_latitude'].values,
               taxi_data['dropoff_longitude'].values)
    taxi_data['degree'] = degree
    
    # time series information
    taxi_data['month'] = taxi_data['pickup_datetime'].map(lambda x: x.month)
    taxi_data['weekday'] = taxi_data['pickup_datetime'].map(lambda x: x.weekday())
    taxi_data['day'] = taxi_data['pickup_datetime'].map(lambda x: x.day)
    taxi_data['hour'] = taxi_data['pickup_datetime'].map(lambda x: x.hour)
    taxi_data['minute'] = taxi_data['pickup_datetime'].map(lambda x: x.minute)
    taxi_data['minute_oftheday'] = taxi_data['hour'] * 60 + taxi_data['minute']

    # if we want to include fare amount feature (get from LightGBM model we trained before)
    if with_fare:
        taxi_data['fare_predict'] = fp.fare_gbm(taxi_data.loc[:,\
                    ['pickup_longitude','pickup_latitude','pickup_datetime',\
                     'dropoff_longitude','dropoff_latitude',"passenger_count"]])
    
    return taxi_data



