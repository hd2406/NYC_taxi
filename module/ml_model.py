#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Quan Yuan
"""
from sklearn.externals import joblib
from sklearn.model_selection import KFold
from sklearn import metrics

import numpy as np

import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model

# bagging distance model using GBDT
# we are not going to use it since math-formula-based distance is better
# after evaluating the result
def gbdt_dist(X):
    '''
    X:
    pickup_longitude
    pickup_latitude
    dropoff_longitude
    dropoff_latitude
    '''
    result = 0
    for i in range(1, 6):
        loaded_model = joblib.load('models/dist/GBDT_fold{0}.sav'.format(i))
        result += loaded_model.predict(X)
    result = result/5
    return result

# XGBoost model
def xgb_model(X, y, fold = 5, first = True, model_path = None):
    skf = KFold(n_splits = fold, shuffle = True)
    fold_num = 0
    xgbm = []
    r2valid = []
    for train_index, valid_index in skf.split(X, y):
        # dataset
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        # xgb matrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_valid, label=y_valid)
        # parameters
        param = {'max_depth': 3, 
                 'eta': 0.08, 
                 'booster': 'gbtree',
                 'subsample': 1,
                 'lambda': 0.3,
                 'alpha': 0.2,
                 'verbosity': 0,
                 'min_child_weight': 0.9,
                 'objective': 'reg:linear'}
        param['nthread'] = 4
        param['eval_metric'] = 'rmse'
        evallist = [(dvalid, 'eval'), (dtrain, 'train')]
        # model
        num_round = 300
        # if this is the first time to train this model
        if first:
            bst = xgb.train(param, dtrain, num_round, evallist)
        else:
            bst = xgb.train(param, dtrain, num_round, xgb_model = model_path)
        # save model
        xgbm.append(bst)
        # predict
        y_train_pred = bst.predict(xgb.DMatrix(X_train))
        y_valid_pred = bst.predict(xgb.DMatrix(X_valid))
        # evaluation
        # r2
        r2_train = metrics.r2_score(y_train, y_train_pred)
        r2_valid = metrics.r2_score(y_valid, y_valid_pred)
        # append score
        r2valid.append(r2_valid)
        # mse
        mse_train = metrics.mean_squared_error(y_train, y_train_pred)
        mse_valid = metrics.mean_squared_error(y_valid, y_valid_pred)
        # rmsle
        rmsle_train = np.sqrt(metrics.mean_squared_log_error(y_train, y_train_pred))
        rmsle_valid = np.sqrt(metrics.mean_squared_log_error(y_valid, y_valid_pred))
        # print
        print("Fold {0} Train R2 {1}".format(fold_num, r2_train))
        print("Fold {0} Test R2 {1}".format(fold_num, r2_valid))
        print("Fold {0} Train mse {1}".format(fold_num, mse_train))
        print("Fold {0} Test mse {1}".format(fold_num, mse_valid))
        print("Fold {0} Train rmsle {1}".format(fold_num, rmsle_train))
        print("Fold {0} Test rmsle {1}".format(fold_num, rmsle_valid))
        fold_num += 1
    # get best model
    mod = xgbm[r2valid.index(max(r2valid))]
    return mod, r2valid

# Deep Neural Network Model
def dnn_model(X, y, fold = 5, first = True, model_path = None):
    skf = KFold(n_splits = fold, shuffle = True)
    fold_num = 1
    nn = []
    r2valid = []
    #train using best alpha
    for train_index, valid_index in skf.split(X, y):
        # dataset
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        
        # initialize parameter
        n = X_train.values.shape[1]
        
        # if this is the first time to train this model
        if first:
            # deep learning model
            taxi_dnn = Sequential()
            taxi_dnn.add(Dense(n, activation = 'relu', input_shape = (n, )))
            taxi_dnn.add(Dense(int(0.8*n), activation = 'relu'))
            taxi_dnn.add(Dense(int(0.4*n), activation = 'relu'))
            taxi_dnn.add(Dense(int(0.3*n), activation = 'relu'))
            taxi_dnn.add(Dense(1, activation = 'linear'))
            taxi_dnn.compile(loss = 'mse', optimizer = 'rmsprop', metrics = ['mse'])
            taxi_dnn.fit(X_train, y_train.values, epochs = 4)
        else:
            taxi_dnn = load_model(model_path)
            taxi_dnn.fit(X_train, y_train.values, epochs = 4)
            
        # save model
        nn.append(taxi_dnn)
        
        # predict
        y_train_pred = taxi_dnn.predict(X_train)
        y_valid_pred = taxi_dnn.predict(X_valid)
        
        # result
        # r2
        r2_train = metrics.r2_score(y_train, y_train_pred)
        r2_valid = metrics.r2_score(y_valid, y_valid_pred)
        r2valid.append(r2_valid)
        # mse
        mse_train = metrics.mean_squared_error(y_train, y_train_pred)
        mse_valid = metrics.mean_squared_error(y_valid, y_valid_pred)
        # rmsle
        rmsle_train = np.sqrt(metrics.mean_squared_log_error(y_train, y_train_pred))
        rmsle_valid = np.sqrt(metrics.mean_squared_log_error(y_valid, y_valid_pred))
        # print
        print("Fold {0} Train R2 {1}".format(fold_num, r2_train))
        print("Fold {0} Test R2 {1}".format(fold_num, r2_valid))
        print("Fold {0} Train mse {1}".format(fold_num, mse_train))
        print("Fold {0} Test mse {1}".format(fold_num, mse_valid))
        print("Fold {0} Train rmsle {1}".format(fold_num, rmsle_train))
        print("Fold {0} Test rmsle {1}".format(fold_num, rmsle_valid))
        fold_num += 1
    # get best model
    bst_dnn = nn[r2valid.index(max(r2valid))]
    return bst_dnn, r2valid

# Random Forest Model
def rf_model(X, y, fold = 5):
    skf = KFold(n_splits = fold, shuffle = True)
    fold_num = 1
    rf = []
    r2valid = []
    for train_index, valid_index in skf.split(X, y):
        # dataset
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        # model
        reg = RandomForestRegressor(n_estimators=100, random_state=0, max_depth=4, \
                                    criterion = "mse")
        reg.fit(X_train, y_train)
        # save model
        rf.append(reg)
        y_train_pred = reg.predict(X_train)
        y_valid_pred = reg.predict(X_valid)
        # result
        # r2
        r2_train = metrics.r2_score(y_train, y_train_pred)
        r2_valid = metrics.r2_score(y_valid, y_valid_pred)
        r2valid.append(r2_valid)
        # mse
        mse_train = metrics.mean_squared_error(y_train, y_train_pred)
        mse_valid = metrics.mean_squared_error(y_valid, y_valid_pred)
        # rmsle
        rmsle_train = np.sqrt(metrics.mean_squared_log_error(y_train, y_train_pred))
        rmsle_valid = np.sqrt(metrics.mean_squared_log_error(y_valid, y_valid_pred))
        # print
        print("Fold {0} Train R2 {1}".format(fold_num, r2_train))
        print("Fold {0} Test R2 {1}".format(fold_num, r2_valid))
        print("Fold {0} Train mse {1}".format(fold_num, mse_train))
        print("Fold {0} Test mse {1}".format(fold_num, mse_valid))
        print("Fold {0} Train rmsle {1}".format(fold_num, rmsle_train))
        print("Fold {0} Test rmsle {1}".format(fold_num, rmsle_valid))
        fold_num += 1
    # get best model
    bst_rf = rf[r2valid.index(max(r2valid))]
    return bst_rf, r2valid

# GBDT model
def gbdt_model(X, y, fold = 5):
    skf = KFold(n_splits = fold, shuffle = True)
    fold_num = 1
    gbdt = []
    r2valid = []
    for train_index, valid_index in skf.split(X, y):
        # dataset
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        # model
        reg = GradientBoostingRegressor(n_estimators=50, random_state=0, max_depth=4, \
                                        learning_rate = 0.15, subsample = 0.8)
        reg.fit(X_train, y_train)
        # save model
        gbdt.append(reg)
        y_train_pred = reg.predict(X_train)
        y_valid_pred = reg.predict(X_valid)
        # result
        # r2
        r2_train = metrics.r2_score(y_train, y_train_pred)
        r2_valid = metrics.r2_score(y_valid, y_valid_pred)
        r2valid.append(r2_valid)
        # mse
        mse_train = metrics.mean_squared_error(y_train, y_train_pred)
        mse_valid = metrics.mean_squared_error(y_valid, y_valid_pred)
        # rmsle
        rmsle_train = np.sqrt(metrics.mean_squared_log_error(y_train, y_train_pred))
        rmsle_valid = np.sqrt(metrics.mean_squared_log_error(y_valid, y_valid_pred))
        # print
        print("Fold {0} Train R2 {1}".format(fold_num, r2_train))
        print("Fold {0} Test R2 {1}".format(fold_num, r2_valid))
        print("Fold {0} Train mse {1}".format(fold_num, mse_train))
        print("Fold {0} Test mse {1}".format(fold_num, mse_valid))
        print("Fold {0} Train rmsle {1}".format(fold_num, rmsle_train))
        print("Fold {0} Test rmsle {1}".format(fold_num, rmsle_valid))
        fold_num += 1
    # get best model
    bst_rf = gbdt[r2valid.index(max(r2valid))]
    return bst_rf, r2valid