# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 00:43:36 2020

@author: alex
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import statsmodels.formula.api as smf

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, precision_recall_curve, auc, roc_curve, recall_score
from sklearn.metrics import mean_squared_log_error

from xgboost import XGBRegressor
import xgboost as xgb

import catboost
print(catboost.__version__)
from catboost import *
from catboost import datasets
from catboost import CatBoostRegressor, Pool

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

from sklearn.ensemble import RandomForestRegressor


SEED = 1970

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
submission = pd.read_csv('sample_submission.csv')
train_data.info()
train_data.rename(columns = {'TARGET(PRICE_IN_LACS)' : 'price'}, inplace = True)
train_data.columns.tolist()
categorical_features = ['POSTED_BY', 'UNDER_CONSTRUCTION', 'RERA', 'BHK_OR_RK', 'READY_TO_MOVE', 'RESALE', 'nbrhd', 'city']

le = LabelEncoder()

train_data.isna().sum()
test_data.isna().sum()


train_data.drop(train_data.loc[train_data.price > 10000].index, inplace =True)
train_data.drop(train_data.loc[train_data.SQUARE_FT > 300000].index, inplace =True)


df_combined = pd.concat([train_data, test_data])
df_combined['SQUARE_FT'] = df_combined['SQUARE_FT'].astype('int64')

city = df_combined['ADDRESS'].str.split(',', n = 1, expand = True)
df_combined['nbrhd'] = city[0] 
df_combined['city'] = city[1]
df_combined.drop(columns = ['ADDRESS'], inplace = True)

df_combined1 = pd.get_dummies(df_combined, columns = categorical_features, dtype=int, drop_first=None)
df_combined1.fillna(0, inplace=True)

df_train_mod = df_combined1.iloc[:train_data.shape[0]].copy()
df_test_mod = df_combined1.iloc[train_data.shape[0]:].copy()
price = df_train_mod.price
df_train_mod.drop(columns = 'price', inplace = True)
df_test_mod.drop(columns = 'price', inplace = True)


X_train, X_valid, y_train, y_valid = train_test_split(df_train_mod, price, train_size=0.8, random_state = SEED)

XGB_model = XGBRegressor(n_estimators = 10000,
                           depth = 10, 
                           learning_rate = 0.005,
                           objective = 'reg:squarederror', 
                           verbosity = 1,
                           random_state = SEED, n_jobs=-1) 

print(XGB_model)
XGB_model.fit(X_train, y_train,
                eval_set = [(X_valid, y_valid)],
                eval_metric=['rmse'],
                # eval_metric=['rmsle','rmse'],
                early_stopping_rounds=50, verbose = 500)

XGB_preds = XGB_model.predict(X_valid)
XGB_test_preds = XGB_model.predict(df_test_mod)

np.sum((XGB_preds<0))
np.sum((XGB_test_preds<0))
XGB_preds[XGB_preds<0] = 0.001
XGB_test_preds[XGB_test_preds<0] = 0.001

XGB_score = np.sqrt(mean_squared_log_error(y_valid, XGB_preds))


Cat_model = CatBoostRegressor(
                                learning_rate = 0.05, 
                                iterations= 20000, 
                                depth= 12, 
                                random_seed = SEED,
                                loss_function='RMSE',
                                eval_metric = 'RMSE',
                                # task_type = 'GPU',
                                 )

Cat_model.fit(X_train, y_train,
                eval_set = [(X_valid, y_valid)],
                early_stopping_rounds = 100, verbose = 500)

Cat_preds = Cat_model.predict(X_valid)
Cat_preds_test = Cat_model.predict(df_test_mod)
np.sum((Cat_preds<0))

np.sum((Cat_preds_test<0))
Cat_preds[Cat_preds<0] = 0.001
Cat_preds_test[Cat_preds_test<0] = 0.001

Cat_score = np.sqrt(mean_squared_log_error(y_valid, Cat_preds))
    

model = DecisionTreeRegressor()
model.fit(X_train, y_train)
DTR_pred = model.predict(X_valid)
np.sum((DTR_pred <0))
DTR_score = np.sqrt(mean_squared_log_error(y_valid, DTR_pred))
print('RMSLE: ', DTR_score )
DecisionTree_preds = model.predict(df_test_mod)

model = RandomForestRegressor(n_jobs = -1,
                              random_state = SEED,
                              verbose = 0,
                               n_estimators = 500    ) #0.31137
model.fit(X_train, y_train)
RF_pred = model.predict(X_valid)
np.sum((RF_pred <0))
RF_score = np.sqrt(mean_squared_log_error(y_valid, RF_pred))
print('RMSLE: ', RF_score)
RF_preds = model.predict(df_test_mod)

# mean_preds = (DTR_pred * 0.4  + XGB_preds * 0.1 + Cat_preds * 0.1 + RF_pred *0.4 ) 
mean_preds = (DTR_pred * 0.1  + XGB_preds * 0.1 + Cat_preds * 0.1 + RF_pred * 0.7) 
# mean_preds = (y_pred*0.3  + XGB_preds *0.3 + Cat_preds *0.4) 
# mean_preds = (y_pred*0.7  + XGB_preds *0.3) 
print('RMSLE: ',np.sqrt(mean_squared_log_error(y_valid, mean_preds)))

mean_test_preds = (DecisionTree_preds * 0.1  + XGB_test_preds *0.1 + Cat_preds_test * 0.1 + RF_preds * 0.7) 
# mean_test_preds = (DecisionTree_preds * 0.3  + XGB_test_preds *0.3 + Cat_preds_test * 0.4) 
# mean_test_preds = (DecisionTree_preds * 0.7  + XGB_test_preds *0.3) 
submission['TARGET(PRICE_IN_LACS)'] = mean_test_preds 
submission.to_csv('DT01_XGB01_Cat01_RF07_03091.csv', index = False)
