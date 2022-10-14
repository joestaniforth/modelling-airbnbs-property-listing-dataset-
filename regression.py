from tabular_data import load_airbnb
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, plot_confusion_matrix
import pandas as pd
import numpy as np
import itertools
import json
import os
import joblib
import plotly.express as px
from progress.bar import ChargingBar
import xgboost as xgb

def tune_regression_model_hyperparameters(model_class, X_train : pd.DataFrame, y_train : list):
    model = model_class()
    class_name = model.__class__.__name__
    param_grid_1 = {
        'learning_rate': [0.05, 0.1, 0.15],
        'max_depth': [3, 5, 7, 9],
        'gamma': [0, 0.1, 0.2],
        'min_child_weight': [3, 5, 7],
        'n_estimators': [100, 250, 500]
        }
    param_grid_2 = {
        'max_depth': [5, 7, 9],
        'min_samples_split': [2, 5, 7],
        'min_weight_fraction_leaf': [0, 0.25, 0.5]
        }
    try:
        rgr = GridSearchCV(
            model, 
            cv = 10, 
            scoring = ['r2', 'neg_root_mean_squared_error'], 
            param_grid = param_grid_1, 
            refit = 'r2')
        rgr.fit(X_train, y_train)
    except ValueError as e:
        rgr = GridSearchCV(
            model, 
            cv = 10, 
            scoring = ['r2', 'neg_root_mean_squared_error'], 
            param_grid = param_grid_2, 
            refit = 'r2')
        rgr.fit(X_train, y_train)
    return rgr, rgr.best_params_, rgr.best_score_, class_name

def custom_tune_regression_model_hyperparameters(
    model_class, parameters: dict,
    X_train : pd.DataFrame, X_test : pd.DataFrame, X_valid : pd.DataFrame, 
    y_train : list, y_test : list, y_valid : list):
    if 'class' in parameters:
        parameters.pop('class')
    params, values = zip(*parameters.items())
    all_combinations = [dict(zip(params, vals)) for vals in itertools.product(*values)]
    model_RMSE = float()
    optimal_dict = dict()
    optimal_model = None
    model_name = model_class()
    with ChargingBar(f'{model_name.__class__.__name__}, {len(all_combinations)}', max = len(all_combinations)) as bar:
        for combination in all_combinations:
            model = model_class(**combination)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_valid)
            if model_RMSE == 0.0:
                model_RMSE = mean_squared_error(y_valid, y_pred, squared = False)
            if mean_squared_error(y_valid, y_pred, squared = False) < model_RMSE:
                model_RMSE = mean_squared_error(y_valid, y_pred, squared = False)
                optimal_dict = combination
                optimal_model = model
            bar.next()
    test_y_pred = optimal_model.predict(X_test)
    metrics_dict = {
        'Validation RMSE': mean_squared_error(y_test, test_y_pred, squared = False),
        'r2': r2_score(y_test, test_y_pred)
    }
    figure = px.scatter(x = y_test, y = test_y_pred)
    return optimal_model, optimal_dict, metrics_dict, figure