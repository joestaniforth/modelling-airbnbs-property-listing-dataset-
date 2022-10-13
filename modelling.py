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
        
def evaluate_all_models(model_classes: list, X: pd.DataFrame, y:list):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size= 0.75)
    for model_class in model_classes:
        opt_model = tune_regression_model_hyperparameters(model_class, X_train, y_train)
        os.makedirs(os.path.join('models', 'regression', f'{opt_model[3]}'), exist_ok = True)
        with open(os.path.join('models', 'regression', f'{opt_model[3]}', 'model.joblib'), 'wb') as file:
            joblib.dump(opt_model[0], file)
        y_pred = opt_model[0].predict(X_test)
        figure = px.scatter(y = y_test, x = y_pred)
        figure.write_html(os.path.join('models', 'regression', f'{opt_model[3]}', 'plot.html'))
        with open(os.path.join('models', 'regression', f'{opt_model[3]}', 'model_params.json'), 'w') as file:
            json.dump(opt_model[1], file)
        metrics_dict = {
            'RMSE': mean_squared_error(y_test, y_pred, squared = False),
            'MSE' : mean_squared_error(y_test, y_pred),
            'R Squared': r2_score(y_test, y_pred)
        }
        with open(os.path.join('models', 'regression', f'{opt_model[3]}', 'model_metrics.json'), 'w') as file:
            json.dump(metrics_dict, file)

def find_best_model(model_type, evaluate_by):
    model_directories = os.listdir(os.path.join('models', model_type))
    eval_metric = float()
    eval_dict = dict()
    for directory in model_directories:
        with open(os.path.join('models', model_type, directory, 'model_metrics.json'), 'r') as model_metrics:
            metrics = json.load(model_metrics)
        if evaluate_by == 'R Squared' and metrics['R Squared'] > eval_metric:
            eval_metric = metrics['R Squared']
        elif eval_metric == 0:
            eval_metric = metrics[evaluate_by]
        elif metrics[evaluate_by] < eval_metric:
            eval_metric = metrics[evaluate_by]
        eval_dict.update({directory: eval_metric})
    if evaluate_by == 'R Squared':
        best_model = max(eval_dict, key = eval_dict.get)
    else:
        best_model = min(eval_dict, key = eval_dict.get)
    with open(os.path.join('models', model_type, best_model, 'model.joblib'), 'rb') as model_file:
        model = joblib.load(model_file)
    with open(os.path.join('models', model_type, best_model, 'model_metrics.json'), 'r') as metrics_json:
        metrics_dict = json.load(metrics_json)
    with open(os.path.join('models', model_type, best_model, 'model_params.json'), 'r') as params_json:
        model_params = json.load(params_json)
    return model, metrics_dict, model_params


if __name__ == '__main__':
    df = pd.read_csv(os.path.join('data', 'clean_tabular_data.csv'))
    X, y = load_airbnb(df, 'Category')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics_dict = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precison': precision_score(y_test, y_pred, average = 'weighted'),
        'Recall': recall_score(y_test, y_pred, average = 'weighted'),
        'f1': (2*(precision_score(y_test, y_pred, average = 'weighted') * recall_score(y_test, y_pred, average = 'weighted'))
                /(precision_score(y_test, y_pred, average = 'weighted') + recall_score(y_test, y_pred, average = 'weighted')))
    }
    print(metrics_dict)
    for p, t in zip(y_pred, y_test):
        print(p, t)
    #model_list = [DecisionTreeRegressor, RandomForestRegressor, GradientBoostingRegressor, xgb.XGBRegressor]
    #evaluate_all_models(model_list, X, y)
    #model_bundle = find_best_model('regression', 'RMSE')
    #print(model_bundle[1], model_bundle[2])