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
from regression import tune_regression_model_hyperparameters
        
def evaluate_all_models(model_classes: list, X: pd.DataFrame, y:list, task_folder):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size= 0.75)
    for model_class in model_classes:
        if task_folder == 'regression':
            opt_model = tune_regression_model_hyperparameters(model_class, X_train, y_train)
        elif task_folder == 'classification':
            opt_model = tune_classification_model_hyperparameters(model_class, X_train, y_train)
        os.makedirs(os.path.join('models', f'{task_folder}', f'{opt_model[3]}'), exist_ok = True)
        with open(os.path.join('models', f'{task_folder}', f'{opt_model[3]}', 'model.joblib'), 'wb') as file:
            joblib.dump(opt_model[0], file)
        y_pred = opt_model[0].predict(X_test)
        with open(os.path.join('models', f'{task_folder}', f'{opt_model[3]}', 'model_params.json'), 'w') as file:
            json.dump(opt_model[1], file)
        metrics_dict = {
            'RMSE': mean_squared_error(y_test, y_pred, squared = False),
            'MSE' : mean_squared_error(y_test, y_pred),
            'R Squared': r2_score(y_test, y_pred)
        }
        with open(os.path.join('models', f'{task_folder}', f'{opt_model[3]}', 'model_metrics.json'), 'w') as file:
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

def save_model(model, path, y_test, y_pred):
    os.makedirs(path, exist_ok = True)
    with open(os.path.join(path, 'model.jolib'), 'wb') as model_file:
        joblib.dump(model, model_file)
    with open(os.path.join(path, 'model_params.json'), 'wb') as params_file:
        json.dump(model.get_params(), params_file)
    if 'regression' in path:
        metrics_dict = {
            'RMSE': mean_squared_error(y_test, y_pred, squared = False),
            'MSE' : mean_squared_error(y_test, y_pred),
            'R Squared': r2_score(y_test, y_pred)
        }
        with open(os.path.join(path, 'model_metrics.json'), 'w') as metrics_file:
            json.dump(metrics_dict, metrics_file)
    elif 'classification' in path:
        metrics_dict = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precison': precision_score(y_test, y_pred, average = 'weighted'),
        'Recall': recall_score(y_test, y_pred, average = 'weighted'),
        'f1': (2*(precision_score(y_test, y_pred, average = 'weighted') * recall_score(y_test, y_pred, average = 'weighted'))
                /(precision_score(y_test, y_pred, average = 'weighted') + recall_score(y_test, y_pred, average = 'weighted')))
        }
        with open(os.path.join(path, 'model_metrics.json'), 'w') as metrics_file:
            json.dump(metrics_dict, metrics_file)
    
def tune_classification_model_hyperparameters(model_class, X_train, y_train):
    model = model_class()
    param_grid = {    
        'solver': ['saga', 'lbfgs', 'sag', 'newton-cg'],
        'penalty': ['l2'],
        'C': np.arange(0.2, 1.2, 0.2),
        'max_iter': [10000]
        }
    param_grid_2 = {    
        'solver': ['saga', 'lbfgs', 'sag', 'newton-cg'],
        'penalty': ['l2'],
        'C': np.arange(0.2, 1.2, 0.2),
        'max_iter': [10000]
        }
    param_grid_3 = {    
        'solver': ['saga', 'lbfgs', 'sag', 'newton-cg'],
        'penalty': ['l2'],
        'C': np.arange(0.2, 1.2, 0.2),
        'max_iter': [10000]
        }
    try:
        clf = GridSearchCV(
        model,
        cv = 10,
        scoring = 'accuracy',
        param_grid = param_grid,
        refit = 'accuracy'
        )
        clf.fit(X_train, y_train)
    except ValueError:
        clf = GridSearchCV(
        model,
        cv = 10,
        scoring = 'accuracy',
        param_grid = param_grid_2,
        refit = 'accuracy'
        )
        clf.fit(X_train, y_train)
    except ValueError:
        clf = GridSearchCV(
        model,
        cv = 10,
        scoring = 'accuracy',
        param_grid = param_grid_3,
        refit = 'accuracy'
        )
        clf.fit(X_train, y_train)
    return clf

if __name__ == '__main__':
    df = pd.read_csv(os.path.join('data', 'clean_tabular_data.csv'))
    X, y = load_airbnb(df, 'Category')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    model = tune_classification_model_hyperparameters(LogisticRegression, X_train, y_train)
    y_pred = model.predict(X_test)
    save_model(
        model = model,
        path = os.path.join('models', 'classification', 'LogisticRegression'), 
        y_test= y_test, 
        y_pred = y_pred)
