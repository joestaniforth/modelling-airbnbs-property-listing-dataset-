from tabular_data import load_airbnb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import accuracy_score, recall_score, precision_score, plot_confusion_matrix
import pandas as pd
import numpy as np
import json
import os
import joblib
from regression import tune_regression_model_hyperparameters
from classification import tune_classification_model_hyperparameters
from models import Models
        

def evaluate_all_models(model_classes: list, X: pd.DataFrame, y:list, task_folder):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size= 0.8)
    for key in model_classes:
        if task_folder == 'regression':
            opt_model = tune_regression_model_hyperparameters(key, X_train, y_train, param_grid = model_classes[key])
        elif task_folder == 'classification':
            opt_model = tune_classification_model_hyperparameters(key, X_train, y_train,param_grid = model_classes[key])
        save_model(
            opt_model[0], 
            path = os.path.join('models', f'{task_folder}', f'{opt_model[3]}'), 
            y_test = y_test,
            y_pred = opt_model[0].predict(X_test))

def find_best_model(task_directory, evaluate_by):
    model_directories = os.listdir(os.path.join('models', task_directory))
    eval_dict = dict()
    for directory in model_directories:
        with open(os.path.join('models', task_directory, directory, 'model_metrics.json'), 'r') as model_metrics:
            metrics = json.load(model_metrics)
        eval_dict.update({directory: metrics[evaluate_by]})
    low_metrics = ['RMSE', 'MSE', 'MAE'] #Metrics where lower scores are better
    if evaluate_by in low_metrics:
        best_model = min(eval_dict, key = eval_dict.get)
    else:
        best_model = max(eval_dict, key = eval_dict.get)
    with open(os.path.join('models', task_directory, best_model, 'model.joblib'), 'rb') as model_file:
        model = joblib.load(model_file)
    with open(os.path.join('models', task_directory, best_model, 'model_metrics.json'), 'r') as metrics_json:
        metrics_dict = json.load(metrics_json)
    with open(os.path.join('models', task_directory, best_model, 'model_params.json'), 'r') as params_json:
        model_params = json.load(params_json)
    return model, metrics_dict, model_params

def save_model(model, path, y_test, y_pred):
    os.makedirs(path, exist_ok = True)
    with open(os.path.join(path, 'model.joblib'), 'wb') as model_file:
        joblib.dump(model, model_file)
    with open(os.path.join(path, 'model_params.json'), 'w') as params_file:
        params = {k:v if type(v) == str else float(v) for k, v in model.best_params_.items()}
        json.dump(params, params_file)
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

if __name__ == '__main__':
    df = pd.read_csv(os.path.join('data', 'clean_tabular_data.csv'))
    X, y = load_airbnb(df, 'Category')
    param_set = Models()
    evaluate_all_models(model_classes = param_set.classifier_params, X = X, y = y, task_folder = 'classification')
    evaluate_all_models(model_classes = param_set.regressor_params, X = X, y = y, task_folder = 'regression')
