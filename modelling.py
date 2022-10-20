from classification import tune_classification_model_hyperparameters
from datetime import date, datetime
from models import Models
from regression import tune_regression_model_hyperparameters
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import accuracy_score, recall_score, precision_score, plot_confusion_matrix
from tabular_data import load_airbnb
import json
import joblib
import neural_net as nnw
import numpy as np
import os
import pandas as pd
import plotly.express as px
import torch

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
    if isinstance(model, torch.nn.modules.module.Module):
        pass
    else:
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

def save_model(model, path, y_test = None, y_pred = None, nn_metrics_dict:dict = None, nn_params:dict = None):
    os.makedirs(path, exist_ok = True)
    if isinstance(model, torch.nn.modules.module.Module):
        state = model.state_dict()
        now = datetime.now()
        strf_now = now.strftime("%Y%m%d_%H%M")
        os.mkdir(os.path.join(path, f'{strf_now}'))
        with open(os.path.join(path, f'{strf_now}', 'model.pt'), 'wb') as file:
            torch.save(state, file)
        with open(os.path.join(path, f'{strf_now}', 'model_metrics.json'), 'w') as file:
            json.dump(nn_metrics_dict, file)
        with open(os.path.join(path, f'{strf_now}', 'model_params.json'), 'w') as file:
            params = dict()
            for key, value in nn_params.items():
                if type(value) == int:
                    params.update({key: value})
                elif type(value) == float:
                    params.update({key: float(value)})
                else:
                    params.update({key: str(value)})
            json.dump(params, file)
    else:
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
    dfx, dfy = load_airbnb(df, 'Price_Night')
    X = torch.tensor(dfx.values).float()
    y = torch.tensor(dfy).float()
    os.makedirs(os.path.join('models', 'neural_networks', 'regression', 'optimal'), exist_ok = True)
    configs = nnw.get_nn_config(
        os.path.join('models', 'neural_networks', 'regression', 'optimal', '20221020_1400', 'model_params.json'))
    model = nnw.neural_net(configs = configs)
    state = torch.load(
        os.path.join('models', 'neural_networks', 'regression', 'optimal', '20221020_1400', 'model.pt'))
    model.load_state_dict(state)
    model.eval()
    y_pred = model(X)
    y_pred = y_pred.detach().numpy()
    y = y.detach().numpy()
    fig = px.scatter(y_pred, y)
    fig.write_html('nn.html')
    


