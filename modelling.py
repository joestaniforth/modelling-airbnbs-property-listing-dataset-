from tabular_data import load_airbnb
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
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
        
def evaluate_all_models(model_classes: list, X: pd.DataFrame, y:list, task_folder):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size= 0.75)
    for model_class in model_classes:
        if task_folder == 'regression':
            opt_model = tune_regression_model_hyperparameters(model_class, X_train, y_train)
        elif task_folder == 'classification':
            opt_model = tune_classification_model_hyperparameters(model_class, X_train, y_train)
        save_model(
            opt_model[0], 
            path = os.path.join('model', f'{task_folder}', f'{opt_model[3]}'), 
            y_test = y_test,
            y_pred = opt_model[0].predict(y_test))

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

def save_model(model, path, y_test, y_pred, params):
    os.makedirs(path, exist_ok = True)
    with open(os.path.join(path, 'model.jolib'), 'wb') as model_file:
        joblib.dump(model, model_file)
    with open(os.path.join(path, 'model_params.json'), 'w') as params_file:
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    
    param_dicts = {
        DecisionTreeClassifier:{
            'criterion': ['gini', 'entropy', 'log_loss'],
            'min_samples_leaf': np.arange(1, 7, 2),
            'max_leaf_nodes': [None, 100, 250, 500]
        },
        RandomForestClassifier:{
            'criterion': ['gini', 'entropy', 'log_loss'],
        },
        GradientBoostingClassifier:{

        },
        LogisticRegression:{
            'solver': ['saga', 'lbfgs', 'sag', 'newton-cg'],
            'penalty': ['l2'],
            'C': np.arange(0.2, 1.2, 0.2),
        }
    }
    opt_model = tune_classification_model_hyperparameters(LogisticRegression, X_train, y_train, param_dicts[LogisticRegression])
    y_pred = opt_model[0].predict(X_test)
    save_model(
        opt_model[0],
        os.path.join('models', 'classification', 'LogisticRegression'),
        y_test = y_test, 
        y_pred = y_pred, 
        params= opt_model[1])
