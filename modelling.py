from tabular_data import load_airbnb
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import itertools

hyperparameters = {
    'alpha': np.arange(0.0001, 0.001, 0.00015),
    'epsilon': np.arange(0.1, 0.5, 0.05),
    'eta0': np.arange(0.01, 0.05, 0.005),
    'penalty':['l2', 'l1', 'elasticnet'],
    'power_t': np.arange(0.25, 2, 0.25),
}

def tune_regression_model_hyperparameters():
    pass

def custom_tune_regression_model_hyperparameters(
    model_class, hyperparameter_dict: dict,
    X_train : pd.DataFrame, X_test : pd.DataFrame, X_valid : pd.DataFrame, 
    y_train : list, y_test : list, y_valid : list):
    params, values = zip(*hyperparameter_dict.items())
    all_combinations = [dict(zip(params, vals)) for vals in itertools.product(*values)]
    model_RMSE = float()
    optimal_dict = dict()
    optimal_model = None
    for combination in all_combinations:
        model = model_class(**combination)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_valid)
        if model_RMSE == 0.0:
            model_RMSE = RMSE(y_valid, y_pred)
        if RMSE(y_valid, y_pred) < model_RMSE:
            model_RMSE = RMSE(y_valid, y_pred)
            optimal_dict = combination
            optimal_model = model
    test_y_pred = optimal_model.predict(X_test)
    metrics_dict = {
        'RMSE': RMSE(y_test, test_y_pred),
        'MAE': MAE(y_test, test_y_pred),
        'r2': r2(y_test, test_y_pred)
    }
    return optimal_model, optimal_dict, metrics_dict
        

def RMSE(targets, predicted):
    return np.sqrt(np.mean(np.square(targets - predicted)))

def MAE(targets, predicted):
    return np.mean(np.abs(targets - predicted))

def r2(targets, predicted):
    return 1 - (np.square(RMSE(targets, predicted))/np.var(targets))

if __name__ == '__main__':
    df = pd.read_csv('data\\clean_tabular_data.csv')
    data = load_airbnb(df, 'Price_Night')
    X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], test_size = 0.3)
    X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size = 0.5)
    data = custom_tune_regression_model_hyperparameters(
        model_class = SGDRegressor,
        hyperparameter_dict = hyperparameters,
        X_train = X_train,
        X_test = X_test,
        X_valid = X_valid,
        y_train = y_train,
        y_test = y_test,
        y_valid = y_valid
    )
    print(data[0], data[1], data[2])
