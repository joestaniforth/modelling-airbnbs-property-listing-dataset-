from os.path import join
from os import mkdir, makedirs
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tabular_data import load_airbnb
from time import time
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import torch
import torch.nn.functional as F
import yaml

class AirbnbNightlyPriceImageDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        assert len(X) == len(y)
        self.X = X
        self.y = y

    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
    def __len__(self):
        return len(self.y)

class neural_net(torch.nn.Module):
    def __init__(self, configs, input_neurons = 11, ouput_neurons = 1) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
           torch.nn.Linear(input_neurons, configs['hidden_layer_width']),
           torch.nn.Dropout(p = configs['dropout']),
           torch.nn.ReLU(),
           torch.nn.Linear(configs['hidden_layer_width'], ouput_neurons),
        )

    def forward(self, features):
        return self.layers(features)

def train(model, config: dict, training_loader):
    writer = SummaryWriter()
    optimiser = config['optimiser'](params = model.parameters(), lr = config['learning_rate'])
    start_time = time()
    for epoch in range(config['epochs']):
        for X_train, y_train in training_loader:
            optimiser.zero_grad()
            y_pred = model(X_train)
            unsq_y_train = y_train.unsqueeze(1)
            loss = F.mse_loss(y_pred, unsq_y_train)
            writer.add_scalar(tag = 'MSE_loss', scalar_value = loss, global_step = epoch)
            loss.backward()
            optimiser.step()
    train_time = time() - start_time
    return train_time

def grid_train(model, config: dict, training_loader, optimiser_algo):
    writer = SummaryWriter()
    batch_idx = 0
    optimiser = optimiser_algo(params = model.parameters(), lr = config['learning_rate'])
    start_time = time()
    for epoch in range(config['epochs']):
        for X_train, y_train in training_loader:
            optimiser.zero_grad()
            y_pred = model(X_train)
            unsq_y_train = y_train.unsqueeze(1)
            loss = F.mse_loss(y_pred, unsq_y_train)
            writer.add_scalar(tag = 'MSE_loss', scalar_value = loss.item(), global_step = batch_idx)
            loss.backward()
            optimiser.step()
            batch_idx += 1
    train_time = time() - start_time
    return train_time

def prep_data_for_nn(path: str, test_size = 0.2, valid_size = 0.2, random_state = 13):
    df = pd.read_csv(path)
    dfx, dfy = load_airbnb(df, 'Price_Night')
    X = torch.tensor(dfx.values).float()
    y = torch.tensor(dfy).float()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = 13)
    X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size = valid_size, random_state = 13)
    return X_train, X_test, X_valid, y_train, y_test, y_valid

def get_nn_config(path:str):
    with open(path, 'r') as stream:
        config = yaml.unsafe_load(stream) #unsafe load as constructor in yaml
    return config

def evaluate_model(model, X_train, X_test, X_valid, y_train, y_test, y_valid, train_time):
    start_time = time()
    train_pred = model(X_train)
    test_pred = model(X_test)
    valid_pred = model(X_valid)
    end_time = time() - start_time
    avg_latency = end_time/(len(X_train) + len(X_valid) + len(X_test)) 
    metrics_dict = {
        'RMSE_loss': {
            'train': float(mean_squared_error(y_train.detach().numpy(), train_pred.detach().numpy(), squared = False)), 
            'test': float(mean_squared_error(y_test.detach().numpy(), test_pred.detach().numpy(), squared = False)), 
            'valid': float(mean_squared_error(y_valid.detach().numpy(), valid_pred.detach().numpy(), squared = False))
            },
        'R_Squared': {
            'train': float(r2_score(y_train.detach().numpy(), train_pred.detach().numpy())), 
            'test': float(r2_score(y_test.detach().numpy(), test_pred.detach().numpy())), 
            'valid': float(r2_score(y_valid.detach().numpy(), valid_pred.detach().numpy()))
            },
        'training_time': float(train_time),
        'inference_latency': float(avg_latency)
    }
    return metrics_dict

def generate_nn_configs(param_grid_path):
    grid = get_nn_config(param_grid_path)
    params, values = zip(*grid.items())
    all_combinations = [dict(zip(params, vals)) for vals in itertools.product(*values)]
    return all_combinations

def optimise_nn_grid(X_train, X_test, X_valid, y_train, y_test, y_valid, param_grid):
    train_dataset = AirbnbNightlyPriceImageDataset(X_train, y_train)
    test_dataset = AirbnbNightlyPriceImageDataset(X_test, y_test)
    valid_dataset = AirbnbNightlyPriceImageDataset(X_valid, y_valid)
    params, values = zip(*param_grid.items())
    all_combinations = [dict(zip(params, vals)) for vals in itertools.product(*values)]
    optimal_dict = dict()
    optimal_metric = float()
    opt_metrics_dict = dict()
    for combination in all_combinations:
        training_loader = DataLoader(batch_size = combination['training_batch_size'], shuffle = True, dataset = train_dataset)
        model = neural_net(combination)
        train_time = grid_train(model, combination, training_loader, torch.optim.SGD)
        eval_dict = evaluate_model(model, X_train, X_test, X_valid, y_train, y_test, y_valid, train_time)
        print(eval_dict['RMSE_loss']['test'])
        if optimal_metric == 0:
            optimal_metric = eval_dict['RMSE_loss']['test']
            optimal_dict = combination
            state = model.state_dict()
            opt_metrics_dict = eval_dict
            print('init metrics set')
        if eval_dict['RMSE_loss']['test'] < optimal_metric:
            optimal_metric = eval_dict['RMSE_loss']['test']
            optimal_dict = combination
            state = model.state_dict()
            opt_metrics_dict = eval_dict
            print('metrics updated')
    return optimal_dict, opt_metrics_dict, state


if __name__ == '__main__':
    X_train, X_test, X_valid, y_train, y_test, y_valid = prep_data_for_nn('data\\clean_tabular_data.csv')
    param_grid = get_nn_config('nn_config_grid.yml')
    print(param_grid)
    results = optimise_nn_grid(X_train, X_test, X_valid, y_train, y_test, y_valid, param_grid)
    makedirs(join('models', 'neural_networks', 'regression', 'optimal'), exist_ok = True)
    model = neural_net(configs = results[0])
    model.load_state_dict(results[2])
    model.eval()

    
