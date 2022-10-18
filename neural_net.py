from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import torch
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tabular_data import load_airbnb
from os.path import join
from os import mkdir
import matplotlib.pyplot as plt
import itertools



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
    def __init__(self, hidden_neurons, input_neurons = 3, ouput_neurons = 1) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
           torch.nn.Linear(input_neurons, hidden_neurons),
           torch.nn.ReLU(),
           torch.nn.Linear(hidden_neurons, ouput_neurons),
        )

    def forward(self, features):
        return self.layers(features)



def train(learning_rate: float, hidden_neurons: int, epochs = 100):
    model = neural_net(hidden_neurons = hidden_neurons)
    optimiser = torch.optim.SGD(params = model.parameters(), lr = learning_rate)
    for epoch in range(epochs):
        for X_train, y_train in training_loader:
            optimiser.zero_grad()
            y_pred= model(X_train)
            loss = F.mse_loss(y_pred, y_train)
            loss.backward()
            optimiser.step()
    return model



if __name__ == '__main__':

    df = pd.read_csv('feature_engineering\\engineered_data.csv')
    dfx, dfy = load_airbnb(df, 'Price_Night')
    X = torch.tensor(dfx.values).float()
    y = torch.tensor(dfy).float()
    y.view(y.shape[0], 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size = 0.2)

    train_dataset = AirbnbNightlyPriceImageDataset(X_train, y_train)
    test_dataset = AirbnbNightlyPriceImageDataset(X_test, y_test)

    training_loader = DataLoader(batch_size = 16, shuffle = True, dataset = train_dataset)
    testing_loader = DataLoader(batch_size = 16, shuffle = True, dataset = test_dataset)

    lr_list = [0.1, 0.01, 0.001, 0.0001]
    h_list = [22, 16, 11, 8, 2]
    comb_list = itertools.product(lr_list, h_list)
    #for i in comb_list:
    opt_model = train(1e-4, 8)
    pred = opt_model(X).detach().numpy()
    

    print(r2_score(y.detach().numpy(), pred), mean_squared_error(y.detach().numpy(), pred, squared = False))
    plt.plot(X, y, 'ro')
    plt.plot(X, pred, 'b')
    plt.show()

    
