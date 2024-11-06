import torch
import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from metrics.metrics import Metrics


cols = [7, 0, 1, 2, 3, 4, 5, 6, 10, 11, 12] # features to use

lookback = 30
horizon = 90
max_epochs = 300
batch_size = 1024
n_features =  len(cols)

split = [1,0.00001, 0.00001]
n_neurons=[25,25]
n_layers=2
dropout = 0.0


lr = 0.0002


trainval_df_raw = pd.read_pickle('IFE_dataset_model/trainval_df.pkl')
trainval_df = trainval_df_raw.iloc[:,cols]
# X_trainval_sc = np.load('trainval_C_sc01.npy')
index_trainval = pd.read_pickle('IFE_dataset_model/trainval_C_index.pkl')
#!TODO Dataset class

# Idea: split the data set into train and validation sets
# Then, normalize the data set with min-max scaling for every column and only according min max of train set
split = 1/2
# Split the data set into train and validation sets
train_data = trainval_df.iloc[:int(len(index_trainval)*split)]
val_data = trainval_df.iloc[int(len(index_trainval)*(1-split)):]
# Normalize the data set with min-max scaling for every column and only according min max of train set

class Scaler():
    def __init__(self, tain_data):
        self.min_vals = train_data.min()
        self.max_vals = train_data.max()

        self.target_min = self.min_vals['GHI']
        self.target_max = self.max_vals['GHI']
    def transform(self, data):
        return (data - self.min_vals) / (self.max_vals - self.min_vals)
    def inverse_transform(self, data):
        return data * (self.max_vals - self.min_vals) + self.min_vals
    def inverse_results_transform(self, X, y):
        X = X * (self.max_vals["GHI"] - self.min_vals["GHI"]) + self.min_vals["GHI"]
        y = y * (self.max_vals["GHI"] - self.min_vals["GHI"]) + self.min_vals["GHI"]
        return X, y

scaler = Scaler(train_data)
train_data = scaler.transform(train_data)
val_data = scaler.transform(val_data)



class IFE_Dataset(torch.utils.data.Dataset):
    def __init__(self, X, y, lookback, horizon):
        self.data = X
        self.X = X.values
        self.y = y
        self.lookback = lookback
        self.horizon = horizon
    
    def __len__(self):
        return len(self.X) - (self.lookback + self.horizon) + 1
    
    def __getitem__(self, idx):
        x = self.X[idx:idx+self.lookback]
        y = self.y[idx+self.lookback:idx+self.lookback+self.horizon]
        return x,y

train_dataset = IFE_Dataset(train_data, train_data['GHI'].values, lookback, horizon)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = IFE_Dataset(val_data, val_data['GHI'].values, lookback, horizon)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)





class Linear_LSTM(torch.nn.Module):
    def __init__(self, n_layers, nodes, horizon,n_features, return_sequences=False, dropout=False, dropout_rate=0.2):
        super(Linear_LSTM, self).__init__()
        self.n_layers = n_layers
        self.nodes = nodes
        self.horizon = horizon
        self.n_features = n_features
        self.return_sequences = return_sequences
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        
        self.lstm = torch.nn.LSTM(input_size=self.n_features, hidden_size=self.nodes, num_layers=self.n_layers, batch_first=True, dropout=self.dropout_rate if self.dropout else 0.0,dtype=torch.float64)
        self.out = torch.nn.Linear(self.nodes, self.horizon,dtype=torch.float64)
    
    def forward(self, X):
        lstm_out, _ = self.lstm(X)
        lstm_out = lstm_out[:,-1]
        y = self.out(lstm_out)
        return y
    

model = Linear_LSTM(n_layers, n_neurons[0], horizon, n_features, return_sequences=False, dropout=True, dropout_rate=dropout)
loss = torch.nn.MSELoss()
val_metrics = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

metric_dict = {"RMSE": None, "MAE": None, "skill_score": None}
extra_options = {"lookback": lookback, "horizon": horizon}
metrics = Metrics(metric_dict)
for epoch in range(max_epochs):
    losses = []
    for X, y in train_loader:
        optimizer.zero_grad()
        y_pred = model(X)
        l = loss(y_pred, y.type(torch.double)).type(torch.double)
        l.backward()
        optimizer.step()
        losses.append(l.item())
    with torch.no_grad():
        val_metrics = {}
        for X, y in val_loader:
            y_pred = model(X)
            y_pred,y = scaler.inverse_results_transform(y_pred, y)
            l = metrics(y_pred, y.type(torch.double),X,extra_options)
            for key, value in l.items():
                if key in val_metrics:
                    val_metrics[key].append(value.item())
                else:
                    val_metrics[key] = [value.item()]

    #print(f'Epoch {epoch} - Loss: {np.mean(losses)} - Validation Loss: {np.mean(val_losses)}')
    step_meta = {"Epoch": epoch, "Loss": np.mean(losses)}
    metrics.print_metrics({**step_meta, **val_metrics})