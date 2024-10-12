import torch
import torch.nn as nn  
from res.model import LSTMWithAttention
from res.data import SimpleDataset, data_import
import neptune
import numpy as np
# Make a neptune experiment
LOG = True

if LOG:
    run = neptune.init_run(
        project='n1kl4s/timeseries-test',
        api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhNzA1MjE5Yy04ZTc4LTRjNDEtOWU4Yi00OTE1NWRkN2JjNTAifQ==', 
        name='n1kl4s',)

# define parameters

params = {'input_size': 1,
            'hidden_size': 32,
            'num_layers': 2,
            'learning_rate': 0.00001,
            'num_epochs': 100,
            'window': 64,
            'horizon': 24}

if LOG:
    run['parameters'] = params

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
train,train_target,valid,valid_target,_,_ = data_import()
train = train[:,11]
valid = valid[:,11]
data_train = SimpleDataset(window=params['window'], horizon=params['horizon'],targets=train_target, data=train)
train_loader = torch.utils.data.DataLoader(data_train, batch_size=3, shuffle=True)

# load validation set
val_dataset = SimpleDataset(window=params['window'], horizon=params['horizon'],targets=valid_target, data=valid)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=params['horizon'], shuffle=False)

# log data

if LOG:
    run['data/type'] = 'Station 11 irradiance Sunpoint' # data description
    run['train/size'] = len(train)
    run['valid/size'] = len(valid)


# Define the model
model = LSTMWithAttention(params['input_size'], params['hidden_size'], params['num_layers'], params['horizon']).to(device)

if LOG:
    run['model_summary'] = str(model)

# Define the loss function and optimizer

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])


# Train the model
for epoch in range(params['num_epochs']):
    train_losses = []
    for i, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)
        loss = criterion(y_pred, y)
        train_losses.append(loss.item())
        #print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if LOG:
        run['train/loss'].log(np.mean(np.array(train_losses)))
    # test the model
    with torch.no_grad():
        y_pred_series = np.zeros((len(val_loader)*params["horizon"]))
        y_series = np.zeros((len(val_loader)*params["horizon"]))
        losses = []
        for i, (x, y) in enumerate(val_loader):
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            losses.append(loss.item())
            h = params['horizon']
            y_pred_series[i*h:(i+1)*h]=y_pred.detach().cpu().numpy()[0,:]
            y_series[i*h:(i+1)*h]=y.detach().cpu().numpy()[0,:]
        if LOG:
            if epoch != 0:
                del run['valid/series_summer']
                del run['valid/series_winter']
                del run['valid/pred_series_summer']
                del run['valid/pred_series_winter']
            run['valid/series_summer'].extend(list(y_series[4000:4100]))
            run['valid/series_winter'].extend(list(y_series[8000:8100]))
            run['valid/pred_series_summer'].extend(list(y_pred_series[4000:4100]))
            run['valid/pred_series_winter'].extend(list(y_pred_series[8000:8100]))
            run['valid/loss'].log(np.mean(np.array(losses)))
        print(f'Epoch {epoch+1}, Loss: {np.mean(np.array(losses))}')
if LOG:
    # save last model
    torch.save(model.state_dict(), 'model.pth')
    run['model'].upload('model.pth')
    run.stop()
    