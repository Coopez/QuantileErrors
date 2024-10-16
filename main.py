import torch
device = torch.device(f'cuda:{torch.cuda.current_device()}') if torch.cuda.is_available() else 'cpu'
torch.set_default_device(device)
#TODO GPU performance is worse than CPU unless batch size is increased. Maybe need better data loading.

import pytorch_lattice.enums as enums

import numpy as np
import pandas as pd
from res.data import data_import, Data_Normalizer

from models.Calibrated_lattice_model import CalibratedLatticeModel

from dataloader.calibratedDataset import CalibratedDataset
from pytorch_lattice.models.features import NumericalFeature

from losses.qr_loss import sqr_loss
from models.SQR_LSTM_Lattice import SQR_LSTM_Lattice
import time


# Hyperparameters

_BATCHSIZE = 128
_RANDOM_SEED = 42
_LEARNING_RATE = 0.001
# LSTM Hyperparameters
_INPUT_SIZE_LSTM = 1
_HIDDEN_SIZE_LSTM = 2
_NUM_LAYERS_LSTM = 1
_WINDOW_SIZE = 1
_PRED_LENGTH = 1 # Horizon size
# Lattice Hyperparameters
_NUM_LAYERS_LATTICE = 1
_NUM_KEYPOINTS = 5
_INPUT_DIM_LATTICE_FIRST_LAYER = 1 
_NUM_LATTICE_FIRST_LAYER = _HIDDEN_SIZE_LSTM + 1 



train,train_target,valid,valid_target,_,_ = data_import()
train = train[:,11]
valid = valid[:,11]

# normalize train and valid
Normalizer = Data_Normalizer(train,train_target,valid,valid_target)
train,train_target,valid,valid_target = Normalizer.transform_all()

# Training set
quantiles = np.random.uniform(0,1,len(train))
dset = {"irradiance":train,"quantiles":quantiles}
X = pd.DataFrame(dset)
y = train_target
# Validation set
quantiles_valid = np.random.uniform(0,1,len(valid))
vset = {"irradiance":valid,"quantiles":quantiles_valid}
Xv = pd.DataFrame(vset)

# Data
features = [NumericalFeature("irradiance", X["irradiance"].values, num_keypoints=_NUM_KEYPOINTS), NumericalFeature("quantiles", quantiles,num_keypoints=_NUM_KEYPOINTS, monotonicity=enums.Monotonicity.INCREASING)]
data = CalibratedDataset(X, y, features, window_size=_WINDOW_SIZE,horizon_size=_PRED_LENGTH,device=device) 
dataloader = torch.utils.data.DataLoader(data, batch_size=_BATCHSIZE, shuffle=True,generator=torch.Generator(device=device))

data_valid = CalibratedDataset(Xv, valid_target, features, window_size=_WINDOW_SIZE,horizon_size=_PRED_LENGTH,device=device)
data_loader_valid = torch.utils.data.DataLoader(data_valid, batch_size=_BATCHSIZE, shuffle=True,generator=torch.Generator(device=device))

# Model

lstm = SQR_LSTM_Lattice(input_size=_INPUT_SIZE_LSTM, hidden_size=_HIDDEN_SIZE_LSTM, layers=_NUM_LAYERS_LSTM, window_size=_WINDOW_SIZE, output_size=1, pred_length=_PRED_LENGTH)
features_lattice = []
gen_LSTM_out = np.random.uniform(0,1,(_BATCHSIZE,1))
for i in range(_HIDDEN_SIZE_LSTM):
    features_lattice.append(NumericalFeature(f"feature_{i}", gen_LSTM_out, num_keypoints=_NUM_KEYPOINTS))
features_lattice.append(NumericalFeature("quantiles", quantiles,num_keypoints=_NUM_KEYPOINTS, monotonicity=enums.Monotonicity.INCREASING))



lattice = CalibratedLatticeModel(features_lattice, output_min=0, output_max=1, num_layers=_NUM_LAYERS_LATTICE, output_size=_PRED_LENGTH, input_dim_per_lattice = _INPUT_DIM_LATTICE_FIRST_LAYER, num_lattice_first_layer = _NUM_LATTICE_FIRST_LAYER, calibration_keypoints = _NUM_KEYPOINTS)
# Forward pass
# Define loss function and optimizer

criterion = sqr_loss
# need to have both models on the same optimizer
optimizer = torch.optim.Adam(list(lstm.parameters()) + list(lattice.parameters()), lr=_LEARNING_RATE)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
lstm.train()
lattice.train()

epochs = 50
for epoch in range(epochs):
    start_time = time.time()
    train_losses = []
    lstm.train()
    lattice.train()
    for batch in dataloader:
        training_data, quantile, target = batch
        
        # Forward pass
        x = lstm(training_data)
        x = torch.cat((x, quantile.squeeze(-1)), dim=-1)
        output = lattice(x)
        
        # Compute loss
        loss = criterion(output.unsqueeze(-1), target, quantile, type='pinball')
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    
    lstm.eval()
    lattice.eval()
    with torch.no_grad():
        valid_losses = []
        for batch in data_loader_valid:
            training_data, quantile, target = batch
            # Forward pass
            x = lstm(training_data)
            x = torch.cat((x, quantile.squeeze(-1)), dim=-1)
            output = lattice(x)
            
            # Compute loss
            loss = criterion(output.unsqueeze(-1), target, quantile, type='pinball')
            valid_losses.append(loss.item())
    
    epoch_time = time.time() - start_time
    print(f"Epoch {epoch+1:02d}/{epochs}, Loss: {np.mean(train_losses):.6f}, Validation Loss: {np.mean(valid_losses):.6f}, Time: {epoch_time:.2f}s")



"""
TODO
DONE - Hyperparameter support
DONE - add Data Normalization 
DONE - Debugging monotonocity with calibration and layer layout
ISSUE - GPU support  
DONE - SQR integration
DONE - Validation
- Test
- Neptune
- Sky cam model&data integration
- More Loss functions
- Probabilistic Metrics

"""