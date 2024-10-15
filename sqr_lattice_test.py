import torch
# device = torch.device(f'cuda:{torch.cuda.current_device()}') if torch.cuda.is_available() else 'cpu'
# torch.set_default_device(device)
#TODO it seems like overarching to cuda calls do not touch subcomponents of methods with sparse tensors in pytorch 2.4.
# May need to export all relevant functions and do a manual device assign for gpu support

import pytorch_lattice.enums as enums

import numpy as np
import pandas as pd
from res.data import data_import

from models.Calibrated_lattice_model import CalibratedLatticeModel

from dataloader.calibratedDataset import CalibratedDataset
from pytorch_lattice.models.features import NumericalFeature

from losses.qr_loss import sqr_loss
from models.SQR_LSTM_Lattice import SQR_LSTM_Lattice


# Hyperparameters

_BATCHSIZE = 16
_RANDOM_SEED = 42
_LEARNING_RATE = 0.0001
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
#valid = valid[:,11]
#
# normalize train target real quick
train_target = (train_target - np.min(train_target))/(np.max(train_target) - np.min(train_target))


quantiles = np.random.uniform(0,1,len(train))
dset = {"irradiance":train,"quantiles":quantiles}
X = pd.DataFrame(dset)
y = train_target


# Data
features = [NumericalFeature("irradiance", X["irradiance"].values, num_keypoints=_NUM_KEYPOINTS), NumericalFeature("quantiles", quantiles,num_keypoints=_NUM_KEYPOINTS, monotonicity=enums.Monotonicity.INCREASING)]
data = CalibratedDataset(X, y, features, window_size=_WINDOW_SIZE,horizon_size=_PRED_LENGTH) 
dataloader = torch.utils.data.DataLoader(data, batch_size=_BATCHSIZE, shuffle=True)
# Model

lstm = SQR_LSTM_Lattice(input_size=_INPUT_SIZE_LSTM, hidden_size=_HIDDEN_SIZE_LSTM, layers=_NUM_LAYERS_LSTM, window_size=_WINDOW_SIZE, output_size=1, pred_length=_PRED_LENGTH)
features_lattice = []
gen_LSTM_out = np.random.uniform(0,1,(_BATCHSIZE,1))
for i in range(_HIDDEN_SIZE_LSTM):
    features_lattice.append(NumericalFeature(f"feature_{i}", gen_LSTM_out, num_keypoints=_NUM_KEYPOINTS))
features_lattice.append(NumericalFeature("quantiles", quantiles,num_keypoints=_NUM_KEYPOINTS, monotonicity=enums.Monotonicity.INCREASING))
#[NumericalFeature("irradiance", X["irradiance"].values, num_keypoints=_NUM_KEYPOINTS), NumericalFeature("quantiles", quantiles,num_keypoints=_NUM_KEYPOINTS, monotonicity=enums.Monotonicity.INCREASING)]


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
    train_losses = []
    for batch in dataloader:
        training_data, quantile, target = batch
        
        
        # Forward pass
        x = lstm(training_data)
        x = torch.cat((x, quantile.squeeze(-1)), dim=-1)
        output = lattice(x.squeeze())
        
        # Compute loss
        loss = criterion(output.unsqueeze(-1), target, quantile,type='pinball')
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    print(f"Epoch {epoch+1}/{epochs}, Loss: {np.mean(train_losses)}")





"""
TODO
DONE - Hyperparameter support
DONE - add Data Normalization 
DONE - Debugging monotonocity with calibration and layer layout
ISSUE - GPU support  
DONE - SQR integration
- Validation
- Test
- Neptune
- Sky cam model&data integration

"""