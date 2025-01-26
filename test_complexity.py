import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from api_key import _NEPTUNE_API_TOKEN
import neptune
from models.Calibrated_lattice_model import CalibratedLatticeModel

from dataloader.calibratedDataset import CalibratedDataset

from res.data import  Data_Normalizer
from losses.qr_loss import sqr_loss
from res.ife_data import import_ife_data 
import time

from utils.helper_func import generate_surrogate_quantiles, return_features, return_Dataframe,NumericalFeature
run = neptune.init_run(
    project="n1kl4s/QuantileError",
    name = "Lattice_Complexity_testing",
    tags = ["Lattice","Complexity"],
    api_token=_NEPTUNE_API_TOKEN,
)

# Hyperparameters

params = dict(
_BATCHSIZE = 64, # Batchsize
_RANDOM_SEED = 0, # Random seed
_SHUFFLE_train = True, # Determines if data is shuffled
_SHUFFLE_valid = True, # Determines if data is shuffled
_IFE_TARGET = 'GHI', # or 'GHI' or 'ERLING_SETTINGS'
_LEARNING_RATE = 0.0001, #0.1, # Learning rate
_EPOCHS = 300, # Number of epochs
_DETERMINISTIC_OPTIMIZATION= False, # Determines if optimization is deterministic
# LSTM Hyperparameters
_INPUT_SIZE_LSTM = 22,  # Number of features 246 if all stations of sunpoint are used or 11,22 for IFE
_HIDDEN_SIZE_LSTM = 24, # Number of nodes in hidden layer
_NUM_LAYERS_LSTM = 1, # Number of layers
_WINDOW_SIZE = 1,#30,#24, # Lookback size
_PRED_LENGTH = 1,#90,#12, # Horizon size
# Lattice Hyperparameters
_NUM_LAYERS_LATTICE = 1, # Number of layers
_NUM_KEYPOINTS = 5, # Number of keypoints
_INPUT_DIM_LATTICE_FIRST_LAYER = 12, # Number of input dimensions in first layer - from this number of lattices in layer is derived
# Extra Loss Hyperparameters
_BEYOND_LAMBDA = 0.0, # Lambda for beyond loss
_SCALE_SHARPNESS = True, # Determines if sharpness is scaled by quantile

_LOSS = 1, #Index of loss_option
loss_option = ['calibration_sharpness_loss', 'pinball_loss'],

_REGULAR_OPTIMIZER = 0, #Index of optimizer_option
optimizer_option = ['Adam', 'RAdam', 'NAdam', 'RMSprop', 'AdamW'],


_Metrics =  {"PICP": None,"ACE": None,"PINAW": None, "MAE": None, "RMSE": None, "CRPS": None}, #"Calibration": None}#{"RMSE": None, "MAE": None, "skill_score": None, 'CRPS': None}

_MODEL = 1, #Index of model_options
_model_options = ["LSTM_Lattice", "LSTM_Linear"], 

_METRICS_EVERY_X = 1 # Determines how often metrics are calculated depending on Epoch number
)

_NUM_LAYERS = 1 
_NUM_KEYPOINTS = 5
_INPUT_DIM_LATTICE_FIRST_LAYER = 15
_NUM_LATTICE_FIRST_LAYER = 2



run["lattice_params"] = dict(
    _NUM_LAYERS = _NUM_LAYERS,
    _NUM_KEYPOINTS = _NUM_KEYPOINTS,
    _INPUT_DIM_LATTICE_FIRST_LAYER = _INPUT_DIM_LATTICE_FIRST_LAYER,
    _NUM_LATTICE_FIRST_LAYER = _NUM_LATTICE_FIRST_LAYER
)
epochs = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)

train,train_target,valid,valid_target,cs_train, cs_valid= import_ife_data(params) # has 22 features now, 11 without preprocessing
column_names = train.columns
train,train_target,valid,valid_target, cs_train, cs_valid= train.values,train_target.values,valid.values,valid_target.values, cs_train.values, cs_valid.values
#valid = valid[:,11]
#
# normalize train real quick
Normalizer = Data_Normalizer(train,train_target,valid,valid_target)
train,train_target,valid,valid_target = Normalizer.transform_all()

take_out = int(len(train)/4)
train = train[:-take_out]
train_target = train_target[:-take_out]

quantiles = generate_surrogate_quantiles(len(train),params)

y = train_target
X = return_Dataframe(quantiles,train)
#X = pd.concat([train, pd.DataFrame(quantiles, columns = ['quantile'],index=train.index)], axis=1)

# Data
features = return_features(quantiles,params,data = train)
data = CalibratedDataset(X, y, features,device = device,data_source = "IFE Skycam", params=params) # window can only be 1 because of the lattice.
dataloader = torch.utils.data.DataLoader(data, batch_size=params['_BATCHSIZE'], shuffle=params['_SHUFFLE_train'], generator=torch.Generator(device=device),pin_memory=True)
# Model

model = CalibratedLatticeModel(features, output_min=0, output_max=1, num_layers=_NUM_LAYERS, output_size=1, input_dim_per_lattice = _INPUT_DIM_LATTICE_FIRST_LAYER, num_lattice_first_layer = _NUM_LATTICE_FIRST_LAYER, calibration_keypoints = _NUM_KEYPOINTS).to(device)
# Forward pass
# Define loss function and optimizer

criterion = sqr_loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

parameter_names = [name for name,params in model.named_parameters()]
parameter_sizes = [params.numel() for name,params in model.named_parameters()]
parameter_sum = sum(parameter_sizes)
print(f"Number of parameters: {parameter_sum}, Lattice Input: {_INPUT_DIM_LATTICE_FIRST_LAYER}, Input Features: {len(features)+1}")
# Training loop
model.train()
train_time = time.time()
for epoch in range(epochs):
    time_start = time.time()
    train_losses = []
    for i,batch in enumerate(dataloader):
        training_data, quantile, target = batch
        x = torch.cat((training_data, quantile), dim=-1)
        
        # Forward pass
        output = model(x.squeeze())
        
        # Compute loss
        loss = criterion(output.unsqueeze(-1), target, quantile)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    time_end = time.time()
    epoch_time = time_end - time_start
    run['train/time'].log(epoch_time)
    run['train/loss'].log(np.mean(train_losses))
    print(f"Epoch {epoch+1}/{epochs}, Loss: {np.mean(train_losses)}, Time: {epoch_time}")

train_time = time.time() - train_time
print(f"Overall training time: {train_time}")

run.stop()