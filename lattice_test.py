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



# Hyperparameters

_BATCHSIZE = 16
_RANDOM_SEED = 42
_NUM_LAYERS = 1 # + 1 for the output layer. Final layer number is num_layers + 1
_NUM_KEYPOINTS = 5
_INPUT_DIM_LATTICE_FIRST_LAYER = 1
_NUM_LATTICE_FIRST_LAYER = 2



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
data = CalibratedDataset(X, y, features, window_size=1,horizon_size=1) # window can only be 1 because of the lattice.
dataloader = torch.utils.data.DataLoader(data, batch_size=_BATCHSIZE, shuffle=True)
# Model

model = CalibratedLatticeModel(features, output_min=0, output_max=1, num_layers=_NUM_LAYERS, output_size=1, input_dim_per_lattice = _INPUT_DIM_LATTICE_FIRST_LAYER, num_lattice_first_layer = _NUM_LATTICE_FIRST_LAYER, calibration_keypoints = _NUM_KEYPOINTS)
# Forward pass
# Define loss function and optimizer

criterion = sqr_loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
model.train()

epochs = 50
for epoch in range(epochs):
    train_losses = []
    for batch in dataloader:
        training_data, quantile, target = batch
        x = torch.cat((training_data, quantile), dim=-1)
        
        # Forward pass
        output = model(x.squeeze())
        
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
- Debugging monotonocity with calibration and layer layout
ISSUE - GPU support  
- SQR integration
- Validation
- Test
- Neptune
- Sky cam model&data integration

"""