import torch
import torch.nn as nn

import pytorch_lattice.layers as llayers
import pytorch_lattice.enums as enums

import numpy as np
import pandas as pd
from res.data import data_import

from models.Calibrated_lattice_model import CalibratedLatticeModel
from layers.calibrated_lattice_layer import CalibratedLatticeLayer
from dataloader.calibratedDataset import CalibratedDataset
from pytorch_lattice.models.features import NumericalFeature

from losses.qr_loss import sqr_loss
# Examples time series

_BATCHSIZE = 16
_RANDOM_SEED = 42

train,train_target,valid,valid_target,_,_ = data_import()
train = train[:,11]
#valid = valid[:,11]

quantiles = np.random.uniform(0,1,len(train))
dset = {"irradiance":train,"quantiles":quantiles}
X = pd.DataFrame(dset)
y = train_target


# Data
features = [NumericalFeature("irradiance", X["irradiance"].values), NumericalFeature("quantiles", quantiles,monotonicity=enums.Monotonicity.INCREASING)]
data = CalibratedDataset(X, y, features, window_size=1,horizon_size=1) # window can only be 1 because of the lattice.
dataloader = torch.utils.data.DataLoader(data, batch_size=_BATCHSIZE, shuffle=True)
# Model
#model = CalibratedLatticeLayer(features, lattice_type='rtl')
model = CalibratedLatticeModel(features, output_min=0, output_max=1,lattice_type='rtl', num_layers=1, output_size=1)
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
1. quantiles are a variable which needs monotonous control
2. rest not
3. variables mayhaps needs headers via pandas

3 every variables needs to be embedded in the lattice seperately. 

"""