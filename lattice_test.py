import torch
import torch.nn as nn

import pytorch_lattice.layers as llayers
import pytorch_lattice.enums as enums

import numpy as np
import pandas as pd
from res.data import data_import
_RANDOM_SEED = 42
from calibrated_lattice_layer import CalibratedLatticeLayer, CalibratedDataset
from pytorch_lattice.models.features import NumericalFeature
class LatticeNet(nn.Module):
    """
    LatticeNet is a neural network module that utilizes lattice layers for 
    learning monotonic functions. It consists of a numerical calibrator and 
    a lattice ensemble.
    Attributes:
        batch_size (int): The size of the batches used during training.
        input_size (int): The size of the input features.
        output_size (int): The size of the output features.
        num_lattices (int): The number of lattices used in the lattice ensemble.
        horizon_size (int): The horizon size for the lattice ensemble.
        calibrator (llayers.NumericalCalibrator): A numerical calibrator layer.
        lattice_ensemble (llayers.RTL): A lattice ensemble layer.
    Methods:
        forward(x):
            Passes the input through the calibrator and lattice ensemble layers.
            Args:
                x (torch.Tensor): The input tensor.
            Returns:
                torch.Tensor: The output tensor after passing through the layers.
    """
    def __init__(self, batch_size, input_size, output_size, num_lattices, horizon_size,control):
        super(LatticeNet, self).__init__()
         
        self.batch_size = batch_size
        self.input_size = input_size
        self.output_size = output_size
        self.num_lattices = num_lattices
        self.horizon_size = horizon_size
        self.control = control
        
        
        self.control_calibrator = llayers.NumericalCalibrator(
            input_keypoints=np.linspace(0., 1., num=5),
            output_min=0.0, output_max = 1.0,
            monotonicity = enums.Monotonicity.INCREASING,
            kernel_init= enums.NumericalCalibratorInit.EQUAL_HEIGHTS
            
              )
        self.calibrator = llayers.NumericalCalibrator(
            input_keypoints=np.linspace(0., 1., num=5),
            output_min=0.0, output_max = 1.0,
            monotonicity = None,
            kernel_init= enums.NumericalCalibratorInit.EQUAL_HEIGHTS
        )



        #self.lattice = pyl.Lattice(input_size, output_size, num_lattices, horizon_size)
        self.lattice_ensemble = llayers.RTL(
            monotonicities = [enums.Monotonicity.INCREASING]*input_size,
            num_lattices = 1,
            lattice_rank= horizon_size,
            lattice_size = output_size,
            output_min = 0.0,
            output_max = 1.0,
            kernel_init = enums.LatticeInit.LINEAR,
            interpolation=enums.Interpolation.HYPERCUBE,
            random_seed=_RANDOM_SEED
        )
    def forward(self, x):
        x = self.calibrator(x)
        x = self.lattice_ensemble(x)

        return x
    

# Examples time series

_BATCHSIZE = 16


train,train_target,valid,valid_target,_,_ = data_import()
train = train[:,11]
valid = valid[:,11]


X = pd.DataFrame(train, columns=["irradiance"])
y = valid

quantiles = torch.rand((_BATCHSIZE,len(y),1))
# Data
features = [NumericalFeature("irradiance", X["irradiance"].values), NumericalFeature("quantiles", quantiles,monotonicity=enums.Monotonicity.INCREASING)]
data = CalibratedDataset(X, y, quantiles, features, window_size=4,horizon_size=1, batch_size=_BATCHSIZE)

# Model
model = CalibratedLatticeLayer(features, lattice_type='rtl')

# Forward pass
for batch in data:
    training_data,quantile, target = batch
    x = torch.stack((training_data,quantile),dim=-1)
    output = model(x)




"""
1. quantiles are a variable which needs monotonous control
2. rest not
3. variables mayhaps needs headers via pandas

3 every variables needs to be embedded in the lattice seperately. 

"""