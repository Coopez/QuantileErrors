import pandas as pd
import numpy as np
import torch
from typing import Union

from pytorch_lattice.models.features import NumericalFeature, CategoricalFeature

class CalibratedDataset(torch.utils.data.Dataset):
    """A class for loading a dataset for a calibrated model."""

    def __init__(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        cs: np.ndarray = None,
        idx: np.ndarray = None,
        device: str = "cpu",
        params: str = dict(),
    ):
        Xc = X.copy()
        self.length = len(Xc)
        self.horizon_size = params['horizon_size']  
        self.window_size = params['window_size']   
        
        # self.embedded_size = params['lstm_hidden_size'][-1] if params['input_model'] == "lstm" else params['dnn_hidden_size'][-1]

        # self.FLAG_pass_CS = True if data_source == "IFE Skycam" and params['_IFE_TARGET'] == "CSI" else False


        """Initializes an instance of `Dataset`."""
        self.device = device
        # selected_features = [feature.feature_name for feature in features]
        # unavailable_features = set(selected_features) - set(Xc.columns)
        # if len(unavailable_features) > 0:
        #     raise ValueError(f"Features {unavailable_features} not found in dataset.")

        # drop_features = list(set(Xc.columns) - set(selected_features))
        # Xc.drop(drop_features, inplace=True)
        
        
        # find all quantiles in Xc and pop them into quantiles, stacking them at dim -1
        quantile_columns = [col for col in Xc.columns if "quantile" in col]
        if quantile_columns:
            self.quantiles = torch.stack([torch.from_numpy(Xc.pop(col).values) for col in quantile_columns], dim=-1).to(self.device)
        else:
            self.quantiles = None

        self.idx = torch.tensor(idx).to(device) 
        self.data = torch.from_numpy(Xc.values).to(device) 
        self.targets = torch.from_numpy(y.copy())[:, None].to(device)
        if cs is not None:
            self.cs = torch.from_numpy(cs.copy())[:, None].to(device)

    def __len__(self):
        return self.length - self.window_size - self.horizon_size + 1

    def __getitem__(self, idx):

        x = self.data[idx : idx + self.window_size]
        y = self.targets[idx + self.window_size : idx + self.window_size + self.horizon_size]
        cs = self.cs[idx + self.window_size : idx + self.window_size + self.horizon_size]
        idx = self.idx[idx : idx + self.window_size + self.horizon_size]
        return [x,y,cs,idx] 
    def return_quantile(self, batchsize,quantile_dim=1):
        """ returns a quantile tensor of shape (batchsize,window_size,quantile_dim) and a quantile range tensor of shape (quantile_dim)"""
        #TODO needs to be 1 not window_size
        if quantile_dim <= 2:
            quantile = torch.rand(1, device = self.device)
            quantiles = quantile.repeat(batchsize,1)
        if quantile_dim == 2: # put inverted quantiles in last dim
            quantiles = quantiles.unsqueeze(-1)
            quantiles = torch.cat([quantiles, 1-quantiles], dim=-1)
        if quantile_dim > 2: # make arbitrary many quantiles if required
            exponent = np.ceil(np.log10(quantile_dim))
            min_range = 1.0 / 10**exponent
            max_range = 1.0 - min_range
            quantile_range = torch.linspace(min_range, max_range, quantile_dim, device=self.device)
            quantiles = quantile_range.repeat(batchsize,1,1)    
            return quantiles, quantile_range
        return quantiles