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
        features: list[Union[NumericalFeature, CategoricalFeature]],
        window_size: int,
        horizon_size: int,
        device: str = "cpu",
    ):
        self.horizon_size = horizon_size   
        self.window_size = window_size
        
        """Initializes an instance of `Dataset`."""
        self.X = X.copy()
        self.y = y.copy()
        self.device = device
        selected_features = [feature.feature_name for feature in features]
        unavailable_features = set(selected_features) - set(self.X.columns)
        if len(unavailable_features) > 0:
            raise ValueError(f"Features {unavailable_features} not found in dataset.")

        drop_features = list(set(self.X.columns) - set(selected_features))
        self.X.drop(drop_features, inplace=True)
        
        
        # find all quantiles in X and pop them into quantiles, stacking them at dim -1
        quantile_columns = [col for col in self.X.columns if "quantile" in col]
        if quantile_columns:
            self.quantiles = torch.stack([torch.from_numpy(self.X.pop(col).values).double() for col in quantile_columns], dim=-1).to(self.device)
        else:
            self.quantiles = None

        #self.quantiles = self.X.pop("quantiles") if "quantiles" in self.X.columns else None

        self.data = torch.from_numpy(self.X.values).double().to(device) 
        self.targets = torch.from_numpy(self.y)[:, None].double().to(device)

    def __len__(self):
        return len(self.X) - self.window_size - self.horizon_size + 1

    def __getitem__(self, idx):
        # if isinstance(idx, torch.Tensor):
        #     idx = idx.tolist()
        x = self.data[idx : idx + self.window_size]
        y = self.targets[idx + self.window_size : idx + self.window_size + self.horizon_size]
        # if self.quantiles is not None:
        #     quantile = torch.rand(1,dtype=torch.double, device = self.device)
        #     # expand q to x
        #     q = quantile.repeat(self.window_size)
        #     #q = self.quantiles[idx + self.window_size]
        #     #q = torch.from_numpy(q)[:, None].double().to(self.device)
        #     return [x, q,y]
        return [x, y]
    def return_quantile(self,batchsize):
        #quantile = torch.rand((batchsize),dtype=torch.double, device = self.device)
        #quantiles = quantile.repeat(self.window_size,1).reshape(batchsize,self.window_size)
        quantile = torch.rand(1,dtype=torch.double, device = self.device)
        quantiles = quantile.repeat(batchsize,self.window_size)
        return quantiles