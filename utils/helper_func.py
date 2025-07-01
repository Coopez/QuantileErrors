import numpy as np
import pandas as pd
from pytorch_lattice.models.features import NumericalFeature
from pytorch_lattice.enums import InputKeypointsInit,Monotonicity
from typing import Optional
import torch

def generate_surrogate_quantiles(length: int, params: dict):
    #quantiles = np.random.uniform(0,1,length)
    quantiles = np.random.uniform(0,1,length)
    #quantiles = np.repeat(quantiles, length)
    #quantiles = np.stack([quantiles,1-quantiles],axis=-1) if params['loss_option'][params['_LOSS']] == 'calibration_sharpness_loss' else quantiles
    return quantiles

def return_features(quantiles:np.ndarray,params:dict,data:np.ndarray = None):
    #TODO this function seems to be inprecise and potentially buggy, need to check the logic and the data flow
    if data is None:
        amount = params['lstm_hidden_size'][-1]
    features = []
    data_features = params['lstm_input_size'] # this is the expected number of features in the data
    quantiles = np.expand_dims(quantiles, axis=-1) if len(quantiles.shape) == 1 else quantiles
    if data is not None:
        data = np.expand_dims(data, axis=-1) if len(data.shape) == 1 else data
    
    if params['input_model'] == 'lstm':
        amount = params['lstm_hidden_size'][-1]
    elif params['input_model'] == 'dnn':
        amount = params['dnn_hidden_size'][-1]
    else:
        amount = data_features
    data = np.random.uniform(0, 1, (quantiles.shape[0], amount)) if data is None else data #.astype('f')
    
    for i in range(amount):
        features.append(NumericalFeature(f"feature_{i}", data[...,i], num_keypoints=params['lattice_calibration_num_keypoints']))
    features.append(NumericalFeature(f"quantiles_0", quantiles[...,0], num_keypoints=params['lattice_calibration_num_keypoints_quantile'], monotonicity=Monotonicity.INCREASING))
    return features

# def return_features(params:dict, data_source: str, feature_length:int, column_names: list = None):
#     if data_source == 'LSTM':
#         min_value = -1
#         max_value = 1
#     elif data_source == 'linear':
#         min_value = 0
#         max_value = 1
#     else:
#         raise ValueError('Unknown data source')
#     features = []
#     for i in range(feature_length):
#         name = f"feature_{i}" if column_names is None else column_names[i]
#         features.append(NumericalFeature(name,min_value=min_value,max_value=max_value, num_keypoints=params['_NUM_KEYPOINTS']))

#     features.append(NumericalFeature(f"quantile", num_keypoints=params['_NUM_KEYPOINTS'], monotonicity=Monotonicity.INCREASING, min_value=0, max_value=1))
#     return features



def return_Dataframe_with_q(quantiles,data):
    data = np.expand_dims(data, axis=-1) if len(data.shape) == 1 else data
    quantiles = np.expand_dims(quantiles, axis=-1) if len(quantiles.shape) == 1 else quantiles
    dset = {}
    for i in range(data.shape[-1]):
        dset[f"feature_{i}"] = data[...,i]
    for i in range(quantiles.shape[-1]):
        dset[f"quantiles_{i}"] = quantiles[...,i]
    df = pd.DataFrame(dset)
    return df

def return_Dataframe(data):
    data = np.expand_dims(data, axis=-1) if len(data.shape) == 1 else data
    dset = {}
    for i in range(data.shape[-1]):
        dset[f"feature_{i}"] = data[...,i]
    df = pd.DataFrame(dset)
    return df

# class NumericalFeature:
#     def __init__(
#         self,
#         feature_name: str,
#         num_keypoints: int = 5,
#         input_keypoints_init: InputKeypointsInit = InputKeypointsInit.UNIFORM,
#         missing_input_value: Optional[float] = None,
#         monotonicity: Optional[Monotonicity] = None,
#         projection_iterations: int = 8,
#         lattice_size: int = 2,
#         min_value: float = 0,
#         max_value: float = 1,
#     ) -> None:
       
#         self.feature_name = feature_name

#         self.num_keypoints = num_keypoints
#         self.input_keypoints_init = input_keypoints_init
#         self.missing_input_value = missing_input_value
#         self.monotonicity = monotonicity
#         self.projection_iterations = projection_iterations
#         self.lattice_size = lattice_size

#         self.input_keypoints = np.linspace(min_value, max_value, num=num_keypoints)
        

def rank_batches_by_variance(dataloader: torch.utils.data.DataLoader):
    batch_variances = []
    
    for i, batch in enumerate(dataloader):
        training_data, target,cs, idx = batch

        variance = torch.var(target)
        batch_variances.append((i, variance.item()))
    
    ranked_batches = sorted(batch_variances, key=lambda x: x[1], reverse=True)
    
    return ranked_batches