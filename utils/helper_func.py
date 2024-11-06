import numpy as np
import pandas as pd
from pytorch_lattice.models.features import NumericalFeature
from pytorch_lattice.enums import Monotonicity


def generate_quantiles(length: int, params: dict):
    quantiles = np.random.uniform(0,1,length)
    #quantiles = np.stack([quantiles,1-quantiles],axis=-1) if params['loss_option'][params['_LOSS']] == 'calibration_sharpness_loss' else quantiles
    return quantiles

def return_features(quantiles:np.ndarray,params:dict,data:np.ndarray = None,LSTM_out: int=None):
    assert data is not None or LSTM_out is not None, "Either data or LSTM_out must be provided"
    features = []
    data_features = params['_INPUT_SIZE_LSTM'] # this is the expected number of features in the data
    quantiles = np.expand_dims(quantiles, axis=-1) if len(quantiles.shape) == 1 else quantiles
    if data is not None:
        data = np.expand_dims(data, axis=-1) if len(data.shape) == 1 else data
    

    amount = data_features if LSTM_out is None else LSTM_out
    data = np.random.uniform(0, 1, (quantiles.shape[0], amount)) if data is None else data
    
    for i in range(amount):
        features.append(NumericalFeature(f"feature_{i}", data[...,i], num_keypoints=params['_NUM_KEYPOINTS']))
    # for i in range(quantiles.shape[-1]):
    #     features.append(NumericalFeature(f"quantiles_{i}", quantiles[...,i], num_keypoints=params['_NUM_KEYPOINTS'], monotonicity=Monotonicity.INCREASING))
    features.append(NumericalFeature(f"quantiles_0", quantiles[...,0], num_keypoints=params['_NUM_KEYPOINTS'], monotonicity=Monotonicity.INCREASING))
    return features

def return_Dataframe(quantiles,data):
    data = np.expand_dims(data, axis=-1) if len(data.shape) == 1 else data
    quantiles = np.expand_dims(quantiles, axis=-1) if len(quantiles.shape) == 1 else quantiles
    dset = {}
    for i in range(data.shape[-1]):
        dset[f"feature_{i}"] = data[...,i]
    for i in range(quantiles.shape[-1]):
        dset[f"quantiles_{i}"] = quantiles[...,i]
    df = pd.DataFrame(dset)
    return df
