import numpy as np

def generate_quantiles(length: int, params: dict):
    quantiles = np.random.uniform(0,1,length)
    quantiles = np.concatenate([quantiles,1-quantiles]) if params['loss_option'][params['_LOSS']] == 'calibration_sharpness_loss' else quantiles
    return quantiles