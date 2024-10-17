import torch
device = torch.device(f'cuda:{torch.cuda.current_device()}') if torch.cuda.is_available() else 'cpu'
torch.set_default_device(device)
#TODO GPU performance is worse than CPU unless batch size is increased. Maybe need better data loading.

import pytorch_lattice.enums as enums

import numpy as np
import pandas as pd
from res.data import data_import, Data_Normalizer

from models.Calibrated_lattice_model import CalibratedLatticeModel

from dataloader.calibratedDataset import CalibratedDataset
from pytorch_lattice.models.features import NumericalFeature

from losses.qr_loss import sqr_loss
from models.LSTM_Lattice import LSTM_Lattice
import time

from config import _LOG_NEPTUNE

if _LOG_NEPTUNE:
    import neptune
    from config import _DATA_DESCRIPTION
    from api_key import _NEPTUNE_API_TOKEN
    run = neptune.init_run(
        project="n1kl4s/QuantileError",
        api_token=_NEPTUNE_API_TOKEN,
    )
    run['data/type'] = _DATA_DESCRIPTION

from config import params
_NUM_LATTICE_FIRST_LAYER = params['_HIDDEN_SIZE_LSTM'] + 1
if _LOG_NEPTUNE:
    run['parameters'] = params
    
train,train_target,valid,valid_target,_,_ = data_import()
train = train[:,11]
valid = valid[:,11]

# normalize train and valid
Normalizer = Data_Normalizer(train,train_target,valid,valid_target)
train,train_target,valid,valid_target = Normalizer.transform_all()

# Training set
quantiles = np.random.uniform(0,1,len(train))
dset = {"irradiance":train,"quantiles":quantiles}
X = pd.DataFrame(dset)
y = train_target
# Validation set
quantiles_valid = np.random.uniform(0,1,len(valid))
vset = {"irradiance":valid,"quantiles":quantiles_valid}
Xv = pd.DataFrame(vset)

# Data
features = [NumericalFeature("irradiance", X["irradiance"].values, num_keypoints=params['_NUM_KEYPOINTS']), 
            NumericalFeature("quantiles", quantiles, num_keypoints=params['_NUM_KEYPOINTS'], monotonicity=enums.Monotonicity.INCREASING)]
data = CalibratedDataset(X, y, features, window_size=params['_WINDOW_SIZE'], horizon_size=params['_PRED_LENGTH'], device=device) 
dataloader = torch.utils.data.DataLoader(data, batch_size=params['_BATCHSIZE'], shuffle=True, generator=torch.Generator(device=device))

data_valid = CalibratedDataset(Xv, valid_target, features, window_size=params['_WINDOW_SIZE'], horizon_size=params['_PRED_LENGTH'], device=device)
data_loader_valid = torch.utils.data.DataLoader(data_valid, batch_size=params['_BATCHSIZE'], shuffle=True, generator=torch.Generator(device=device))

features_lattice = []
gen_LSTM_out = np.random.uniform(0, 1, (params['_BATCHSIZE'], 1))
for i in range(params['_HIDDEN_SIZE_LSTM']):
    features_lattice.append(NumericalFeature(f"feature_{i}", gen_LSTM_out, num_keypoints=params['_NUM_KEYPOINTS']))
features_lattice.append(NumericalFeature("quantiles", quantiles, num_keypoints=params['_NUM_KEYPOINTS'], monotonicity=enums.Monotonicity.INCREASING))

# Model Definition
lstm_paras = {
    'input_size':params['_INPUT_SIZE_LSTM'],
    'hidden_size':params['_HIDDEN_SIZE_LSTM'],
    'num_layers':params['_NUM_LAYERS_LSTM'],
    'window_size':params['_WINDOW_SIZE'],
    'pred_length':params['_PRED_LENGTH'],
    }
lattice_paras = {
    'features':features_lattice,
    'clip_inputs': None,
    'output_min':None,
    'output_max':None,
    'kernel_init':None,
    'interpolation':None,
    'num_layers':params['_NUM_LAYERS_LATTICE'],
    'input_dim_per_lattice':params['_INPUT_DIM_LATTICE_FIRST_LAYER'],
    'num_lattice_first_layer':_NUM_LATTICE_FIRST_LAYER,
    'output_size':params['_PRED_LENGTH'],
    'calibration_keypoints':params['_NUM_KEYPOINTS'],
    }
model = LSTM_Lattice(lstm_paras, lattice_paras)

# Forward pass
# Define loss function and optimizer
if _LOG_NEPTUNE:
    run['model_summary'] = str(model) #str(lstm) + str(lattice)

criterion = sqr_loss


if params['_DETERMINISTIC_OPTIMIZATION']:
    from pytorch_minimize.optim import MinimizeWrapper
    minimizer_args = dict(method='SLSQP', options={'disp':True, 'maxiter':100}) # supports a range of methods
    optimizer = MinimizeWrapper(model.parameters(), minimizer_args)
else:
    #optimizer = torch.optim.RAdam(list(lstm.parameters()) + list(lattice.parameters()), lr=params['_LEARNING_RATE'])
    #optimizer = torch.optim.NAdam(list(lstm.parameters()) + list(lattice.parameters()), lr=params['_LEARNING_RATE'])
    optimizer = torch.optim.Adam(model.parameters(), lr=params['_LEARNING_RATE'])
    #optimizer = torch.optim.RMSprop(list(lstm.parameters()) + list(lattice.parameters()), lr=params['_LEARNING_RATE'])
    #optimizer = torch.optim.AdamW(list(lstm.parameters()) + list(lattice.parameters()), lr=params['_LEARNING_RATE'])

# Training loop
# lstm.train()
# lattice.train()

epochs = params['_EPOCHS']
if params['_DETERMINISTIC_OPTIMIZATION']:
    epochs = 1
for epoch in range(epochs):
    start_time = time.time()
    train_losses = []
    model.train()
    for training_data, quantile, target in dataloader:
        if params['_DETERMINISTIC_OPTIMIZATION']:
            def closure():
                optimizer.zero_grad()
                output = model(training_data,quantile)
                loss = criterion(output.unsqueeze(-1), target, quantile, type='pinball')
                #loss.backward()
                return loss
            optimizer.step(closure)
        else:
            # Normal forward pass
            output = model(training_data,quantile)
            # Compute loss
            loss = criterion(output.unsqueeze(-1), target, quantile, type='pinball')
            
            #Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            if _LOG_NEPTUNE:
                run['train/loss'].log(np.mean(train_losses))
    # lstm.eval()
    # lattice.eval()
    model.eval()
    with torch.no_grad():
        valid_losses = []
        for batch in data_loader_valid:
            training_data, quantile, target = batch
            # Forward pass
            output = model(training_data,quantile)
            
            # Compute loss
            loss = criterion(output.unsqueeze(-1), target, quantile, type='pinball')
            valid_losses.append(loss.item())
    if _LOG_NEPTUNE:
        run['valid/loss'].log(np.mean(valid_losses))
    epoch_time = time.time() - start_time
    print(f"Epoch {epoch+1:02d}/{epochs}, Loss: {np.mean(train_losses):.6f}, Validation Loss: {np.mean(valid_losses):.6f}, Time: {epoch_time:.2f}s")


if _LOG_NEPTUNE:
    run.stop()


"""
TODO
DONE - Hyperparameter support
DONE - add Data Normalization 
DONE - Debugging monotonocity with calibration and layer layout
DONE - GPU support  
DONE - SQR integration
DONE - Validation
DONE - Neptune
DONE - Optimizer experiments
DONE - Metric initialization and looking at potential packages

WAIT - GPU optimization: Need more consideration towards optimal batch size, and data loading.
WAIT - Test
- Erling sky cam model&data integration
WAIT - More Loss functions
- Probabilistic Metrics

"""