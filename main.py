import torch
device = torch.device(f'cuda:{torch.cuda.current_device()}') if torch.cuda.is_available() else 'cpu'
torch.set_default_device(device)
#TODO GPU performance is worse than CPU unless batch size is increased. Maybe need better data loading.
import numpy as np
import pandas as pd
import pytorch_lattice.enums as enums
import time
### Debug tool################
from debug.plot import Debug_model
##############################
from metrics.metrics import Metrics
from res.data import data_import, Data_Normalizer, Batch_Normalizer
from res.ife_data import import_ife_data 
from dataloader.calibratedDataset import CalibratedDataset
from pytorch_lattice.models.features import NumericalFeature

from losses.qr_loss import SQR_loss, sharpness_loss
from models.LSTM_Lattice import LSTM_Lattice

from utils.helper_func import generate_surrogate_quantiles, return_features, return_Dataframe
from config import _LOG_NEPTUNE, _VERBOSE, params, _DATA_DESCRIPTION

if _LOG_NEPTUNE:
    import neptune
    from neptune.utils import stringify_unsupported
    from api_key import _NEPTUNE_API_TOKEN
    run = neptune.init_run(
        project="n1kl4s/QuantileError",
        name = params['_model_options'][params['_MODEL']] +"-"+ str(params["_HIDDEN_SIZE_LSTM"]) +"-"+ str(params["_NUM_LAYERS_LSTM"])+"-"+ str(params["_NUM_LAYERS_LATTICE"])+"-"+ str(params["_NUM_KEYPOINTS"])+"-"+params["loss_option"][params['_LOSS']],
        api_token=_NEPTUNE_API_TOKEN,
    )
    run['data/type'] = _DATA_DESCRIPTION


# pytorch random seed
torch.manual_seed(params['_RANDOM_SEED'])

_NUM_LATTICE_FIRST_LAYER = params['_HIDDEN_SIZE_LSTM'] + 1
if _LOG_NEPTUNE:
    run['parameters'] = stringify_unsupported(params) # neptune only supports float and string

if _DATA_DESCRIPTION ==  "Station 11 Irradiance Sunpoint":
    train,train_target,valid,valid_target,_,_ = data_import()
    if params['_INPUT_SIZE_LSTM'] == 1:
        train = train[:,11] # disable if training on all features/stations
    valid = valid[:,11]
elif _DATA_DESCRIPTION == "IFE Skycam":
    train,train_target,valid,valid_target= import_ife_data(params) # has 11 features
    train,train_target,valid,valid_target= train.values,train_target.values,valid.values,valid_target.values
else:
    raise ValueError("Data description not implemented")

# normalize train and valid
Normalizer = Data_Normalizer(train,train_target,valid,valid_target)
train,train_target,valid,valid_target = Normalizer.transform_all()

# Training set
quantiles = generate_surrogate_quantiles(len(train),params)

y = train_target
X = return_Dataframe(quantiles,train)

# Validation set
quantiles_valid = generate_surrogate_quantiles(len(valid),params)

Xv = return_Dataframe(quantiles_valid,valid)
features = return_features(quantiles,params,data=train)
data = CalibratedDataset(X, y, features, window_size=params['_WINDOW_SIZE'], horizon_size=params['_PRED_LENGTH'], device=device) 
dataloader = torch.utils.data.DataLoader(data, batch_size=params['_BATCHSIZE'], shuffle=params['_SHUFFLE_train'], generator=torch.Generator(device=device))

data_valid = CalibratedDataset(Xv, valid_target, features, window_size=params['_WINDOW_SIZE'], horizon_size=params['_PRED_LENGTH'], device=device)
data_loader_valid = torch.utils.data.DataLoader(data_valid, batch_size=params['_BATCHSIZE'], shuffle=params['_SHUFFLE_valid'], generator=torch.Generator(device=device))

features_lattice = return_features(quantiles,params,data=None,LSTM_out=params['_HIDDEN_SIZE_LSTM'])

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
    'device':device
    }
model = LSTM_Lattice(lstm_paras, lattice_paras,
                     params=params).to(device)

# Forward pass
# Define loss function and optimizer
if _LOG_NEPTUNE:
    run['model_summary'] = str(model)

criterion = SQR_loss(type=params['loss_option'][params['_LOSS']], lambda_=params['_BEYOND_LAMBDA'], scale_sharpness=params['_SCALE_SHARPNESS'])
metric = Metrics(params,Normalizer)

if params['_DETERMINISTIC_OPTIMIZATION']:
    from pytorch_minimize.optim import MinimizeWrapper
    minimizer_args = dict(method='SLSQP', options={'disp':True, 'maxiter':100}) # supports a range of methods
    optimizer = MinimizeWrapper(model.parameters(), minimizer_args)
else:
    optimizer_class = getattr(torch.optim, params['optimizer_option'][params['_REGULAR_OPTIMIZER']])
    optimizer = optimizer_class(model.parameters(), lr=params['_LEARNING_RATE'])

# Training loop
extra_options = {"lookback": params["_WINDOW_SIZE"], "horizon": params["_PRED_LENGTH"]}
epochs = params['_EPOCHS']
if params['_DETERMINISTIC_OPTIMIZATION']:
    epochs = 1
for epoch in range(epochs):
    start_time = time.time()
    train_losses = []
    sharp_losses = []
    model.train()
    for training_data, target in dataloader:
        quantile = data.return_quantile(training_data.shape[0])
        Batchnorm_train = Batch_Normalizer(training_data)
        if params['_DETERMINISTIC_OPTIMIZATION']:
            def closure():
                optimizer.zero_grad()
                output = model(training_data,quantile)
                loss = criterion(output, target, quantile.unsqueeze(-1))
                #loss.backward()
                return loss
            optimizer.step(closure)
        else:
            # Normal forward pass with batchnorm
            # training_data = Batchnorm_train.transform(training_data)
            output = model(training_data,quantile)
            # if params["_INPUT_SIZE_LSTM"] == 1:
            #     output = Batchnorm_train.inverse_transform(output)
            # else:
            #     output = Batchnorm_train.inverse_transform(output,pos=11)
            # Compute loss
            loss = criterion(output, target, quantile.unsqueeze(-1))

            #Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            if params['loss_option'][params['_LOSS']] == 'calibration_sharpness_loss':
                with torch.no_grad():
                    sharpness = sharpness_loss(output,quantile.unsqueeze(-1),scale_sharpness_scale=params['_SCALE_SHARPNESS'])
                    sharp_losses.append(sharpness.item())
            if _LOG_NEPTUNE:
                run['train/'+params['loss_option'][params['_LOSS']]].log(np.mean(train_losses))
                if params['loss_option'][params['_LOSS']] == 'calibration_sharpness_loss':
                    run['train/sharpness'].log(np.mean(sharp_losses))
    model.eval()
    metric_dict = {}
    if epoch % params['_METRICS_EVERY_X'] == 0:
        # Validation
        with torch.no_grad():
            #valid_losses = []
            
            for batch in data_loader_valid:
                
                training_data, target = batch
                quantile = data_valid.return_quantile(training_data.shape[0])
                # Batchnorm_valid = Batch_Normalizer(training_data)
                # Forward pass
                # training_data = Batchnorm_valid.transform(training_data)
                output = model(training_data,quantile)
                # if params["_INPUT_SIZE_LSTM"] == 1:
                #     output = Batchnorm_valid.inverse_transform(output)
                # else:
                #     output = Batchnorm_valid.inverse_transform(output,pos=11)
                # Compute loss
                #criterion(output, target, quantile.unsqueeze(-1),
                        #         type=params['loss_option'][params['_LOSS']])
                #deb = Debug_model(model,training_data,target)
                #deb.plot_out()
                loss = metric(output, target.type(torch.double),input=training_data ,model=model,quantile=quantile.unsqueeze(-1),options = extra_options)
                
                for key, value in loss.items():
                    # value is a tensor, we need to extract the value, but they may be a dict
                    # if isinstance(value, dict):
                    #     value = [v.item() for v in value.values()]
                    # else:
                    #     value = value.item()
                    
                    if key in metric_dict:
                        metric_dict[key].append(value)
                    else:
                        metric_dict[key] = [value]
                    #valid_losses.append(loss.item())
        if _LOG_NEPTUNE:
            #log all metrics in metric_dict to neptune
            for key, value in metric_dict.items():
                # test if value is a list 
                if isinstance(value[0], np.ndarray): # need to check if in the list of batch accumulated values we have an array
                        value = np.stack(value,axis=-1).mean(axis=-1).tolist()
                        run['valid/'+key].log(stringify_unsupported(value))
                else:
                    run['valid/'+key].log(np.mean(value))

    
    epoch_time = time.time() - start_time
    step_meta = {"Epoch": f"{epoch+1:02d}/{epochs}", "Time": epoch_time , "Train_Loss": np.mean(train_losses)}
    if params['loss_option'][params['_LOSS']] == 'calibration_sharpness_loss':
        step_meta["Sharpness"] = np.mean(sharp_losses)
    if _VERBOSE:
        metric.print_metrics({**step_meta, **metric_dict})



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
DONE - Data denorm for validation
DONE  - Update Neptune


WAIT - GPU optimization: Batch size, Data loading
WAIT - Test
50%  - Erling sky cam model&data integration
WAIT - More Loss functions
60%  - Probabilistic Metrics
00% - Adapt to ML nodes


Need to log/check quantile epoch distribution?
"""
