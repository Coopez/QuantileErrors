import torch
device = torch.device(f'cuda:{torch.cuda.current_device()}') if torch.cuda.is_available() else 'cpu'
torch.set_default_device(device)
#TODO GPU performance is worse than CPU unless batch size is increased. Maybe need better data loading.
### Debug tool################
from debug.plot import Debug_model
##############################
from metrics.metrics import Metrics
from res.data import data_import, Data_Normalizer
from res.ife_data import import_ife_data 
from dataloader.calibratedDataset import CalibratedDataset
from pytorch_lattice.models.features import NumericalFeature

from losses.qr_loss import SQR_loss

from utils.helper_func import generate_surrogate_quantiles, return_features, return_Dataframe
from config import _LOG_NEPTUNE, _VERBOSE, params, _DATA_DESCRIPTION
from models.builder import build_model, build_optimizer
from training.train_loop import train_model
if _LOG_NEPTUNE:
    import neptune
    from neptune.utils import stringify_unsupported
    from api_key import _NEPTUNE_API_TOKEN
    run = neptune.init_run(
        project="n1kl4s/QuantileError",
        name = params['input_model']+"-"+ params['output_model'] +"-"+params["loss_option"][params['_LOSS']],
        api_token=_NEPTUNE_API_TOKEN,
        tags= params["neptune_tags"]
    )
    run['data/type'] = _DATA_DESCRIPTION
else:
    run = None


# pytorch random seed
torch.manual_seed(params['random_seed'])

if _LOG_NEPTUNE:
    run['parameters'] = stringify_unsupported(params) # neptune only supports float and string

if _DATA_DESCRIPTION ==  "Station 11 Irradiance Sunpoint":
    train,train_target,valid,valid_target,_,_ = data_import()
    if params['_INPUT_SIZE_LSTM'] == 1:
        train = train[:,11] # disable if training on all features/stations
    valid = valid[:,11]
elif _DATA_DESCRIPTION == "IFE Skycam":
    train,train_target,valid,valid_target,cs_train, cs_valid= import_ife_data(params) # has 22 features now, 11 without preprocessing
    train,train_target,valid,valid_target, cs_train, cs_valid= train.values,train_target.values,valid.values,valid_target.values, cs_train.values, cs_valid.values
else:
    raise ValueError("Data description not implemented")

# normalize train and valid
Normalizer = Data_Normalizer(train,train_target,valid,valid_target)
train,train_target,valid,valid_target = Normalizer.transform_all()
# Training set
quantiles = generate_surrogate_quantiles(len(train),params)
y = train_target
X = return_Dataframe(train)
# Validation set
# quantiles_valid = generate_surrogate_quantiles(len(valid),params)

Xv = return_Dataframe(valid)
#features = return_features(quantiles,params,data=train)
data = CalibratedDataset(X, y,cs_train, device=device,params=params) 
dataloader = torch.utils.data.DataLoader(data, batch_size=params['batch_size'], shuffle=params['train_shuffle'], generator=torch.Generator(device=device))

data_valid = CalibratedDataset(Xv, valid_target,cs_valid, device=device,params=params)
data_loader_valid = torch.utils.data.DataLoader(data_valid, batch_size=params['batch_size'], shuffle=params['valid_shuffle'], generator=torch.Generator(device=device))

features_lattice = return_features(quantiles,params,data=None)


model = build_model(params=params,device=device,features=features_lattice).to(device)

# Forward pass
# Define loss function and optimizer
if _LOG_NEPTUNE:
    run['model_summary'] = str(model)

criterion = SQR_loss(type=params['loss'], lambda_=params['loss_calibration_lambda'], scale_sharpness=params['loss_calibration_scale_sharpness'])
metric = Metrics(params,Normalizer,_DATA_DESCRIPTION)

optimizer = build_optimizer(params, model)

model = train_model(params = params,
                model = model,
                optimizer = optimizer,
                criterion = criterion,
                metric = metric,
                dataloader = dataloader,
                dataloader_valid = data_loader_valid,
                data = data,
                data_valid = data_valid,
                log_neptune=_LOG_NEPTUNE,
                verbose=_VERBOSE, 
                neptune_run=  run)


if _LOG_NEPTUNE:
    run.stop()


