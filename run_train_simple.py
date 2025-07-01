import torch
device = torch.device(f'cuda:{torch.cuda.current_device()}') if torch.cuda.is_available() else 'cpu'
torch.set_default_device(device)
#TODO GPU performance is worse than CPU unless batch size is increased. Maybe need better data loading.
### Debug tool################
from debug.plot import Debug_model
##############################
from metrics.metrics import Metrics
from metrics.metric_plots import MetricPlots
from res.data import data_import, Data_Normalizer
from res.ife_data import import_ife_data 
from dataloader.calibratedDataset import CalibratedDataset
from pytorch_lattice.models.features import NumericalFeature
import numpy as np
from losses.qr_loss import SQR_loss

from utils.helper_func import generate_surrogate_quantiles, return_features, return_Dataframe
from config import _LOG_NEPTUNE, _VERBOSE, params, _DATA_DESCRIPTION
from models.builder import build_model, build_optimizer
from training.train_loop import train_model
# from models.persistence_model import Persistence_Model
from models.smart_day_persistence import sPersistence_Forecast
from data_provider.data_loader import return_cs
import os
import pandas as pd
def train(Seed=0):

    if _LOG_NEPTUNE:
        import neptune
        from neptune.utils import stringify_unsupported
        from api_key import _NEPTUNE_API_TOKEN
        run = neptune.init_run(
            project="n1kl4s/QuantileError",
            name = params['input_model']+"-"+ params['output_model'] +"-"+params["loss"],
            api_token=_NEPTUNE_API_TOKEN,
            tags= params["neptune_tags"]
        )
        run['data/type'] = _DATA_DESCRIPTION
    else:
        run = None

    # pytorch random seed
    torch.manual_seed(Seed)

    if _LOG_NEPTUNE:
        run['parameters'] = stringify_unsupported(params) # neptune only supports float and string

    if _DATA_DESCRIPTION ==  "Station 11 Irradiance Sunpoint":
        train,train_target,valid,valid_target,_,test_target= data_import() #dtype="float64"
        _loc_data = os.getcwd()
        cs_valid, cs_test, cs_train, day_mask,cs_de_norm = return_cs(os.path.join(_loc_data,"data"))

        start_date = "2016-01-01 00:30:00"
        end_date = "2020-12-31 23:30:00"    
        index = pd.date_range(start=start_date, end = end_date, freq = '1h', tz='CET')
        i_series = np.arange(0, len(index), 1)
        train_index = i_series[len(test_target)+len(valid_target):]
        valid_index = i_series[len(test_target):len(test_target)+len(valid_target)]
        overall_time = index.values
        
    elif _DATA_DESCRIPTION == "IFE Skycam":
        train,train_target,valid,valid_target,cs_train, cs_valid, overall_time, train_index, valid_index= import_ife_data(params) # has 22 features now, 11 without preprocessing
        train,train_target,valid,valid_target, cs_train, cs_valid, overall_time= train.values,train_target.values,valid.values,valid_target.values, cs_train.values, cs_valid.values, overall_time.values
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
    data = CalibratedDataset(X, y,cs_train, train_index, device=device,params=params) 
    dataloader = torch.utils.data.DataLoader(data, batch_size=params['batch_size'], shuffle=params['train_shuffle'], generator=torch.Generator(device=device))

    data_valid = CalibratedDataset(Xv, valid_target,cs_valid,valid_index, device=device,params=params)
    data_loader_valid = torch.utils.data.DataLoader(data_valid, batch_size=params['batch_size'], shuffle=params['valid_shuffle'], generator=torch.Generator(device=device))

    features_lattice = return_features(quantiles,params,data=None)


    model = build_model(params=params,device=device,features=features_lattice).to(device)

    # Forward pass
    # Define loss function and optimizer
    if _LOG_NEPTUNE:
        run['model_summary'] = str(model)

    criterion = SQR_loss(type=params['loss'], lambda_=params['loss_calibration_lambda'], scale_sharpness=params['loss_calibration_scale_sharpness'])
    persistence = sPersistence_Forecast(Normalizer,params)

    metric = Metrics(params,Normalizer,_DATA_DESCRIPTION)
    metric_plots = MetricPlots(params,Normalizer,sample_size=params["valid_plots_sample_size"],log_neptune=_LOG_NEPTUNE)
    optimizer = build_optimizer(params, model)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of model parameters: {num_params}")
    
    model = train_model(params = params,
                    model = model,
                    optimizer = optimizer,
                    criterion = criterion,
                    metric = metric,
                    metric_plots = metric_plots,
                    dataloader = dataloader,
                    dataloader_valid = data_loader_valid,
                    data = data,
                    data_valid = data_valid,
                    log_neptune=_LOG_NEPTUNE,
                    verbose=_VERBOSE, 
                    neptune_run=  run,
                    overall_time = overall_time,
                    persistence = persistence)
    

    if _LOG_NEPTUNE:
        run.stop()

if __name__ == "__main__":
    for number,seed in enumerate(params['random_seed'],start=1):
        model = train(Seed=seed)
        model_name = f"models_save/{number}_{params['input_model']}-{params['output_model']}_test.pt"
        if params['model_save']:
            torch.save(model,model_name)