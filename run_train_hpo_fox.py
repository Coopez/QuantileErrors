import torch
### Debug tool################
from debug.plot import Debug_model
##############################
from metrics.metrics import Metrics
from metrics.metric_plots import MetricPlots
from res.data import data_import, Data_Normalizer
from res.ife_data import import_ife_data 
from dataloader.calibratedDataset import CalibratedDataset
from data_provider.data_loader import return_cs
from losses.qr_loss import SQR_loss

from utils.helper_func import generate_surrogate_quantiles, return_features, return_Dataframe
from config import _LOG_NEPTUNE, _VERBOSE, params, _DATA_DESCRIPTION
from models.builder import build_model, build_optimizer
from training.train_loop import train_model

import optuna
import logging
import sys
import os
import pandas as pd
from losses.qr_loss import sharpness_loss
import numpy as np
from neptune.utils import stringify_unsupported
import neptune.integrations.optuna as optuna_utils
# from models.persistence_model import Persistence_Model
from models.smart_day_persistence import sPersistence_Forecast
from torch.optim.lr_scheduler import ReduceLROnPlateau

CHECKPOINT_DIR = "optuna_checkpoint"


BASE_PATH = os.environ['SLURM_SUBMIT_DIR']
os.makedirs(BASE_PATH, exist_ok=True)

RUN_NAME = "HO_SMNN_8" # "HO_LSTM_LINEAR" # "HO_LSTM_DNN" # "HO_DNN_LINEAR" # "HO_QLATTICE" # "HO_LSTM_CONSTRAINED_LINEAR"

"""
We already had:
HPO is the old name. We are going with HO now 
LSTM_LINEAR-0 - test run
LSTM_LINEAR-1 - run with scheduler tuned to ACE. 
LSTM_LINEAR-2 - run with scheduler tuned to CRPS.
LSTM_LINEAR-3 - testing larger hidden size up to 256

LSTM_LINEAR- 6 - ACE
LSTM_LINEAR- 7 - CRPS

LSTM_DNN-0 - run with scheduler tuned to CRPS.
LSTM_DNN-1 - testing larger hidden size up to 256
LSTM_DNN-2 - 
LSTM_DNN-3 - 
LSTM_DNN-4 - ACe
LSTM_DNN-5 - crps

LSTM_CONSTRAINED_LINEAR-0 - ACE
LSTM_CONSTRAINED_LINEAR-1 - CRPS


DNN_LINEAR-0 - initial run with up to 4 layers and 128 ns

QLATTICE-0 - initial run. with only ACE - botched. Ran with dnn
QLATTICE-1 - initial run. with only CRPS - botched. Ran with dnn
QLATTICE-2 - initial run. with only ACE
QLATTICE-3 - initial run. with only CRPS
QLATTICE-4 - SCRAPED RUN
QLATTICE-5 - only CRPS, more parameter settings
QLATTICE-6 - only ACE, more parameter settings
QLATTICE-7 - only Ace, even more parameter settings,
QLATTICE-8 - only crps, even more parameter settings, 

"""

def objective(trial):
    print(f"STUDY: Starting trial: {trial.number}")
    #### HPO Logic #####
    # hpo_hidden_size = trial.suggest_categorical("hidden_size", params["hpo_hidden_size"])
    # hpo_num_layers = trial.suggest_categorical("num_layers", params["hpo_num_layers"])
    # params["learning_rate"] = trial.suggest_categorical("learning_rate", params["hpo_lr"])
    # # params["window_size"] = trial.suggest_categorical("window_size", params["hpo_window_size"])

    # if params["input_model"] == 'lstm':
    #     params['lstm_hidden_size'] = [hpo_hidden_size] * hpo_num_layers
    #     params['lstm_num_layers'] = hpo_num_layers
    # if params["input_model"] == 'dnn':
    #     params['dnn_hidden_size'] = [hpo_hidden_size] * hpo_num_layers
    #     params['dnn_num_layers'] = hpo_num_layers
    # params["lattice_dim_input"] = trial.suggest_categorical("lattice_dim_input", params["hpo_lattice_dim_input"])
    # params["lattice_num_keypoints"] = trial.suggest_categorical("lattice_keypoints", params["hpo_lattice_keypoints"])
    # params["lattice_calibration_num_keypoints"] = trial.suggest_categorical("calibration_keypoints", params["hpo_calibration_keypoints"])
    # params["lattice_calibration_num_keypoints_quantile"] = trial.suggest_categorical("calibration_keypoints_quantile", params["hpo_calibration_keypoints_quantile"])    

    #smnn stuff

    hpo_smnn_exp_1 = trial.suggest_categorical("smnn_exp_1", params["hpo_smnn_exp_1"])
    hpo_smnn_exp_2 = trial.suggest_categorical("smnn_exp_2", params["hpo_smnn_exp_2"])
    hpo_smnn_relu_1 = trial.suggest_categorical("smnn_relu_1", params["hpo_smnn_relu_1"])
    hpo_smnn_relu_2 = trial.suggest_categorical("smnn_relu_2", params["hpo_smnn_relu_2"])
    hpo_smnn_conf_1 = trial.suggest_categorical("smnn_conf_1", params["hpo_smnn_conf_1"])
    hpo_smnn_conf_2 = trial.suggest_categorical("smnn_conf_2", params["hpo_smnn_conf_2"])

    params["smnn_exp"] = (hpo_smnn_exp_1, hpo_smnn_exp_2)
    params["smnn_relu"] = (hpo_smnn_relu_1, hpo_smnn_relu_2)
    params["smnn_conf"] = (hpo_smnn_conf_1, hpo_smnn_conf_2)



    # disable plotting
    params['valid_plots_every'] = 900000
    params['array_metrics'] = {},

    params["metrics"] = {"ACE": [], 
            "CS_L": [], 
            "CRPS": []}
    
    ###############################


    device = torch.device(f'cuda:{torch.cuda.current_device()}') if torch.cuda.is_available() else 'cpu'
    print(f"STUDY: Using device: {device}")
    # pytorch random seed
    torch.manual_seed(params['random_seed'][0])
    np.random.seed(params['random_seed'][0])
    if _LOG_NEPTUNE:
        run['parameters'] = stringify_unsupported(params) # neptune only supports float and string

    if _DATA_DESCRIPTION ==  "Station 11 Irradiance Sunpoint":
        train,train_target,valid,valid_target,_,test_target= data_import()
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
        train,train_target,valid,valid_target,cs_train, cs_valid, overall_time, train_index, valid_index= import_ife_data(params,base_path=BASE_PATH) # has 22 features now, 11 without preprocessing
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

    Xv = return_Dataframe(valid)
    data = CalibratedDataset(X, y,cs_train, train_index, device=params['dataloader_device'],params=params) 
    dataloader = torch.utils.data.DataLoader(data, batch_size=params['batch_size'], shuffle=params['train_shuffle'], generator=torch.Generator(device=device))

    data_valid = CalibratedDataset(Xv, valid_target,cs_valid,valid_index, device=params['dataloader_device'],params=params)
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
    metric_plots = MetricPlots(params,Normalizer,sample_size=params["valid_plots_sample_size"],log_neptune=_LOG_NEPTUNE,trial_num=trial.number)
    optimizer = build_optimizer(params, model)

    final_loss = train_model(params = params,
                    model = model,
                    optimizer = optimizer,
                    criterion = criterion,
                    metric = metric,
                    metric_plots = metric_plots,
                    dataloader = dataloader,
                    dataloader_valid = data_loader_valid,
                    data = data,
                    data_valid = data_valid,
                    trial = trial,
                    log_neptune=_LOG_NEPTUNE,
                    verbose=_VERBOSE, 
                    neptune_run=  run,
                    overall_time = overall_time,
                    persistence = persistence
                    )

    return final_loss


def return_loss(metrics, compount=True):
    for metric, value in metrics.items():
            metrics[metric] = np.mean(value)
    if compount:
        metrics["optuna_loss"] = metrics["ACE"]#metrics["CRPS"] + metrics["ACE"] * metrics["CRPS"]
    else:
        metrics["optuna_loss"] = metrics["CRPS"]
    return metrics 

# Training loop



def train_model(params,
                model,
                optimizer,
                criterion,
                metric,
                metric_plots,
                dataloader,
                dataloader_valid,
                data,
                data_valid,
                trial,
                log_neptune=False,
                verbose=False, 
                neptune_run=None,
                overall_time = [],
                persistence = None
                ):

    run = neptune_run 
    run["status/trial"] = trial.number
    device = torch.device(f'cuda:{torch.cuda.current_device()}') if torch.cuda.is_available() else 'cpu'
    last_checkpoint = trial.user_attrs.get("last_checkpoint", False)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=params["scheduler_patience"], factor=params["scheduler_factor"], min_lr=params["scheduler_min_lr"])
    epoch_begin = 0
    if last_checkpoint:
        if os.path.exists(f"./tmp/tmp_model_{last_checkpoint}.pt"):
            checkpoint = torch.load(f"./tmp/tmp_model_{last_checkpoint}.pt")
            epoch = checkpoint["epoch"]
            epoch_begin = epoch + 1

            print(f"RESTORATOR: Loading a checkpoint from trial {last_checkpoint} in epoch {epoch}.")

            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            #accuracy = checkpoint["accuracy"]


    epochs = params['epochs']
    plot_ids = sorted(list(set([int(x * (512 / params["batch_size"])) for x in [7,24,14,22,37,8,3]])))
    #26,27,7,24,14,22,37,8,3,4,38 with 512
    #old 122, 128, 136, 131, 184, 124, 278 with 64 
    min_loss = 1000.0
    for epoch in range(epoch_begin,epochs):
        model.train()
        for batch in dataloader:
            training_data, target, cs, time_idx = batch
             # 2 quantiles because sharpness loss is logged
            training_data, target,cs = training_data.to(device), target.to(device), cs.to(device)
            # if params["inject_persistence"]:
            #     pers= persistence.forecast(time_idx[:,params["window_size"]]).unsqueeze(-1)
            # else:
            #     pers = None    

            quantile = data.return_quantile(training_data.shape[0],quantile_dim=2)
            quantile = quantile.to(device) # this line may be unnecessary, but it is not a problem to have it here.



            output = forward_pass(params,model,training_data,quantile,quantile_dim=quantile.shape[-1])
            # Compute loss
            loss = criterion(output, target, quantile) #TODO criterion should be able to handle quantile dim

            #Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # train_losses.append(loss.item())

            # with torch.no_grad():
            #     sharpness = sharpness_loss(output,quantile,scale_sharpness_scale=params['loss_calibration_scale_sharpness'])
            #     sharp_losses.append(sharpness.item())
            # if log_neptune:
            #     run['train/'+params['loss']].log(np.mean(train_losses))
            #     if params['loss'] == 'calibration_sharpness_loss':
            #         run['train/sharpness'].log(np.mean(sharp_losses))
    
        model.eval()
        
        if epoch % params['valid_metrics_every'] == 0:
            metric_dict = params['metrics']
            sample_counter = 1
            with torch.no_grad():
                for b_idx,batch in enumerate(dataloader_valid):
                    training_data, target,cs, time_idx = batch
                    training_data, target,cs = training_data.to(device), target.to(device), cs.to(device)
                    # There is not reason not to already use a quantile range here, as we are not training
                    pers= persistence.forecast(time_idx[:,0]).unsqueeze(-1)
                    pers_denorm = persistence.forecast_raw(time_idx[:,0]).unsqueeze(-1)
                    quantile,q_range = data_valid.return_quantile(training_data.shape[0],quantile_dim=params["metrics_quantile_dim"])
                    quantile, q_range = quantile.to(device), q_range.to(device)
                    output = forward_pass(params,model,training_data,quantile,quantile_dim=quantile.shape[-1])
                    # if params['valid_clamp_output']:
                    #     output = torch.clamp(output,min=0)
                    
                    metric_dict= metric(pred = output, truth = target, quantile = quantile, cs = cs, metric_dict=metric_dict,q_range=q_range,pers=pers_denorm)
                    if epoch == params['epochs'] - 1:
                        # metric_array_dict = metric_plots.accumulate_array_metrics(metric_array_dict,pred = output, truth = target, quantile = q_range) #q_range is just the quantiles in a range arrangement. 
                        if b_idx in plot_ids and sample_counter != params["valid_plots_sample_size"]:
                            metric_plots.generate_result_plots(training_data,output, target, quantile, cs, overall_time[time_idx.detach().cpu().numpy()],sample_num = sample_counter, neptune_run=neptune_run)
                            sample_counter += 1


        report_metrics = return_loss(metric_dict)
        if params["scheduler_enable"]:
            scheduler.step(report_metrics["CRPS"])
        trial.report(report_metrics["optuna_loss"],epoch)
        # run["status/epoch_time"] = epoch_time
        # run["status/epoch"] = epoch

        # min_loss = report_metrics["optuna_loss"] if report_metrics["optuna_loss"] < min_loss else min_loss
        min_loss = report_metrics["optuna_loss"]
        run[f"trials/{trial.number}/best_value"] = report_metrics["optuna_loss"]

        # print(f"Saving a checkpoint in epoch {epoch}.")
        model_save_path = f"./tmp/tmp_model_{trial.number}.pt"
        temp_path = os.path.join(BASE_PATH, model_save_path)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            temp_path,
        )
        trial.set_user_attr("last_checkpoint", trial.number)
        # # Handle pruning based on the intermediate value.
        # if trial.should_prune():
        #     if os.path.exists(temp_path):
        #         os.remove(temp_path)
        #     #return min_loss
        #     raise optuna.exceptions.TrialPruned()
    
    if os.path.exists(temp_path):
        os.remove(temp_path)
     #del run["trials"] # remove trials from server as that is unnecessary meta data.
    return min_loss #report_metrics["optuna_loss"]



def forward_pass(params:dict,
                 model: torch.nn.Module,
                 batch:torch.Tensor, 
                 quantile: torch.Tensor,
                 quantile_dim:int,
                 persistence = None,
                 device='cuda',):
    """
    Handels forward pass through model and does X amount of passes for different quantiles."""
    
    assert quantile_dim == quantile.shape[-1], 'Quantile dimension must match quantile tensor'
    if params['dataloader_device'] == 'cpu':
        batch = batch.to(device)
        quantile = quantile.to(device)
    output = torch.zeros((batch.size()[0],params['horizon_size'],quantile_dim))
    model.train()
    embedded = model[0](batch)
    
    for i in range(quantile_dim):
        aggregated_input = torch.cat([embedded,quantile[...,i]],dim=-1)
        output[...,i] = model[1](aggregated_input)
        if persistence is not None:
            output[...,i] = model[2](x = output[...,i],c = persistence.squeeze(),tau = quantile[0,0,i])
    return output


    

if __name__ == "__main__":
    
    
    import neptune
    from neptune.utils import stringify_unsupported
    from api_key import _NEPTUNE_API_TOKEN
    run = neptune.init_run(
        project="n1kl4s/QuantileError",
        custom_run_id=RUN_NAME,
        name = "HPO_FOX",
        api_token=_NEPTUNE_API_TOKEN,
        tags= params["neptune_tags"]
    )
    run['data/type'] = _DATA_DESCRIPTION

    
    # pytorch random seed
    torch.manual_seed(params['random_seed'][0])

    neptune_callback = optuna_utils.NeptuneCallback(run,plots_update_freq=1,log_all_trials=False)

    # Add stream handler of stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = RUN_NAME  # Unique identifier of the study.
    storage_path = os.path.join(BASE_PATH,study_name)
    storage_name = "sqlite:///{}.db".format(storage_path)


    # search_space = {
    #     "hidden_size": params["hpo_hidden_size"],
    #     "num_layers": params["hpo_num_layers"],
    #     "learning_rate": params["hpo_lr"]
    # }
    # search_space = {
    #     "lattice_dim_input": params["hpo_lattice_dim_input"],
    #     "lattice_keypoints": params["hpo_lattice_keypoints"],
    #     "calibration_keypoints": params["hpo_calibration_keypoints"],
    #     "calibration_keypoints_quantile": params["hpo_calibration_keypoints_quantile"]
    # }


    search_space = {
        "smnn_exp_1": params["hpo_smnn_exp_1"],
        "smnn_exp_2": params["hpo_smnn_exp_2"],
        "smnn_relu_1": params["hpo_smnn_relu_1"],
        "smnn_relu_2": params["hpo_smnn_relu_2"],
        "smnn_conf_1": params["hpo_smnn_conf_1"],
        "smnn_conf_2": params["hpo_smnn_conf_2"]
    }
    sampler = optuna.samplers.GridSampler(search_space)
    
    #optuna.samplers.RandomSampler() # grid wants all values again in searchspace which is weird.
    # pruner = optuna.pruners.SuccessiveHalvingPruner(reduction_factor=3,min_early_stopping_rate=5)
    # pruner = optuna.pruners.HyperbandPruner(min_resource=1,max_resource=params["epochs"],reduction_factor=5)

    study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True, direction="minimize", sampler=sampler)
    # if study.trials == []:
        # Encuing some trials to get a better starting point

    # if len(study.trials) != 0 and (study.trials[-1].state == 0  or study.trials[-1].state == 3):
    #     last_trial = study.trials[-1]
        
    #     if last_trial.user_attrs != {}:
    #         idx = last_trial.user_attrs["last_checkpoint"] 
    #         print("RESTORATOR: Detected past checkpoint.")
    #     else:
    #         idx = last_trial.number
    #     study.enqueue_trial(study.trials[-1].params,user_attrs={"last_checkpoint":idx})
    #     print("RESTORATOR: Continuing unfinished trial.")
    #     study.optimize(objective, n_trials=1,callbacks=[neptune_callback])
    study.optimize(objective, n_trials=1000,callbacks=[neptune_callback],show_progress_bar=False)

    run.stop()

