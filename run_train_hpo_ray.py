import torch
device = torch.device(f'cuda:{torch.cuda.current_device()}') if torch.cuda.is_available() else 'cpu'
torch.set_default_device(device)
#TODO GPU performance is worse than CPU unless batch size is increased. Maybe need better data loading.
### Debug tool################
from debug.plot import Debug_model
##############################
import os
from metrics.metrics import Metrics
from metrics.metric_plots import MetricPlots
from res.data import data_import, Data_Normalizer
from res.ife_data import import_ife_data 
from dataloader.calibratedDataset import CalibratedDataset
from pytorch_lattice.models.features import NumericalFeature

from losses.qr_loss import SQR_loss

from utils.helper_func import generate_surrogate_quantiles, return_features, return_Dataframe
from config import _LOG_NEPTUNE, _VERBOSE, params, _DATA_DESCRIPTION
from models.builder import build_model, build_optimizer
from training.train_loop import train_model

import numpy as np
import tempfile
# ray imports
from ray import tune
import ray
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.bayesopt import BayesOptSearch
# from ray.tune.search.bohb import TuneBOHB # only works with bohb scheduler
from hpo_logic import ray_config_setup, retrieve_config, build_config, return_loss

def train_main(params):
    import torch  # Ensure PyTorch is re-imported in each worker
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.set_device(0)  # Explicitly set GPU device
    # print(f"########################## Running on {device} #############################")  # Debugging output

    train,train_target,valid,valid_target,cs_train, cs_valid, overall_time, train_index, valid_index, params = retrieve_config(params)

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
    data = CalibratedDataset(X, y,cs_train, train_index, device="cpu",params=params) 
    dataloader = torch.utils.data.DataLoader(data, batch_size=params['batch_size'], shuffle=params['train_shuffle'], generator=torch.Generator(device="cpu"))

    data_valid = CalibratedDataset(Xv, valid_target,cs_valid,valid_index, device="cpu",params=params)
    data_loader_valid = torch.utils.data.DataLoader(data_valid, batch_size=params['batch_size'], shuffle=params['valid_shuffle'], generator=torch.Generator(device="cpu"))

    features_lattice = return_features(quantiles,params,data=None)


    model = build_model(params=params,device=device,features=features_lattice).to(device)

    # Forward pass
    # Define loss function and optimizer

    criterion = SQR_loss(type=params['loss'], lambda_=params['loss_calibration_lambda'], scale_sharpness=params['loss_calibration_scale_sharpness'])
    metric = Metrics(params,Normalizer,_DATA_DESCRIPTION)
    metric_plots = MetricPlots(params,Normalizer,sample_size=params["valid_plots_sample_size"],log_neptune=_LOG_NEPTUNE)
    optimizer = build_optimizer(params, model)


    if ray.train.get_checkpoint():
        loaded_checkpoint = ray.train.get_checkpoint()
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state, optimizer_state = torch.load(
                os.path.join(loaded_checkpoint_dir, "checkpoint.pt")
            )
            model.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)




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
                    log_neptune=False,
                    verbose=_VERBOSE, 
                    neptune_run=  None,
                    overall_time = overall_time)



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
                log_neptune=False,
                verbose=False, 
                neptune_run=None,
                overall_time = []):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.set_device(0)  # Explicitly set GPU device
    # print(f"########################## Running on {device} #############################")  #
    epochs = params['epochs']
    for epoch in range(epochs):
        start_time = torch.cuda.Event(enable_timing=True)
        start_time.record()
        end_time = torch.cuda.Event(enable_timing=True)
        train_losses = []
        sharp_losses = []
        model.train()
        for batch in dataloader:
            training_data, target, cs, _ = batch
             # 2 quantiles because sharpness loss is logged
            quantile = data.return_quantile(training_data.shape[0],quantile_dim=2)
            output = forward_pass(params,model,training_data,quantile,quantile_dim=quantile.shape[-1])
            # Compute loss
            loss = criterion(output, target, quantile) #TODO criterion should be able to handle quantile dim

            #Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

    
        model.eval()
        
        if epoch % params['valid_metrics_every'] == 0:
            metric_dict = params['metrics']
            
            with torch.no_grad():
                for b_idx,batch in enumerate(dataloader_valid):
                    training_data, target,cs, idx = batch

                    # There is not reason not to already use a quantile range here, as we are not training
                
                    quantile,q_range = data_valid.return_quantile(training_data.shape[0],quantile_dim=params["metrics_quantile_dim"])

                    output = forward_pass(params,model,training_data,quantile,quantile_dim=quantile.shape[-1])
                    if params['valid_clamp_output']:
                        output = torch.clamp(output,min=0)
                    
                    metric_dict= metric(pred = output, truth = target, quantile = quantile, cs = cs, metric_dict=metric_dict,q_range=q_range)

            
                
        end_time.record()
        torch.cuda.synchronize() 
        epoch_time = start_time.elapsed_time(end_time)/ 1000 # is in ms. Need to convert to seconds
        step_meta = {"Epoch": f"{epoch+1:02d}/{epochs}", "Time": epoch_time, "Train_Loss": np.mean(train_losses)}
        if params['loss']== 'calibration_sharpness_loss':
            step_meta["Sharpness"] = np.mean(sharp_losses)
        metric.summarize_metrics({**step_meta, **metric_dict},verbose = False, neptune = log_neptune,neptune_run=neptune_run)
        loss_dict = return_loss(metric_dict)
        # ray.train.report(metrics=loss_dict)
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
            torch.save(
                (model.state_dict(), optimizer.state_dict()), path
            )
            checkpoint = ray.train.Checkpoint.from_directory(temp_checkpoint_dir)
            ray.train.report(metrics=loss_dict)

    return model


def forward_pass(params:dict,
                 model: torch.nn.Module,
                 batch:torch.Tensor, 
                 quantile: torch.Tensor,
                 quantile_dim:int,
                 device = "cuda",):
    """
    Handels forward pass through model and does X amount of passes for different quantiles."""
    
    assert quantile_dim == quantile.shape[-1], 'Quantile dimension must match quantile tensor'

    output = torch.zeros((batch.size()[0],params['horizon_size'],quantile_dim))
    model.train()
    embedded = model[0](batch.to(device))
    
    for i in range(quantile_dim):
        aggregated_input = torch.cat([embedded,quantile[...,i].to(device)],dim=-1)
        output[...,i] = model[1](aggregated_input)
    return output
    



def trial_str_creator(trial):
    return "{}_{}".format(trial.trainable_name, trial.trial_id)


if __name__ == "__main__":
    
    do_new_search = False

    # pytorch random seed
    torch.manual_seed(params['random_seed'])

    if _DATA_DESCRIPTION ==  "Station 11 Irradiance Sunpoint":
        train,train_target,valid,valid_target,_,_ = data_import()
        if params['_INPUT_SIZE_LSTM'] == 1:
            train = train[:,11] # disable if training on all features/stations
        valid = valid[:,11]
    elif _DATA_DESCRIPTION == "IFE Skycam":
        train,train_target,valid,valid_target,cs_train, cs_valid, overall_time, train_index, valid_index= import_ife_data(params) # has 22 features now, 11 without preprocessing
        train,train_target,valid,valid_target, cs_train, cs_valid, overall_time= train.values,train_target.values,valid.values,valid_target.values, cs_train.values, cs_valid.values, overall_time.values
    else:
        raise ValueError("Data description not implemented")
    params = ray_config_setup(params)
    config_dict = build_config(train,train_target,valid,valid_target,cs_train, cs_valid, overall_time, train_index, valid_index, params)
    
    if do_new_search:
        tuner = tune.Tuner(
        tune.with_resources(
        tune.with_parameters(train_main),
        resources={"cpu":1,"gpu": 1}
        ),
        run_config=ray.train.RunConfig(verbose=1),
        param_space=config_dict,
        tune_config=tune.TuneConfig(
            metric='ray_tune_loss',
            mode='min',
            trial_dirname_creator = trial_str_creator,
            scheduler=ASHAScheduler(
                time_attr='training_iteration',
                max_t=400,
                grace_period=10,
                reduction_factor=3,
                brackets=1),  # Early stopping with ASHA
            #search_alg= BayesOptSearch(),  # Bayesian optimization
        ),
        )
    else:
        checkpoint_path = f"C:\\Users\\nikla\\ray_results\\train_main_2025-02-06_10-57-38"

        # Restore the tuner from the last checkpoint
        tuner = ray.tune.Tuner.restore(path=checkpoint_path,trainable=tune.with_resources(
        tune.with_parameters(train_main),
        resources={"cpu":1,"gpu": 1}
        ), param_space=config_dict,
        resume_unfinished = True )


    results = tuner.fit()

    # Get best hyperparameters
    best_config = results.get_best_result(metric="ray_tune_loss", mode="min").config
    print("Best Hyperparameters:", best_config)
    # Plot by epoch
    dfs = {result.path: result.metrics_dataframe for result in results}
    ax = None  # This plots everything on the same plot
    for d in dfs.values():
        ax = d.mean_accuracy.plot(ax=ax, legend=False)

