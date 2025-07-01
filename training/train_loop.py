# Training loop

import torch
from losses.qr_loss import sharpness_loss
import numpy as np
from neptune.utils import stringify_unsupported
from debug.model import print_model_parameters
from debug.plot import Debug_model
from torch.optim.lr_scheduler import ReduceLROnPlateau,MultiStepLR

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
                overall_time = [],
                persistence = None,
                base_path = None
                ):
    if params['debug']:
        print_model_parameters(model)
    if log_neptune:
        run = neptune_run 
        run_name = run["sys/id"].fetch()
        run["valid/parameters"] = sum(p.numel() for p in model.parameters())
    else:
        run_name = 'local_test_run'

    epochs = params['epochs']
    if params['deterministic_optimization']:
        epochs = 1
    plateau_scheduler = ReduceLROnPlateau(optimizer, 'min', patience=params["scheduler_patience"], factor=params["scheduler_factor"], min_lr=params["scheduler_min_lr"])
    step_scheduler = MultiStepLR(optimizer, milestones=params["step_scheduler_milestones"], gamma=params["step_scheduler_gamma"])
    plot_ids = sorted(list(set([int(x * (512 / params["batch_size"])) for x in [7,24,14,22,37,8,3]])))
    #26,27,7,24,14,22,37,8,3,4,38 with 512
    #old 122, 128, 136, 131, 184, 124, 278 with 64 
    epoch_path = params['save_path_model_epoch']
    early_stopping = ModelCheckpointer(path=epoch_path, tolerance=params['early_stopping_tolerance'], patience=params['early_stopping_patience'])
    for epoch in range(epochs):
        start_time = torch.cuda.Event(enable_timing=True)
        start_time.record()
        end_time = torch.cuda.Event(enable_timing=True)
        train_losses = []
        sharp_losses = []
        model.train()
        for batch in dataloader:
            training_data, target, cs, time_idx = batch

            # pers= persistence.forecast(time_idx[:,params["window_size"]]).unsqueeze(-1)

            # 2 quantiles because sharpness loss is logged
            quantile = data.return_quantile(training_data.shape[0],quantile_dim=2,constant=params['constant_quantile'])
            #### Normal Case
                
            output = forward_pass(params,model,training_data,quantile,quantile_dim=quantile.shape[-1])
            # Compute loss
            loss = criterion(output, target, quantile) #TODO criterion should be able to handle quantile dim
            # debug = Debug_model(model,output,target)
            #Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())


            with torch.no_grad():
                sharpness = sharpness_loss(output,quantile,scale_sharpness_scale=params['loss_calibration_scale_sharpness'])
                sharp_losses.append(sharpness.item())
            if log_neptune:
                run['train/'+params['loss']].log(np.mean(train_losses))
                if params['loss'] == 'calibration_sharpness_loss':
                    run['train/sharpness'].log(np.mean(sharp_losses))
    
        
        save_model_per_epoch(run_name, model, epoch_path, epoch, save_all=params['save_all_epochs'])

        model.eval()
        
        if epoch % params['valid_metrics_every'] == 0:

            metric_dict = {label: [] for label in params['metrics'].keys()}
            metric_array_dict = {label: [] for label in params['array_metrics'].keys()}
            # metric_dict = params['metrics'].copy()
            # metric_array_dict = params['array_metrics'].copy()
            sample_counter = 1
            # pers_losses = []
            with torch.no_grad():
                # batch_var = []
                for b_idx,batch in enumerate(dataloader_valid):
                    training_data, target,cs, time_idx = batch
                    # batch_var.append([torch.var(target[0]).detach().cpu().numpy(),b_idx])
                    # There is not reason not to already use a quantile range here, as we are not training
                    # if params["inject_persistence"]:
                    pers= persistence.forecast(time_idx[:,0]).unsqueeze(-1)
                    pers_denorm = persistence.forecast_raw(time_idx[:,0]).unsqueeze(-1)
                    # else:
                    #     pers = None
                    #     pers_denorm = None
                    quantile,q_range = data_valid.return_quantile(training_data.shape[0],quantile_dim=params["metrics_quantile_dim"])

                    output = forward_pass(params,model,training_data,quantile,quantile_dim=quantile.shape[-1])


                    
                    # pers_loss = torch.mean(torch.abs(target - pers)).detach().cpu().numpy().item()
                    # pers_losses.append(pers_loss)
                    metric_dict= metric(pred = output, truth = target, quantile = quantile, cs = cs, metric_dict=metric_dict,q_range=q_range,pers=pers_denorm)

                    if epoch % params['valid_plots_every'] == 0:
                        metric_array_dict = metric_plots.accumulate_array_metrics(metric_array_dict,pred = output, truth = target, quantile = q_range,pers = pers_denorm) #q_range is just the quantiles in a range arrangement. 
                        if b_idx in plot_ids and sample_counter != params["valid_plots_sample_size"]+1:
                            metric_plots.generate_result_plots(training_data,output, target, quantile, cs, overall_time[time_idx.detach().cpu().numpy()],sample_num = sample_counter, neptune_run=neptune_run)
                            sample_counter += 1
            # print(np.mean(pers_losses))
            if epoch % params['valid_plots_every'] == 0:
                metric_plots.generate_metric_plots(metric_array_dict,neptune_run=neptune_run, dataloader_length=len(dataloader_valid))

            
        
        end_time.record()
        torch.cuda.synchronize() 
        epoch_time = start_time.elapsed_time(end_time)/ 1000 # is in ms. Need to convert to seconds
        step_meta = {"Epoch": f"{epoch+1:02d}/{epochs}", "Time": epoch_time, "Train_Loss": np.mean(train_losses)}
        if params['loss']== 'calibration_sharpness_loss':
            step_meta["Sharpness"] = np.mean(sharp_losses)
        scheduler_metrics = metric.summarize_metrics({**step_meta, **metric_dict},neptune = log_neptune,neptune_run=neptune_run)
        if params["scheduler_enable"]:
            plateau_scheduler.step(scheduler_metrics["CRPS"])
        if params["step_scheduler_enable"]:
            step_scheduler.step()
        # print("Current Learning Rate: ", plateau_scheduler.get_last_lr())
        # step_scheduler.step(epoch=epoch)
        # break_condition, model = early_stopping(model, scheduler_metrics["CRPS"])
        # if break_condition is False:
        #     break
    return model


def forward_pass(params:dict,
                 model: torch.nn.Module,
                 batch:torch.Tensor, 
                 quantile: torch.Tensor,
                 quantile_dim:int,
                 persistence=None,
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
        # output[...,i] = model[1](aggregated_input)
        if persistence is not None:
            output_i = model[1](aggregated_input)
            persistence_output = model[2](x = output_i,c = persistence.squeeze(),tau = quantile[0,0,i], x_input =aggregated_input)
            output[...,i] = persistence_output
        else:
            output[...,i] = model[1](aggregated_input)
    return output


def save_model_per_epoch(run_name, model: torch.nn.Module, path:str, epoch:int, save_all:bool=False):

    if save_all:
        torch.save(model.state_dict(), path+f"{run_name}_epoch_{epoch}.pt")
    else:
        torch.save(model.state_dict(), path+f"{run_name}.pt")
    
class ModelCheckpointer():
    def __init__(self, path:str,tolerance:float=0.0001, patience:int=5):
        self.path = path
        self.last_metric = 9999.0
        self.counter = 0
        self.tolerance = tolerance
        self.patience = patience
    def __call__(self,  model: torch.nn.Module, metric):
        """
        Checks if the metric has improved and saves the model if it has.
        If the metric has not improved for a certain number of epochs, it stops training.
        """
        if metric < (self.last_metric - self.tolerance):
            self._save(model)
            self.last_metric = metric
            self.counter = 0
            return True, model
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Stopping training after {self.counter} epochs without improvement.")
                model = self._load(model)
                return False, model
        return True, model
    def _save(self, model: torch.nn.Module):
        torch.save(model.state_dict(), self.path+"checkpoint_model.pt")
    def _load(self,model: torch.nn.Module):
        model.load_state_dict(torch.load(self.path+"checkpoint_model.pt"))
        return model


def generate_validation_plots(x,y):
    pass #TODO


def freeze_Model(model: torch.nn.Module):
    for param in model.parameters():
        param.requires_grad = False
def unfreeze_Model(model: torch.nn.Module):
    for param in model.parameters():
        param.requires_grad = True

def lattice_forward_pass(params:dict,
                 model: torch.nn.Module,
                 batch:torch.Tensor, 
                 quantile: torch.Tensor,
                 quantile_dim:int,
                 horizon: int,
                 device='cuda'):
    """
    Handels forward pass through model and does X amount of passes for different quantiles."""
    
    assert quantile_dim == quantile.shape[-1], 'Quantile dimension must match quantile tensor'
    if params['dataloader_device'] == 'cpu':
        batch = batch.to(device)
        quantile = quantile.to(device)

    output = torch.zeros((batch.size()[0],quantile_dim))
    model.train()
    embedded = model[0](batch)
    for i in range(quantile_dim):
        aggregated_input = torch.cat([embedded,quantile[...,i]],dim=-1)
        output[...,i] = model[1](aggregated_input,horizon)
    return output