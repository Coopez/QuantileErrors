# Training loop

import torch
from losses.qr_loss import sharpness_loss
import numpy as np
from neptune.utils import stringify_unsupported
from debug.model import print_model_parameters
from debug.plot import Debug_model

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
                overall_time = []
                ):
    if params['debug']:
        print_model_parameters(model)
    if log_neptune:
        run = neptune_run 
        run_name = run["sys/id"].fetch()
    else:
        run_name = 'local_test_run'

    epochs = params['epochs']
    if params['deterministic_optimization']:
        epochs = 1
    plot_ids = sorted(list(set([int(x * (64 / params["batch_size"])) for x in [122, 128, 136, 131, 184, 124, 278]])))
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
            #### Normal Case
            if params['output_model'] != 'lattice' or params['output_model'] != 'linear_lattice':
                
                output = forward_pass(params,model,training_data,quantile,quantile_dim=quantile.shape[-1])
                # Compute loss
                loss = criterion(output, target, quantile) #TODO criterion should be able to handle quantile dim

                #Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            #### Lattice Case
            else:
                # Freeze input model initially to train lattice columns individually
                freeze_Model(model[0])
                lattice_loss = []
                output = torch.zeros((training_data.size()[0],params['horizon_size'],quantile.shape[-1]))
                for i in range(params["horizon_size"]):
                    optimizer.zero_grad()
                    # the output is only one of the 90 horizon outputs.
                    single_output = lattice_forward_pass(params,model,training_data,quantile,quantile_dim=quantile.shape[-1],horizon=i)
                    output[:,i,:] = single_output
                    loss = criterion(single_output, target[:,i,:], quantile)
                    loss.backward()
                    optimizer.step()
                    lattice_loss.append(loss.item())
                optimizer.zero_grad()
                unfreeze_Model(model[0])
                freeze_Model(model[1])
                mean_loss = lattice_loss.mean()
                mean_loss.backward()
                optimizer.step()
                unfreeze_Model(model[1])
                train_losses.append(mean_loss.item())

            with torch.no_grad():
                sharpness = sharpness_loss(output,quantile,scale_sharpness_scale=params['loss_calibration_scale_sharpness'])
                sharp_losses.append(sharpness.item())
            if log_neptune:
                run['train/'+params['loss']].log(np.mean(train_losses))
                if params['loss'] == 'calibration_sharpness_loss':
                    run['train/sharpness'].log(np.mean(sharp_losses))
    
        epoch_path = params['save_path_model_epoch']
        save_model_per_epoch(run_name, model, epoch_path, epoch, save_all=params['save_all_epochs'])

        model.eval()
        
        if epoch % params['valid_metrics_every'] == 0:
            metric_dict = params['metrics']
            metric_array_dict = params['array_metrics']
            sample_counter = 1
            
            with torch.no_grad():
                for b_idx,batch in enumerate(dataloader_valid):
                    training_data, target,cs, idx = batch

                    # There is not reason not to already use a quantile range here, as we are not training
                
                    quantile,q_range = data_valid.return_quantile(training_data.shape[0],quantile_dim=params["metrics_quantile_dim"])

                    output = forward_pass(params,model,training_data,quantile,quantile_dim=quantile.shape[-1])
                    if params['valid_clamp_output']:
                        output = torch.clamp(output,min=0)
                    
                    metric_dict= metric(pred = output, truth = target, quantile = quantile, cs = cs, metric_dict=metric_dict,q_range=q_range)

                    if epoch % params['valid_plots_every'] == 0:
                        metric_array_dict = metric_plots.accumulate_array_metrics(metric_array_dict,pred = output, truth = target, quantile = q_range) #q_range is just the quantiles in a range arrangement. 
                        if b_idx in plot_ids and sample_counter != params["valid_plots_sample_size"]:
                            metric_plots.generate_result_plots(training_data,output, target, quantile, cs, overall_time[idx.detach().cpu().numpy()],sample_num = sample_counter, neptune_run=neptune_run)
                            sample_counter += 1

            if epoch % params['valid_plots_every'] == 0:
                metric_plots.generate_metric_plots(metric_array_dict,neptune_run=neptune_run, dataloader_length=len(dataloader_valid))

                

                    
                    
                
                # metric_plots.generate_result_plots(training_data,output, target, quantile, cs, overall_time[idx.detach().cpu().numpy()], neptune_run=neptune_run)
            # if log_neptune:
            #     #log all metrics in metric_dict to neptune
            #     for key, value in metric_dict.items():
            #         # test if value is a list 
            #         if isinstance(value[0], np.ndarray): # need to check if in the list of batch accumulated values we have an array
            #                 value = np.stack(value,axis=-1).mean(axis=-1).tolist()
            #                 run['valid/'+key].log(stringify_unsupported(value))
            #         else:
            #             run['valid/'+key].log(np.mean(value))
        
        end_time.record()
        torch.cuda.synchronize() 
        epoch_time = start_time.elapsed_time(end_time)/ 1000 # is in ms. Need to convert to seconds
        step_meta = {"Epoch": f"{epoch+1:02d}/{epochs}", "Time": epoch_time, "Train_Loss": np.mean(train_losses)}
        if params['loss']== 'calibration_sharpness_loss':
            step_meta["Sharpness"] = np.mean(sharp_losses)
        metric.summarize_metrics({**step_meta, **metric_dict},neptune = log_neptune,neptune_run=neptune_run)

    return model


def forward_pass(params:dict,
                 model: torch.nn.Module,
                 batch:torch.Tensor, 
                 quantile: torch.Tensor,
                 quantile_dim:int,
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
    return output


def save_model_per_epoch(run_name, model: torch.nn.Module, path:str, epoch:int, save_all:bool=False):

    if save_all:
        torch.save(model.state_dict(), path+f"{run_name}_epoch_{epoch}.pt")
    else:
        torch.save(model.state_dict(), path+f"{run_name}.pt")
    

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

    output = torch.zeros((batch.size()[0],params['horizon_size'],quantile_dim))
    model.train()
    embedded = model[0](batch)
    for i in range(quantile_dim):
        aggregated_input = torch.cat([embedded,quantile[...,i]],dim=-1)
        output[...,i] = model[1](aggregated_input,horizon)
    return output