from ray import tune
import numpy as np
def ray_config_setup(params):
    #make hpo variable to search space.
    params['hpo_lr'] = tune.qloguniform(params['hpo_lr'][0], params['hpo_lr'][1], params['hpo_lr'][0])
    params['hpo_batch_size'] = tune.choice(params['hpo_batch_size'])
    params['hpo_hidden_size'] = tune.choice(params['hpo_hidden_size'])
    params['hpo_num_layers'] = tune.choice(params['hpo_num_layers'])

    # disable plotting
    params['valid_plots_every'] = 900000
    params['array_metrics'] = {},

    params["metrics"] = {"ACE": [], 
            "CS_L": [], 
            "CRPS": []}

    return params
        
def retrieve_config(params):
    
    modeltype = params['input_model']

    if modeltype == 'lstm':
        params['lstm_hidden_size'] = [params['hpo_hidden_size']] * params["hpo_num_layers"]
        params['lstm_num_layers'] = params['hpo_num_layers']
    if modeltype== 'dnn':
        params['dnn_num_layers'] = [params['hpo_hidden_size']] * params["hpo_num_layers"]
        params['dnn_hidden_size'] = params['hpo_num_layers']
    
    params['batch_size'] = params['hpo_batch_size']
    params['lr'] = params['hpo_lr']
    
    train = params['hpo_data_train']
    train_target = params['hpo_data_train_target']
    valid = params['hpo_data_valid']
    valid_target = params['hpo_data_valid_target']
    cs_train = params['hpo_cs_train']
    cs_valid = params['hpo_cs_valid']
    overall_time = params['hpo_overall_time']
    train_index = params['hpo_train_index']
    valid_index = params['hpo_valid_index']

    return train,train_target,valid,valid_target,cs_train, cs_valid, overall_time, train_index, valid_index, params


def build_config (train,train_target,valid,valid_target,cs_train, cs_valid, overall_time, train_index, valid_index, params):

    params['hpo_data_train'] = train
    params['hpo_data_train_target'] = train_target
    params['hpo_data_valid'] = valid
    params['hpo_data_valid_target'] = valid_target
    params['hpo_cs_train'] = cs_train
    params['hpo_cs_valid'] = cs_valid
    params['hpo_overall_time'] = overall_time
    params['hpo_train_index'] = train_index
    params['hpo_valid_index'] = valid_index

    return params



