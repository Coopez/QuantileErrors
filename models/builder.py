import torch
from torch.nn import Sequential, Linear
from .LSTM import LSTM
from .DNN import DNN
from .constrained_linear import Constrained_Linear
from .Calibrated_lattice_model import CalibratedLatticeModel


def build_model(params, device, features=None) -> Sequential:
    
    if params["input_model"] == "lstm":
        input_model = LSTM(input_size= params["lstm_input_size"],
                           hidden_size= params["lstm_hidden_size"],
                           num_layers= params["lstm_num_layers"],
                           window_size= params["window_size"],
                           output_size= 1
                           )
        data_output_size = params["lstm_hidden_size"][-1]

    elif params["input_model"] == "dnn":
        input_model = DNN(input_size= params["dnn_input_size"] * params["window_size"],
                           hidden_size= params["dnn_hidden_size"],
                           num_layers= params["dnn_num_layers"],
                           activation= params["dnn_activation"]
                           )
        data_output_size = params["dnn_hidden_size"][-1]
    else:
        raise ValueError("Input_Model not implemented")
    
    #options = "lattice", "linear", "constrained_linear", "linear_lattice", "lattice_linear"
    if params["output_model"] == "linear":
        output_model = Linear(in_features= data_output_size+1,
                                       out_features= params["horizon_size"])
    elif params["output_model"] == "constrained_linear":
        output_model = Constrained_Linear(input_dim= data_output_size+1,
                                       output_dim= params["horizon_size"],
                                       quantile_idx= -1) # assuming quantile is the last feature
    elif params["output_model"] == "linear_lattice" or params["output_model"] == "lattice_linear" or params["output_model"] == "lattice":
        assert features is not None, "Features must be provided for lattice model"
        output_model = CalibratedLatticeModel( features= features,
                                        output_min= 0,
                                        output_max= 1,
                                        num_layers= params["lattice_num_layers"],
                                        input_dim_per_lattice= params["lattice_dim_input"],
                                        num_lattice_per_layer= params["lattice_num_per_layer"],
                                        output_size= params["horizon_size"],
                                        calibration_keypoints= params["lattice_calibration_num_keypoints"],
                                        lattice_keypoints= params["lattice_num_keypoints"],
                                        model_type= params["output_model"],
                                        input_dim= data_output_size,
                                        downsampled_input_dim= params["lattice_donwsampled_dim"],
                                        device= device
        )

    else:
        raise ValueError("Output_Model not implemented")
 

    model = torch.nn.ModuleList( 
        [input_model,
        output_model]
    ) # was sequential, but I want to be able to run output model x times for different quantiles
    return model


def build_optimizer(params,model):
    if params['deterministic_optimization']:
        from pytorch_minimize.optim import MinimizeWrapper
        minimizer_args = dict(method='SLSQP', options={'disp':True, 'maxiter':100}) # supports a range of methods
        optimizer = MinimizeWrapper(model.parameters(), minimizer_args)
    else:
        optimizer_class = getattr(torch.optim, params['optimizer'])
        optimizer = optimizer_class(model.parameters(), lr=params['learning_rate'])
    return optimizer