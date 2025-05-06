import torch
from torch.nn import Sequential, Linear
from .LSTM import LSTM
from .DNN import DNN
from .constrained_linear import Constrained_Linear
from .Calibrated_lattice_model import CalibratedLatticeModel
from .Parallel_lattice_model import ParallelLatticeModel
from .DNN_out_model import Neural_Net_with_Quantile
import torch.nn as nn

def build_model(params, device, features=None) -> Sequential:
    
    if params["input_model"] == "lstm":
        input_model = LSTM(input_size= params["lstm_input_size"],
                           hidden_size= params["lstm_hidden_size"],
                           num_layers= params["lstm_num_layers"],
                           window_size= params["window_size"],
                           output_size= 1,
                           dtype = torch.float64
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
    elif params["output_model"] == "dnn":
        output_model = Neural_Net_with_Quantile(input_size= data_output_size+1,
                                       output_size= params["horizon_size"])
    elif params["output_model"] == "linear_lattice" or params["output_model"] == "lattice_linear" or params["output_model"] == "lattice":
        assert features is not None, "Features must be provided for lattice model"
        output_model = CalibratedLatticeModel( features= features,
                                        output_min= 0,
                                        output_max= 1,
                                        num_layers= params["lattice_num_layers"],
                                        input_dim_per_lattice= params["lattice_dim_input"],
                                        output_size= params["horizon_size"],
                                        lattice_keypoints= params["lattice_num_keypoints"],
                                        model_type= params["output_model"],
                                        input_dim= data_output_size+1,
                                        downsampled_input_dim= params["lattice_donwsampled_dim"],
                                        device= device
        )
        # output_model = ParallelLatticeModel(features=features,
        #                                 output_min=0,
        #                                 output_max=1,
        #                                 num_layers= params["lattice_num_layers"],
        #                                 input_dim_per_lattice=params["lattice_dim_input"],
        #                                 output_size=params["horizon_size"],
        #                                 lattice_keypoints= params["lattice_num_keypoints"],
        #                                 model_type=params["output_model"],
        #                                 input_dim=data_output_size,
        #                                 downsampled_input_dim=params["lattice_donwsampled_dim"],
        #                                 device=device
        #                                 )
    else:
        raise ValueError("Output_Model not implemented")
 
    if params["inject_persistence"]:
        output_injection = Output_injector(params["horizon_size"]+data_output_size)
        model = torch.nn.ModuleList( 
            [input_model,
            output_model,
            output_injection
            ]
        ) # was sequential, but I want to be able to run output model x times for different quantiles
    else:
        model = torch.nn.ModuleList( 
            [input_model,
            output_model
            ]
        )
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

class Output_injector(nn.Module):
    def __init__(self,  input_size,start_value=1.0):
        super(Output_injector, self).__init__()

        # self.layer = nn.Linear(input_size+1,1)
        # self.layer.bias.data.fill_(0.0)
        # self.activation = nn.Tanh()
        self.parameter = nn.parameter.Parameter(torch.tensor(start_value))
    def forward(self,x_input, x,c,tau=1.0):
        # gate_input = torch.cat([x.clone(),x_input],dim=-1)
        # gate = self.activation(self.layer(gate_input)) + 1.0
        gate = self.parameter
        return x + (gate *c)