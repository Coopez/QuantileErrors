# Config for main.py

_DATA_DESCRIPTION = "IFE Skycam"#"Station 11 Irradiance Sunpoint" # Description of the data set
_LOG_NEPTUNE = False # determines if neptune is used
_VERBOSE = True # determines if printouts to console are made - should be False for ML cluster tasks


# Hyperparameters
params = dict(

batch_size = 64, # Batchsize
random_seed = 0, # Random seed
train_shuffle = True, # Determines if data is shuffled
valid_shuffle = True, # Determines if data is shuffled
target = 'GHI', # or 'GHI' or 'ERLING_SETTINGS'
learning_rate = 0.0001, #0.1, # Learning rate
epochs = 300, # Number of epochs
deterministic_optimization= False, # Determines if optimization is deterministic
window_size = 3,#30,#24, # Lookback size
horizon_size = 6,#90,#12, # Horizon size


# LSTM Hyperparameters
lstm_input_size = 22,  # Number of features 246 if all stations of sunpoint are used or 11,22 for IFE
lstm_hidden_size = [12,12], # LIST of number of nodes in hidden layers TODO will run into error if layers of different sizes. This is because hidden activation
lstm_num_layers = 2, # Number of layers

dnn_input_size = 22,  # input will be that * window_size
dnn_hidden_size = [12,24], # LIST of number of nodes in hidden layers
dnn_num_layers = 2, # Number of layers
dnn_activation = 'relu', # Activation function
# Lattice Hyperparameters
lattice_num_layers = 1, # Number of layers
lattice_num_per_layer = [1], # LIST
lattice_dim_input = [13], # List of input dims of lattices per layer
lattice_num_keypoints = 2, # Number of keypoints
lattice_calibration_num_keypoints = 5, # Number of keypoints in calibration layer


# Extra Loss Hyperparameters
loss_calibration_lambda = 0.0, # Lambda for beyond loss
loss_calibration_scale_sharpness = True, # Determines if sharpness is scaled by quantile

loss = 'pinball_loss', 
#options = 'calibration_sharpness_loss', 'pinball_loss',

optimizer = 'Adam', # dont try to rename. this is used to search for optimizer in builder.py
#optimizer_option = 'Adam', 'RAdam', 'NAdam', 'RMSprop', 'AdamW',


metrics =  {"ACE": [], 
            "MAE": [], 
            "RMSE": [],
            "CS_L": [],  # new abbreviation for Calibration Sharpness Loss or Beyond Loss
            "CRPS": []}, 
            #TODO SkillScore
array_metrics = {"PICP": None,
            "Cali_PICP": None,
            "PINAW": None, 
            }, # metrics which are in lists and thus not suitable for neptune logging. We will calculate them seperately and push them into the example plots



metrics_quantile_dim = 5, # can be 5, 9 for more accuracy, or 99 for full quantile range

input_model = "dnn",
#options = "lstm", "dnn"
output_model = "constrained_linear",
#options = "lattice", "linear", "constrained_linear", "linear_lattice", "lattice_linear"


valid_metrics_every = 1, # Determines how often metrics are calculated depending on Epoch number
valid_plots_every = 5, # Determines how often plots are calculated depending on Validation and epoch number
neptune_tags = [], # List of tags for neptune

save_all_epochs = False, # Determines if all epochs are saved
save_path_model_epoch = "models_save/" # Path for saving models



)

# Check for parameter consistency
assert len(params['lstm_hidden_size']) == params['lstm_num_layers'], "Number of hidden layers must match number of hidden sizes"
assert len(params['lattice_num_per_layer']) == params['lattice_num_layers'], "Number of lattice layers must match number of lattice sizes"
assert len(params['lattice_dim_input']) == params['lattice_num_layers'], "Number of lattice layers must match number of input dimensions"

