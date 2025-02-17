# Config for main.py

_DATA_DESCRIPTION = "IFE Skycam"#"Station 11 Irradiance Sunpoint" # Description of the data set
_LOG_NEPTUNE = True # determines if neptune is used
_VERBOSE = True # determines if printouts to console are made - should be False for ML cluster tasks


# Hyperparameters
params = dict(

debug = False, # Determines some debug outputs

batch_size = 512, # Batchsize 
random_seed = 0, # Random seed
train_shuffle = True, # Determines if data is shuffled
valid_shuffle = False, # Determines if data is shuffled
dataloader_device = 'cpu', # determines if data is loaded on cpu or gpu
target = 'GHI', # or 'GHI' or 'ERLING_SETTINGS'
learning_rate = 0.0001, #0.1, # Learning rate
epochs = 350, # Number of epochs
deterministic_optimization= False, # Determines if optimization is deterministic
window_size = 60,#30,#24, # Lookback size
horizon_size = 90,#90,#12, # Horizon size


# LSTM Hyperparameters
lstm_input_size = 22,  # Number of features 246 if all stations of sunpoint are used or 11,22 for IFE
lstm_hidden_size = [28,28], # LIST of number of nodes in hidden layers TODO will run into error if layers of different sizes. This is because hidden activation
lstm_num_layers = 2, # Number of layers

dnn_input_size = 22,  # input will be that * window_size
dnn_hidden_size = [64,64], # LIST of number of nodes in hidden layers
dnn_num_layers = 2, # Number of layers
dnn_activation = 'relu', # Activation function
# Lattice Hyperparameters
lattice_num_layers = 1, # Number of layers
lattice_num_per_layer = [1], # LIST
lattice_dim_input = [13], # List of input dims of lattices per layer
lattice_num_keypoints = 2, # Number of keypoints
lattice_calibration_num_keypoints = 5, # Number of keypoints in calibration layer
lattice_donwsampled_dim = 13, # Dimension of downsampled input when using linear_lattice


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
array_metrics = {"PICP": [],
            "Cali_PICP": [],
            "PINAW": [], 
            }, # metrics which are in lists and thus not suitable for neptune logging. We will calculate them seperately and push them into the example plots



metrics_quantile_dim = 9, # can be 5, 9 for more accuracy, or 99 for full quantile range

input_model = "lstm",
#options = "lstm", "dnn"
output_model = "linear",
#options = "lattice", "linear", "constrained_linear", "linear_lattice", "lattice_linear"


valid_metrics_every = 1, # Determines how often metrics are calculated depending on Epoch number
valid_plots_every = 1, # Determines how often plots are calculated depending on Validation and epoch number
valid_plots_sample_size = 5, # Sample size for plots - will run into error if larger than the batch size adjusted index list.
valid_plots_save_path = "plots_save/", # Path for saving plots
valid_clamp_output = True, # Determines if output is clamped to 0

neptune_tags = ["IFE Skyimage","HPO"], # List of tags for neptune

save_all_epochs = False, # Determines if all epochs are saved
save_path_model_epoch = "models_save/", # Path for saving models


hpo_lr = [0.000001,0.00001,0.0001, 0.001, 0.01], # Hyperparameter optimization search space for learning rate
#hpo_batch_size = [64,256,1024], # Hyperparameter optimization search space for batch size
hpo_window_size = [60,90,120], # Hyperparameter optimization search space for window size
hpo_hidden_size = [16,32,64,128,256,516], # Hyperparameter optimization search space for hidden size
hpo_num_layers = [1, 2, 3], # Hyperparameter optimization search space for number of layers

fox_valid_plots_save_path = " ", # Path for saving plots on fox
fox_save_path_model_epoch = " ", # Path for saving models on fox

fox_code_dir = " ", # Path for code directory on fox
fox_data_dir = " ", # Path for data directory on fox
fox_neptune_dir = " ", # Path for neptune directory on fox
)

# Check for parameter consistency
assert len(params['lstm_hidden_size']) == params['lstm_num_layers'], "Number of hidden layers must match number of hidden sizes"
assert len(params['lattice_num_per_layer']) == params['lattice_num_layers'], "Number of lattice layers must match number of lattice sizes"
assert len(params['lattice_dim_input']) == params['lattice_num_layers'], "Number of lattice layers must match number of input dimensions"

