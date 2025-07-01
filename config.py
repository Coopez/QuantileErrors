# Config for main.py

_DATA_DESCRIPTION = "Station 11 Irradiance Sunpoint" #"IFE Skycam"# Description of the data set 
_LOG_NEPTUNE = False # determines if neptune is used
_VERBOSE = True # determines if printouts to console are made - should be False for ML cluster tasks


# Hyperparameters
params = dict(

debug = False, # Determines some debug outputs
batch_size = 64 , # Batchsize 
random_seed = [0,1,2,3,4], # Random seeds
model_save = False,

train_shuffle = True, # Determines if data is shuffled
valid_shuffle = False, # Determines if data is shuffled
dataloader_device = 'cpu', # determines if data is loaded on cpu or gpu
target = 'GHI', # or 'GHI' or 'ERLING_SETTINGS'
target_summary = 1, # mean window, will not change anything if set to 1
learning_rate = 0.001, #0.1, # Learning rate
epochs = 150, # Number of epochs
deterministic_optimization= False, # Determines if optimization is deterministic
window_size = 96, # Lookback size
horizon_size = 36,#90,#12, # Horizon size

scheduler_min_lr = 0.00001, # Minimum learning rate for scheduler
scheduler_patience = 2, # Patience for scheduler
scheduler_factor = 0.1, # Factor for scheduler
scheduler_enable = True, # Determines if scheduler is enabled

step_scheduler_enable = False,
step_scheduler_milestones = [1], # Milestones for step scheduler
step_scheduler_gamma = 0.1, # Gamma for step scheduler

constant_quantile = False,

inject_persistence = False, # Determines if persistence model is injected
# LSTM Hyperparameters
lstm_input_size = 246,#246, #22,  # Number of features 246 if all stations of sunpoint are used or 11,22 for IFE
lstm_hidden_size = [128,128], # LIST of number of nodes in hidden layers will run into error if layers of different sizes. This is because hidden activation
lstm_num_layers = 2, # Number of layers

dnn_input_size = 246,  # input will be that * window_size
dnn_hidden_size = [32,32], # LIST of number of nodes in hidden layers
dnn_num_layers = 2, # Number of layers
dnn_activation = 'relu', # Activation function
# Lattice Hyperparameters
lattice_num_layers = 1, # Number of layers
# lattice_num_per_layer = [28], # LIST
lattice_dim_input = 2, # input dim per lattice
lattice_num_keypoints = 21, # Number of keypoints
lattice_calibration_num_keypoints = 61, # Number of keypoints in calibration layer
lattice_calibration_num_keypoints_quantile = 11, # Number of layers in calibration layer
lattice_donwsampled_dim = 13, # Dimension of downsampled input when using linear_lattice
lattice_quantile = "all", # "all" or "single"

# Extra Loss Hyperparameters
loss_calibration_lambda = 0.0, # Lambda for beyond loss
loss_calibration_scale_sharpness = True, # Determines if sharpness is scaled by quantile

loss = 'pinball_loss', 
#options = 'calibration_sharpness_loss', 'huber_pinball_loss',

optimizer = 'Adam', # dont try to rename. this is used to search for optimizer in builder.py
#optimizer_option = 'Adam', 'RAdam', 'NAdam', 'RMSprop', 'AdamW',


metrics =  {"ACE": [], 
            "MAE": [], 
            "RMSE": [],
            #"CS_L": [],  # new abbreviation for Calibration Sharpness Loss or Beyond Loss
            "CRPS": [],
            "SS": []},
            # "SS_filt":[]}, 
            #TODO SkillScore
array_metrics = {"PICP": [],
            "Cali_PICP": [],
            "PINAW": [],
            "Correlation": [],
             "SkillScore": []}, # metrics which are in lists and thus not suitable for neptune logging. We will calculate them seperately and push them into the example plots



metrics_quantile_dim = 11, # can be 5, 9 for more accuracy, or 99 for full quantile range

input_model = "lstm",
#options = "lstm", "dnn"
output_model = "linear", # "linear_lattice", "lattice_linear", "lattice", "dnn"
#options = "lattice", "linear", "constrained_linear", "linear_lattice", "lattice_linear",  


valid_metrics_every = 1, # Determines how often metrics are calculated depending on Epoch number
valid_plots_every = 1, # Determines how often plots are calculated depending on Validation and epoch number
valid_plots_sample_size = 7, # Sample size for plots - will run into error if larger than the batch size adjusted index list.
valid_plots_save_path = "plots_save/", # Path for saving plots
valid_clamp_output = True, # Determines if output is clamped to 0

neptune_tags = ["IFE Skyimage","HPO"], # List of tags for neptune

save_all_epochs = False, # Determines if all epochs are saved
save_path_model_epoch = "models_save/", # Path for saving models


hpo_lr = [0.00001,0.0001,0.001], # Hyperparameter optimization search space for learning rate
#hpo_batch_size = [64,256,1024], # Hyperparameter optimization search space for batch size
#hpo_window_size = [60,90,120,180], # Hyperparameter optimization search space for window size
hpo_hidden_size = [16,32,64,128], # Hyperparameter optimization search space for hidden size
hpo_num_layers = [2,3,4], # Hyperparameter optimization search space for number of layers

hpo_lattice_dim_input = [1, 2], # Hyperparameter optimization search space for number of input dimensions
hpo_lattice_keypoints = [11,21,51], # Hyperparameter optimization search space for number of keypoints
hpo_calibration_keypoints = [51,61,71], # Hyperparameter optimization search space for number of calibration keypoints
hpo_calibration_keypoints_quantile = [21,31,51], # Hyperparameter optimization search space for number of calibration keypoints


comp_model = "pp",# "pp" , qr for quantile regression, pp for point prediction, sp for smart persistence

smnn_exp = (256,512),
smnn_relu = (256,512), 
smnn_conf = (512,256), 

hpo_smnn_exp_1 = [512,1024],
hpo_smnn_exp_2 = [256,1024], # Hyperparameter optimization search space for SMNN exp units
hpo_smnn_relu_1 = [256,1024],
hpo_smnn_relu_2 = [512], # Hyperparameter optimization search space for SMNN relu units
hpo_smnn_conf_1 = [128,256,512],
hpo_smnn_conf_2 = [256,512], # Hyperparameter optimization search space for SMNN confluence units

early_stopping_patience = 150,
early_stopping_tolerance = 0.01, # Tolerance for early stopping


test_model_path = "models_save/model.pth", # Path for loading model
test_model_name = "lattice_linear", # Name of the model to be tested
data_save_path = "data_save", # Path for saving data
)

# Check for parameter consistency
assert len(params['lstm_hidden_size']) == params['lstm_num_layers'], "Number of hidden layers must match number of hidden sizes"
#assert len(params['lattice_num_per_layer']) == params['lattice_num_layers'], "Number of lattice layers must match number of lattice sizes"
#assert len(params['lattice_dim_input']) == params['lattice_num_layers'], "Number of lattice layers must match number of input dimensions"

