# Config for main.py

_DATA_DESCRIPTION = "Station 11 irradiance Sunpoint" # Description of the data set
_LOG_NEPTUNE = True # determines if neptune is used
_VERBOSE = True # determines if printouts to console are made - should be False for ML cluster tasks


# Hyperparameters

params = dict(
_BATCHSIZE = 256, # Batchsize
_RANDOM_SEED = 0, # Random seed
_SHUFFLE_train = False, # Determines if data is shuffled
_SHUFFLE_valid = False, # Determines if data is shuffled
_LEARNING_RATE = 0.001, #0.1, # Learning rate
_EPOCHS = 100, # Number of epochs
_DETERMINISTIC_OPTIMIZATION= False, # Determines if optimization is deterministic
# LSTM Hyperparameters
_INPUT_SIZE_LSTM = 1,  # Number of features 246 if all stations of sunpoint are used
_HIDDEN_SIZE_LSTM = 10, # Number of nodes in hidden layer
_NUM_LAYERS_LSTM = 2, # Number of layers
_WINDOW_SIZE = 8, # Lookback size
_PRED_LENGTH = 4, # Horizon size
# Lattice Hyperparameters
_NUM_LAYERS_LATTICE = 1, # Number of layers
_NUM_KEYPOINTS = 5, # Number of keypoints
_INPUT_DIM_LATTICE_FIRST_LAYER = 1, # Number of input dimensions in first layer - from this number of lattices in layer is derived
# Extra Loss Hyperparameters
_BEYOND_LAMBDA = 1.0, # Lambda for beyond loss
_SCALE_SHARPNESS = True, # Determines if sharpness is scaled by quantile

_LOSS = 1, #Index of loss_option
loss_option = ['calibration_sharpness_loss', 'pinball_loss'],

_REGULAR_OPTIMIZER = 0, #Index of optimizer_option
optimizer_option = ['Adam', 'RAdam', 'NAdam', 'RMSprop', 'AdamW'],


_Metrics =  {"PICP": None,"ACE": None,"PINAW": None, "MAE": None, "RMSE": None}, #"Calibration": None}#{"RMSE": None, "MAE": None, "skill_score": None}

_MODEL = 0, #Index of model_options
_model_options = ["LSTM_Lattice", "LSTM_Linear"], 

_METRICS_EVERY_X = 1 # Determines how often metrics are calculated depending on Epoch number
)