# Config for main.py

_DATA_DESCRIPTION = "Station 11 irradiance Sunpoint" # Description of the data set
_LOG_NEPTUNE = True # determines if neptune is used
_VERBOSE = True # determines if printouts to console are made - should be False for ML cluster tasks


# Hyperparameters

params = dict(
_BATCHSIZE = 128, # Batchsize
_RANDOM_SEED = 0, # Random seed
_LEARNING_RATE = 0.001, # Learning rate
_EPOCHS = 10, # Number of epochs
_DETERMINISTIC_OPTIMIZATION= False, # Determines if optimization is deterministic
# LSTM Hyperparameters
_INPUT_SIZE_LSTM = 1,  # Number of features
_HIDDEN_SIZE_LSTM = 1, # Number of nodes in hidden layer
_NUM_LAYERS_LSTM = 1, # Number of layers
_WINDOW_SIZE = 1, # Lookback size
_PRED_LENGTH = 1, # Horizon size
# Lattice Hyperparameters
_NUM_LAYERS_LATTICE = 1, # Number of layers
_NUM_KEYPOINTS = 5, # Number of keypoints
_INPUT_DIM_LATTICE_FIRST_LAYER = 1, # Number of input dimensions in first layer - from this number of lattices in layer is derived
# Extra Loss Hyperparameters
_BEYOND_LAMBDA = 0.3, # Lambda for beyond loss


_LOSS = 0, #Index of loss_option
loss_option = ['calibration_sharpness_loss', 'pinball_loss'],

_REGULAR_OPTIMIZER = 0, #Index of optimizer_option
optimizer_option = ['Adam', 'RAdam', 'NAdam', 'RMSprop', 'AdamW'],


_Metrics =  {"PICP": None,"ACE": None, "MAE": None, "RMSE": None} #"Calibration": None}#{"RMSE": None, "MAE": None, "skill_score": None}



)