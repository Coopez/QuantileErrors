# Config for main.py

_DATA_DESCRIPTION = "Station 11 irradiance Sunpoint"
_LOG_NEPTUNE = False
# Hyperparameters

params = dict(
_BATCHSIZE = 128,
_RANDOM_SEED = 0,
_LEARNING_RATE = 0.001,
# LSTM Hyperparameters
_INPUT_SIZE_LSTM = 1,
_HIDDEN_SIZE_LSTM = 2,
_NUM_LAYERS_LSTM = 1,
_WINDOW_SIZE = 1,
_PRED_LENGTH = 1, # Horizon size
# Lattice Hyperparameters
_NUM_LAYERS_LATTICE = 1,
_NUM_KEYPOINTS = 5,
_INPUT_DIM_LATTICE_FIRST_LAYER = 1,
_EPOCHS = 10,
_DETERMINISTIC_OPTIMIZATION= False,
_BEYOND_LAMBDA = 0.3,


_LOSS = 1, #Index of loss_option
loss_option = ['calibration_sharpness_loss', 'pinball_loss'],

_REGULAR_OPTIMIZER = 0, #Index of optimizer_option
optimizer_option = ['Adam', 'RAdam', 'NAdam', 'RMSprop', 'AdamW'],


_Metrics =  {"PICP": None,"ACE": None} #"Calibration": None}#{"RMSE": None, "MAE": None, "skill_score": None}



)