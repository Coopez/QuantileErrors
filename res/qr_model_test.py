import torch
import torch.nn as nn
from qr_model import SQR_LSTM, sqr_loss

# Examples time series

training_data = torch.rand(10, 10, 1).type(torch.float32)
test_data = torch.linspace(0, 1, 10,dtype=torch.float32).view(1, 10, 1)
test_data = test_data.repeat(10, 1, 1)

# QR Model

model = SQR_LSTM(input_size=1, window_size=1, hidden_size=10,layers=1,pred_length=10, loss_type='beyond')

# Loss
loss = sqr_loss


# Forward pass

output, quantile = model(training_data)
print(output.shape, quantile.shape)

# Loss

l = loss(output, test_data, quantile, type = 'beyond')
print(l)

# Backward pass

l.backward()

