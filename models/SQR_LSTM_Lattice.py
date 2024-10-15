import torch
import numpy as np
import torch.nn as nn  


class SQR_LSTM_Lattice(nn.Module):
    def __init__(self,input_size,hidden_size,layers,window_size,pred_length,persistence_connection=False,loss_type='pinball', output_size = 1):
        super(SQR_LSTM_Lattice, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.layers = layers
        self.window_size = window_size
        self.pred_length = pred_length

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(self.input_size, self.hidden_size,num_layers = self.layers,batch_first=True,dtype=torch.float64)
        #self.lt = nn.Linear(self.hidden_size,self.pred_length*self.output_size)

        self.cs = True
        self.cs_alpha = nn.Parameter(torch.tensor([2.0]), requires_grad= True if self.cs else False)
        

    def forward(self, x:torch.tensor,cs = None):
        """
        Forward pass for the QR model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_size).
            cs (torch.Tensor, optional): Clear sky values tensor of shape (batch_size, pred_length, output_size). 
                                            Default is None.
        Returns:
            tuple: A tuple containing:
                - out (torch.Tensor): Output tensor of shape (batch_size, pred_length, output_size).
                - quantile (torch.Tensor): Quantile tensor of shape (batch_size,).
        """
        lstm_out, _ = self.lstm(x)
        
        h = lstm_out[:,-1,:] 
        #out = self.lt(h)
        out = h
        #out = out.reshape(-1,self.pred_length,self.output_size)
        if cs is not None and self.cs is not False: # injected clearsky values
            out = cs.expand(-1,-1,self.output_size) + (out*self.cs_alpha)
        return out