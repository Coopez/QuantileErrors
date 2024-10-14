import torch
import numpy as np
import torch.nn as nn  


class SQR_LSTM(nn.Module):
    def __init__(self,input_size,hidden_size,layers,window_size,pred_length,persistence_connection=False,loss_type='pinball', output_size = 1):
        super(SQR_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size + 1 # +1 for concat quantiles 
        self.output_size = output_size + 1 if loss_type == 'beyond' else output_size # +1 for quantile and 1-quantile
        self.layers = layers
        self.window_size = window_size
        self.pred_length = pred_length

        self.draw_func = np.random.uniform

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(self.input_size, self.hidden_size,num_layers = self.layers,batch_first=True,dtype=torch.float32)
        self.lt = nn.Linear(self.hidden_size,self.pred_length*self.output_size)

        self.cs = True
        self.cs_alpha = nn.Parameter(torch.tensor([2.0]), requires_grad= True if self.cs else False)
        

        self.embedding = nn.Linear(self.input_size+1,self.hidden_size) # check lars repo

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

        quantile= torch.tensor(self.draw_func(0,1,x.shape[0]),device= x.device,dtype=torch.float32) # quantile is a random number vector of size batch_size
        quantile_mat = quantile.repeat(1,self.pred_length,1).T # this keeps quantile only varying between batches, but we could do the same for the prediction length, i.e. the horizon
        x = torch.cat([x,quantile_mat],dim=-1)
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        
        h = lstm_out[:,-1,:] 
        out = self.lt(h)
        out = out.reshape(-1,self.pred_length,self.output_size)
        if cs is not None and self.cs is not False: # injected clearsky values
            out = cs.expand(-1,-1,self.output_size) + (out*self.cs_alpha)
        return out,quantile #torch.stack([lq, med, uq], axis=2)
    

