import torch
import torch.nn as nn
from models.Calibrated_lattice_model import CalibratedLatticeModel
from models.LSTM import LSTM

class LSTM_Lattice(nn.Module):
    def __init__(self,lstm_paras: dict,lattice_paras: dict,params: dict):
        super(LSTM_Lattice, self).__init__()

        self.lstm = LSTM(**{k: v for k, v in lstm_paras.items() if v is not None})
        if params['_model_options'][params['_MODEL']] == 'LSTM_Lattice':
            self.out = CalibratedLatticeModel(**{k: v for k, v in lattice_paras.items() if v is not None})
        elif params['_model_options'][params['_MODEL']] == 'LSTM_Linear':
            self.out = nn.Linear(lstm_paras['hidden_size']+1, params['_PRED_LENGTH'],dtype=torch.float64) # +1 for quantile
        else:
            raise ValueError('Model not implemented')
        
        self.double_run = True if params['loss_option'][params['_LOSS']] == 'calibration_sharpness_loss' else False
        self.output_size = params['_PRED_LENGTH']


    def forward(self, x: torch.tensor,quantile: torch.tensor, cs=None, valid_run=False):
        h= self.lstm(x)
        # x = torch.cat((h, quantile.squeeze(-1)), dim=-1)
        out= []
        for i in range(self.output_size):
            c = torch.cat((h, quantile[...,i].unsqueeze(-1)), dim=-1)
            out.append(self.out(c))
        out = torch.stack(out, dim=-1)

        if self.double_run and not valid_run: # we are doing a double run as we need output for 1-quantile
            out = out.squeeze(-1)
            neg_quantile = 1-quantile.detach().clone()
            out_2 = []
            for i in range(self.output_size):
                c = torch.cat((h, neg_quantile[...,i].unsqueeze(-1)), dim=-1)
                out_2.append(self.out(c))
            out_2 = torch.stack(out_2, dim=-1).squeeze(-1)
            out = torch.stack([out, out_2], dim=-1)

        return out