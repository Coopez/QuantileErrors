import torch
import torch.nn as nn
from models.Calibrated_lattice_model import CalibratedLatticeModel
from models.LSTM import LSTM

class LSTM_Lattice(nn.Module):
    def __init__(self,lstm_paras: dict,lattice_paras: dict,loss_option: str='pinball'):
        super(LSTM_Lattice, self).__init__()

        self.lstm = LSTM(**{k: v for k, v in lstm_paras.items() if v is not None})
        self.lattice = CalibratedLatticeModel(**{k: v for k, v in lattice_paras.items() if v is not None})

        self.output_size = 2 if loss_option == 'calibration_sharpness_loss' else 1
    def forward(self, x: torch.tensor,quantile: torch.tensor, cs=None):
        h= self.lstm(x)
        # x = torch.cat((h, quantile.squeeze(-1)), dim=-1)
        out= []
        for i in range(self.output_size):
            c = torch.cat((h, quantile[...,i]), dim=-1)
            out.append(self.lattice(c))
        out = torch.stack(out, dim=-1)
        #out = self.lattice(x)
        # h = self.lstm(x)
        # c = torch.cat((h, quantile.squeeze(-1)), dim=-1)
        # out = self.lattice(c)
        return out