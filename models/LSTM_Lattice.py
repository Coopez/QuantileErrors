import torch
import torch.nn as nn
from models.Calibrated_lattice_model import CalibratedLatticeModel
from models.LSTM import LSTM

class LSTM_Lattice(nn.Module):
    def __init__(self,lstm_paras,lattice_paras):
        super(LSTM_Lattice, self).__init__()


        self.lstm = LSTM(**{k: v for k, v in lstm_paras.items() if v is not None})
        self.lattice = CalibratedLatticeModel(**{k: v for k, v in lattice_paras.items() if v is not None})
    def forward(self, x: torch.tensor,quantile: torch.tensor, cs=None):
        h= self.lstm(x)
        x = torch.cat((h, quantile.squeeze(-1)), dim=-1)
        out = self.lattice(x)
        return out