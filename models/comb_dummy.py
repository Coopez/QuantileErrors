import torch
import torch.nn as nn
from models.Calibrated_lattice_model import CalibratedLatticeModel
from models.SQR_LSTM_Lattice import SQR_LSTM_Lattice

class LSTM_Lattice(nn.Module):
    def __init__(self,lstm_paras,lattice_paras):
        super(LSTM_Lattice, self).__init__()
        self.lstm = SQR_LSTM_Lattice(*lstm_paras)
        self.lattice = CalibratedLatticeModel(*lattice_paras)
    def forward(self, x: torch.tensor,quantile: torch.tensor, cs=None):
        h= self.lstm(x)
        x = torch.cat((h, quantile.squeeze(-1)), dim=-1)
        out = self.lattice(x)
        return out