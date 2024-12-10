import torch
import numpy as np
import matplotlib.pyplot as plt
from res.data import Batch_Normalizer

class Debug_model():
    def __init__(self,model, x: torch.tensor,y: torch.tensor, quantiles=[0.01,0.25,0.5,0.75,0.99]):
        self.model = model
        self.x = x
        self.y = y
        self.quantiles = quantiles
        cdf = []
        # bn = Batch_Normalizer(x)
        for q in quantiles:
            q_in = torch.tensor(q).repeat(x.shape[0],1).to(x.device)
            # x = bn.transform(x)
            pred = model(x,q_in,valid_run=True)
            # pred = bn.inverse_transform(pred)
            cdf.append(pred)
        self.cdf = torch.stack(cdf,dim=-1).squeeze(-2)
    def plot_out(self, batch_index=0):
  
        y_pred_plot = self.cdf.detach().cpu().numpy()
        y_plot = self.y.detach().cpu().numpy()
        plt.plot(y_plot[batch_index,...],'-o',label='True')
        for i,q in enumerate(self.quantiles):
            plt.plot(y_pred_plot[batch_index,:,i],'-o',label=f'Pred {q}')
        plt.legend()
        plt.show(block=True)

    def plot_intervals(self, batch_index=0):
        y_pred_plot = self.cdf.detach().cpu().numpy()
        y_plot = self.y.detach().cpu().numpy()
        
        plt.plot(y_plot[batch_index,...],'-o',label='True')
        
        median_index = self.quantiles.index(0.5)
        median_pred = y_pred_plot[batch_index,:,median_index]
        plt.plot(median_pred, '-o', label='Pred 0.5')
        
        for i in range(len(self.quantiles) - 1):
            lower_bound = y_pred_plot[batch_index,:,i]
            upper_bound = y_pred_plot[batch_index,:,i+1]
            plt.fill_between(range(y_plot.shape[1]), lower_bound, upper_bound, color='orange', alpha=0.5)
        
        plt.legend()
        plt.show(block=True)