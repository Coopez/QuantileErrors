import torch
import numpy as np
import matplotlib.pyplot as plt


class Debug_model():
    def __init__(self,model, x: torch.tensor,y: torch.tensor, quantiles=[0.01,0.25,0.5,0.75,0.99]):
        self.model = model
        self.x = x
        self.y = y
        self.quantiles = quantiles
        cdf = []
        for q in quantiles:
            q_in = torch.tensor(q).repeat(x.shape[0],1).to(x.device)
            pred = model(x,q_in,valid_run=True)
            cdf.append(pred)
        self.cdf = torch.stack(cdf,dim=-1).squeeze(-2)
    def plot_out(self, batch_index=0):
  
        y_pred_plot = self.cdf.detach().cpu().numpy()
        y_plot = self.y.detach().cpu().numpy()
        plt.plot(y_plot[batch_index],'-o',label='True')
        for i,q in enumerate(self.quantiles):
            plt.plot(y_pred_plot[batch_index,:,i],'-o',label=f'Pred {q}')
        plt.legend()
        plt.show(block=True)

    