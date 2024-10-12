import torch
import torch.nn as nn
import numpy as np

# Define a data set
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, window, horizon, data=None, targets=None):

        # need a data simulation based on a sin here but as tensor
        if data is None and targets is None:
            self.data = torch.tensor(np.sin(np.linspace(0, 100, 1000)).astype(np.float32))
            self.targets = torch.tensor(np.sin(np.linspace(0, 100, 1000)).astype(np.float32))
            
        else:
            self.data = torch.tensor(data)
            self.targets = torch.tensor(targets)
            if self.data.dim() == 1:
                self.data = self.data.unsqueeze(-1)
                 #self.targets = self.targets.unsqueeze(-1)
        self.window = window
        self.horizon = horizon

        self.scaler_data = MinMaxScaler()
        self.scaler_target = MinMaxScaler()
        self.scaler_data.fit(X=self.data)
        self.scaler_target.fit(X=self.targets)
        self.data = self.scaler_data.transform(self.data)
        self.targets = self.scaler_target.transform(self.targets)
        
    def __getitem__(self, index):
        x = self.data[index:index+self.window]
        y = self.targets[index+self.window:index+self.horizon+self.window]
        
        return x, y
    
    def __len__(self):
        return len(self.data) - self.window - self.horizon + 1
    

class MinMaxScaler():
    def __init__(self,loss_dist=None):
        self.min = 0.0
        self.max = 0.0
        self.eps = 1e-4
        self.loss_dist = loss_dist
    def fit(self,X):
        self.min = torch.min(X,dim=0)[0]
        self.max = torch.max(X,dim=0)[0]
    def transform(self,X):
        if X.dim() == 1:
            y = (X-self.min)/(self.max-self.min)
        else:
            y = (X-self.min[ None,:] )/(self.max[None,:]-self.min[None,:])
        if self.loss_dist == "Weibull":
            return y + self.eps
        elif self.loss_dist == "JohnsonsSB":
            return y - self.eps
        else:
            return y
    def inverse_transform(self,X):
        y = (X*(self.max[ None,:]-self.min[None,:])) + self.min[ None,:]
        if self.loss_dist == "Weibull":
            return y - self.eps
        elif self.loss_dist == "JohnsonsSB":
            return y + self.eps
        return  y


def data_import():
    filename = "readyDat_time_csi_nights_allseason_nora_cam"
    data = np.load("res/"+filename+ ".npz")
    return data['flat_train'], data['target_train'], data['flat_valid'] ,data['target_valid'], data['flat_test'], data['target_test']


# dataset = SimpleDataset(window=10, horizon=5)
# # make dataloader
# data = torch.utils.data.DataLoader(dataset, batch_size=3, shuffle=False)
# print(len(dataset))  # Should print 986 (1000 - 10 - 5 + 1)
# for batch in data:
#     x,y = batch
#     print(x.shape)
#     print(y.shape)    # Should print the first window and horizon