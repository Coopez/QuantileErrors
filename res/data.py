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


class Data_Normalizer():
    """
    Class normalizing and outputting train, valid, train_target, valid_target, test, test_target
    unless not supplied, then it will output None.
    Always uses min-max normalization.
    """
    def __init__(self,
                 train: np.ndarray,
                 train_target: np.ndarray,
                 valid: np.ndarray = None,
                 valid_target: np.ndarray = None,
                 test: np.ndarray = None,
                 test_target: np.ndarray = None):
        """
        Initialize the Data_Normalizer with training, validation, and test data.

        Args:
            train (np.ndarray): Training data.
            train_target (np.ndarray): Training target data.
            valid (np.ndarray): Validation data.
            valid_target (np.ndarray): Validation target data.
            test (np.ndarray): Test data.
            test_target (np.ndarray): Test target data.
        """
        self.train = train
        self.train_target = train_target
        self.valid = valid
        self.valid_target = valid_target
        self.test = test
        self.test_target = test_target
        
        self.min_train = np.expand_dims(np.min(train, axis=0), axis=0)
        self.max_train = np.expand_dims(np.max(train, axis=0), axis=0)

        self.min_target = np.min(train_target, axis=0)
        self.max_target = np.max(train_target, axis=0)

        self.min_train_expanded = np.expand_dims(self.min_train.copy(), axis=0)
        self.max_train_expanded = np.expand_dims(self.max_train.copy(), axis=0)

        
    ## Helper functions, not supposed to be called directly
    def apply_minmax(self,var:str, data):
        if var == "train":
            return (data - self.min_train) / (self.max_train - self.min_train)
        elif var == "target":
            return (data - self.min_target) / (self.max_target - self.min_target)
        else:
            raise ValueError("Variable not recognized")
        
    
    #######################################
    def transform_all(self):
        """
        Apply min-max normalization to all datasets.

        Returns:
            tuple: Normalized train, train_target, valid, valid_target, test, test_target datasets.
        """
        local = []

        ## need to apply train min-max to all datasets as if they are not observable
        if self.train is not None:
            local.append(self.apply_minmax("train",self.train))
        if self.train_target is not None:
            local.append(self.apply_minmax("target",self.train_target))
        if self.valid is not None:
            local.append(self.apply_minmax("train",self.valid))
        if self.valid_target is not None:
            local.append(self.apply_minmax("target",self.valid_target))
        if self.test is not None:
            local.append(self.apply_minmax("train",self.test))
        if self.test_target is not None:
            local.append(self.apply_minmax("target",self.test_target))

        return local
    
    def inverse_transform(self, data, target: str="train"):
        if target == "train":
            denormed = data*(self.max_train_expanded - self.min_train_expanded) + self.min_train_expanded
            return denormed
        elif target == "target":
            denormed = data*(self.max_target - self.min_target) + self.min_target
            return denormed
        else:
            raise ValueError("Target not recognized")


    
# class Batch_Normalizer():
#     """
#     Normalizing everything in a batch of data
#     """
#     def __init__(self,data: torch.Tensor):
#         self.min = torch.min(data,dim=1).values.unsqueeze(1)
#         self.max = torch.max(data,dim=1).values.unsqueeze(1)



#     def transform(self,data: torch.Tensor):
#         tmin = self.min.repeat_interleave(data.shape[1], dim=1)
#         tmax = self.max.repeat_interleave(data.shape[1], dim=1)
#         return (data - tmin)/(tmax - tmin)
#     def inverse_transform(self,data: torch.Tensor, pos: int = 0):
#         tmin = self.min.repeat_interleave(data.shape[1], dim=1)
#         tmax = self.max.repeat_interleave(data.shape[1], dim=1)
#         return (data*(tmax[...,pos].unsqueeze(-1) - tmin[...,pos].unsqueeze(-1))) + tmin[...,pos].unsqueeze(-1)
    

## need batch norm with mean and std

class Batch_Normalizer():
    """
    Normalizing everything in a batch of data
    """
    def __init__(self,data: torch.Tensor):
        self.mean = torch.mean(data,dim=1).unsqueeze(1)
        self.std = torch.std(data,dim=1).unsqueeze(1)



    def transform(self,data: torch.Tensor):
        tmean = self.mean.repeat_interleave(data.shape[1], dim=1)
        tstd = self.std.repeat_interleave(data.shape[1], dim=1)
        return (data)/tstd
    def inverse_transform(self,data: torch.Tensor, pos: int = 0):
        tmean = self.mean.repeat_interleave(data.shape[1], dim=1)
        tstd = self.std.repeat_interleave(data.shape[1], dim=1)
        return (data*tstd[...,pos].unsqueeze(-1)) 