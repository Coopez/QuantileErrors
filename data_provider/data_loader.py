import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset
#from  pytorch_forecasting.data.timeseries import TimeSeriesDataSet
#from  pytorch_forecasting.data.encoders import TorchNormalizer
import warnings
import torch
from pvlib.irradiance import clearsky_index

import sys
parent_dir = os.path.dirname(os.getcwd())
current_dir = os.getcwd()
sys.path.append(parent_dir)
sys.path.append(current_dir)
from data_provider.download_functions import Sunpoint_import,CAN_import,sliding_window_sum
#warnings.filterwarnings('ignore')


from utils.res import get_day_from_cs, get_csi

### I GIVE UP. pytorch_forecasting SEEMS TO BE A HOT PILE OF GARBAGE WITH WRONG DOCUMENTATION AND BREAKING ISSUES IN THE CODE EVERYWHERE I LOOK. NO THANK YOU.
"""class Dataset_SUNCAM(TimeSeriesDataSet):
    
    def __init__(self,dataset_names: list, horizon : tuple, window: list, freq = "15min", step = 1, CSI = True):
        
        
        self.imports = dataset_names
        self.freq = freq
        self.data = self.data_formater(self.data_importer(CSI), dataset_names)
        
        super().__init__( #init TimeseriesDataset in the wrapper.
            self.data,
            group_ids=["groups"],
            target = self.imports[0], # first of dataset names is being taken as target timeseries to be forecast
            time_idx = "time_idx",
            max_encoder_length = window,
            min_prediction_length = horizon[0],
            max_prediction_length = horizon[1],
            time_varying_unknown_reals = self.imports,
            target_normalizer = [TorchNormalizer]
        )

    def data_importer(self, CSI)-> list:

        can = CAN_import()
        s,e,time,sun = Sunpoint_import()

        if self.freq == "15min":
            sun_values = np.zeros((20,time.shape[0]))
            for i in range(0,20):
                sun_values[i,:] = sun.loc[(slice(None), i), "rsds"]
            can_values= sliding_window_sum(can["ghi"],4,1)
            cs = sliding_window_sum(can["cs"],4,1)
        elif self.freq == "1h":
            # Format data to be the same layout - only needed if can is downloaded every hour
            sun_values = np.zeros((20,time.shape[0]))
            for i in range(0,20):
                sun_values[i,:] = sun.loc[(slice(None), i), "rsds"]
            can_values = can["ghi"]
            cs = can["cs"]
        assert sun_values.shape == can_values.shape, "Format of sun and can isnt the same."
        if CSI:
            csi = clearsky_index(sun_values,cs)
            assert sun_values.shape == can_values.shape == csi.shape, "Format of sun, can, or csi isnt the same."
            return [sun_values,can_values,csi]
        else:
            return [sun_values,can_values]

        

    def data_formater(self,data: list, var_list: list[str]):
        
        # need one column for sun, cam, and cs(i).  should all be diff dataset

        cols = var_list + ["time_idx", "groups"]

        dt = {var: [] for var in cols}
        group_count = 0
        it_count = 0
        for name,set in zip(var_list,data):
            
            dt[name] = set.flatten()
            assert len(set.flatten().shape) == 1, "Flatten no worki " + str(set.flatten().shape)
            s = set.shape

            if it_count ==0:
                for gr in range(group_count,group_count+ np.prod(s[1:])):
                    dt["time_idx"] += [i for i in range(0,s[0])] # time index for each var in dt 0...T
                    dt["groups"] += [gr for i in range(0,s[0])] # group identifier based on how many vars there are
                group_count += np.prod(s[1:])
            it_count+=1
        return pd.DataFrame(dt)
"""

class StandardScaler():
    def __init__(self):
        self.mean = 0.0
        self.std = 0.0
        self.eps = 1e-5
    def fit(self,X):
        self.mean = torch.mean(X,dim=0)
        self.std = torch.std(X,dim=0)
    def transform(self,X):
        return (X-self.mean[ None:,] )/ (self.std[None,:] + self.eps)
    def inverse_transform(self,X):
        return (X*(self.std[ None:,]+self.eps)) + self.mean[ None:,]
    
class MinMaxScaler():
    def __init__(self,loss_dist=None):
        self.min = 0.0
        self.max = 0.0
        self.eps = 1e-4
        self.loss_dist = loss_dist
    def fit(self,X):
        # using eps to widen the interval avoid 0.0 and 1.0 values as input to iPDFs.
        self.min = torch.min(X,dim=0)[0] #-self.eps
        #self.min = torch.where(self.min == 0.0, self.min + self.eps, self.min)
        #self.min = torch.where(self.min < 0.0, self.min - self.eps, self.min)
        self.max = torch.max(X,dim=0)[0] #+ self.eps
    def transform(self,X):
        y = (X-self.min[ None:,] )/(self.max[None,:]-self.min[None,:])
        # if torch.where(y == 0.0, True, False).any():
        #     print("0.0 in minmax")
        # if torch.where(y < 0.0, True, False).any():
        #     print("<0.0 in minmax")
        if self.loss_dist == "Weibull":
            return y + self.eps
        elif self.loss_dist == "JohnsonsSB":
            return y - self.eps
        else:
            return y
    def inverse_transform(self,X):
        y = (X*(self.max[ None:,]-self.min[None,:])) + self.min[ None:,]
        if self.loss_dist == "Weibull":
            return y - self.eps
        elif self.loss_dist == "JohnsonsSB":
            return y + self.eps
        return  y
    
class Batch_Normalizer():
    def __init__(self,scaler="standard"):
        self.scaler = scaler
        if self.scaler == "standard":
            self.mean = 0.0
            self.std = 0.0
            self.mean_t = 0.0
            self.std_t = 0.0
        elif self.scaler == "minmax":
            self.min = 0.0
            self.max = 0.0
            self.min_t = 0.0
            self.max_t = 0.0
        else:
            raise NotImplementedError
        self.eps = 1e-5
    def fit(self,X):
        X_target = X[...,11]
        if self.scaler == "standard":
            self.mean = torch.mean(X,dim=1)
            self.std = torch.std(X,dim=1)
            self.mean_t = torch.mean(X_target,dim=1)
            self.std_t = torch.std(X_target,dim=1)

        elif self.scaler == "minmax":
            # using eps to widen the interval avoid 0.0 and 1.0 values as input to iPDFs.
            self.min = torch.min(X,dim=1)[0] #- self.eps
            self.max = torch.max(X,dim=1)[0] #+ self.eps
            self.min_t = torch.min(X_target,dim=1)[0] #- self.eps
            self.max_t = torch.max(X_target,dim=1)[0] #+ self.eps

    def transform(self,X,data_type="input"):
        if self.scaler == "standard":
            if  data_type=="output":  #X.shape[-1] == 1:
                transformed = (X-self.mean_t.unsqueeze(1) )/ (self.std_t.unsqueeze(1) + self.eps)
                return transformed
            transformed = (X-self.mean.unsqueeze(1) )/ (self.std.unsqueeze(1) + self.eps)
            return transformed
        elif self.scaler == "minmax":
            if data_type=="output":
                transformed = (X-self.min_t.unsqueeze(1) )/(self.max_t.unsqueeze(1)-self.min_t.unsqueeze(1)+ self.eps)
                return transformed+ self.eps
            transformed = (X-self.min.unsqueeze(1) )/(self.max.unsqueeze(1)-self.min.unsqueeze(1)+ self.eps)
            return transformed+ self.eps
        else:
            raise NotImplementedError
    def inverse_transform(self,X,data_type="input"):
        if self.scaler == "standard":
            if data_type == "output":
                return (X*(self.std_t.unsqueeze(1).unsqueeze(1)+self.eps)) + self.mean_t.unsqueeze(1).unsqueeze(1) - self.eps
            return (X*(self.std.unsqueeze(1)+self.eps)) + self.mean.unsqueeze(1)- self.eps
        elif self.scaler == "minmax":
            if data_type=="output":
                return (X*(self.max_t.unsqueeze(1).unsqueeze(1)-self.min_t.unsqueeze(1).unsqueeze(1)+ self.eps)) + self.min_t.unsqueeze(1).unsqueeze(1)- self.eps
            return (X*(self.max.unsqueeze(1)-self.min.unsqueeze(1)+ self.eps)) + self.min.unsqueeze(1)- self.eps
        else:
            raise NotImplementedError



class Dataset_SUNCAM(Dataset):

    def __init__(self,data:list,target:list,horizons: tuple,window: int, scaler, cs_input=None,loss_dist = None):
        
        super(Dataset_SUNCAM,self).__init__()
        
        assert len(data.shape) <= 2, "there needs to be max 2 dims in the data."
        assert data.shape[0] > data.shape[1], "data likely in an incorrect layout. Expects (sample,features)" # assuming sample_n > feature_n
        if len(target.shape) == 1:
            target = target.reshape(-1, 1)
        #assert target.shape[0] > target.shape[1], "target likely in an incorrect layout. Expects (sample,features)"
        self.data = torch.from_numpy(data).to(dtype=torch.float32)
        self.target = torch.from_numpy(target).to(dtype=torch.float32)
        self.horizons = horizons
        self.window = window
        if cs_input is not None:
            self.cs = torch.from_numpy(cs_input).to(dtype=torch.float32).unsqueeze(-1)
        else:
            self.cs = cs_input
        if scaler == "standard":
            self.scaler_data = StandardScaler()
            self.scaler_target = StandardScaler()
            self.scaler_z = StandardScaler()
        elif scaler == "minmax":
            self.scaler_data = MinMaxScaler(loss_dist=loss_dist)
            self.scaler_target = MinMaxScaler(loss_dist=loss_dist)
            self.scaler_z = MinMaxScaler(loss_dist=loss_dist)
        else:
            raise NotImplementedError
        #self.scaler_fitted_data = 
        self.scaler_data.fit(X=self.data) # note to self: scaler needs data of layout (sample,features)
        #target = target.reshape(-1, 1)
        #print(target.shape)
        #self.scaler_fitted_target = 
        self.scaler_target.fit(X=self.target)
        if cs_input is not None:
            self.scaler_z.fit(X=self.cs)

    def __len__(self):
        return len(self.data) - (self.window + max(self.horizons))

    def __getitem__(self, idx):
        window = self.data[idx : idx + self.window]
        target = self.target[idx + min(self.horizons)+self.window -1 : idx + max(self.horizons)+self.window ]
        assert len(window) != 0 and len(target) !=0, "Returned empty batch at " + str(idx)
        assert len(target) == max(self.horizons), "target is not the right length."


        if self.cs == None:
            return self.scaler_data.transform(window), self.scaler_target.transform(target) #.astype('float32')
        else:
            # doesnt need to be scaled because it is already scaled. (see function)
            # only need target idx for injection into model
            cs_data = self.cs[idx + min(self.horizons)+self.window -1 : idx + max(self.horizons)+self.window ]
            return self.scaler_data.transform(window), self.scaler_target.transform(target), self.scaler_z.transform(cs_data)
    
    def  inverse_transform(self,data,z=None): # assuming i transform for the predicted var
        if len(data.size()) > 2:
            for ib in range(data.shape[0]):
                data[ib] = self.scaler_target.inverse_transform(data[ib])
        else: 
            data = self.scaler_target.inverse_transform(data)
        if z is not None:
            z = self.scaler_z.inverse_transform(z)
            return data,z
        return data

class Dataset_SUNCAM_AutoReg(Dataset):
    def __init__(self,data:list,target:list,horizons: tuple,window: int, scaler, cs_input=None) -> None:
        super(Dataset_SUNCAM_AutoReg,self).__init__()  
        assert len(data.shape) <= 2, "there needs to be max 2 dims in the data."
        assert data.shape[0] > data.shape[1], "data likely in an incorrect layout. Expects (sample,features)" # assuming sample_n > feature_n
        if len(target.shape) == 1:
            target = target.reshape(-1, 1)

        self.data = torch.from_numpy(data)
        self.target = torch.from_numpy(target)
        self.horizons = horizons
        self.window = window
        if cs_input is not None:
            self.cs = torch.from_numpy(cs_input)
        else:
            self.cs = cs_input
        if scaler == "standard":
            self.scaler_data = StandardScaler()
            self.scaler_target = StandardScaler()
        elif scaler == "minmax":
            self.scaler_data = MinMaxScaler()
            self.scaler_target = MinMaxScaler()
        else:
            raise NotImplementedError
        self.scaler_data.fit(X=self.data) # note to self: scaler needs data of layout (sample,features)
        self.scaler_target.fit(X=self.target)
    def __len__(self):
        return len(self.data) - (self.window + max(self.horizons))
    def __getitem__(self, idx):
        window = self.data[idx : idx + self.window]
        target = self.target[idx + min(self.horizons)+self.window -1 : idx + max(self.horizons)+self.window ]
        assert len(window) != 0 and len(target) !=0, "Returned empty batch at " + str(idx)
        assert len(target) == max(self.horizons), "target is not the right length."
        if self.cs == None:
            return self.scaler_data.transform(window), self.scaler_target.transform(target) #.astype('float32')
        else:
            cs_data = self.cs[idx + min(self.horizons)+self.window -1 : idx + max(self.horizons)+self.window ]
            return self.scaler_data.transform(window), self.scaler_target.transform(target), cs_data.unsqueeze(-1)
    def  inverse_transform(self,data): # assuming i transform for the predicted var
        if len(data.size()) > 2:
            for ib in range(data.shape[0]):
                data[ib] = self.scaler_target.inverse_transform(data[ib])
        else: 
            data = self.scaler_target.inverse_transform(data)  
        return data
    
class Dataset_SUNCAM_Graph(Dataset):
    pass

# class Data_SUNCAM_Googlestyle(Dataset):  
#     def __init__(self,data:list,target:list,horizons: tuple,window: int, scaler):
#         super(Dataset_SUNCAM,self).__init__()

#         assert len(data.shape) <= 2, "there needs to be max 2 dims in the data."
#         assert data.shape[0] > data.shape[1], "data likely in an incorrect layout. Expects (sample,features)" # assuming sample_n > feature_n
#         if len(target.shape) == 1:
#             target = target.reshape(-1, 1)
#         #assert target.shape[0] > target.shape[1], "target likely in an incorrect layout. Expects (sample,features)"
        
#         self.data = torch.from_numpy(data)
#         self.target = torch.from_numpy(target)
        
#         self.horizons = horizons
#         self.window = window
#         if scaler == "standard":
#             self.scaler_data = StandardScaler()
#             self.scaler_target = StandardScaler()
#         else:
#             raise NotImplementedError
#         #self.scaler_fitted_data = 
#         self.scaler_data.fit(X=self.data) # note to self: scaler needs data of layout (sample,features)
#         #target = target.reshape(-1, 1)
#         #print(target.shape)
#         #self.scaler_fitted_target = 
#         self.scaler_target.fit(X=self.target)

#     def __len__(self):
#         return len(self.data) - (self.window + max(self.horizons))

#     def __getitem__(self, idx):
#         window = self.data[idx : idx + self.window]
#         target = self.target[idx + min(self.horizons)+self.window -1 : idx + max(self.horizons)+self.window ]
#         assert len(window) != 0 and len(target) !=0, "Returned empty batch at " + str(idx)
#         assert len(target) == max(self.horizons), "target is not the right length."
#         return self.scaler_data.transform(window), self.scaler_target.transform(target) #.astype('float32')
    
#     def  inverse_transform(self,data): # assuming i transform for the predicted var
#         if len(data.size()) > 2:
#             for ib in range(data.shape[0]):
#                 data[ib] = self.scaler_target.inverse_transform(data[ib])
#         else: 
#             data = self.scaler_target.inverse_transform(data)
#         return data
    
    
### prob need function to transform nparray to dataframe with correct layout


def print_dict_shapes(d: dict):
    for element in d:
        print(element + ": " + str(np.array(d[element]).shape))

def embedd_time(time):
    
    hour = time.hour.to_numpy(dtype = "float32")
    hsin = np.sin(hour*(2*np.pi/time.hour.unique().shape[0]))
    hcos = np.cos(hour*(2*np.pi/time.hour.unique().shape[0]))
    
    day = time.isocalendar().day.to_numpy(dtype = "float32")
    dsin = np.sin(day*(2*np.pi/time.isocalendar().day.unique().shape[0]))
    dcos = np.cos(day*(2*np.pi/time.isocalendar().day.unique().shape[0]))
    
    week = time.isocalendar().week.to_numpy(dtype = "float32")
    wsin = np.sin(week*(np.pi/time.isocalendar().week.unique().shape[0]))
    wcos = np.cos(week*(np.pi/time.isocalendar().week.unique().shape[0]))
    
    return np.transpose(np.vstack((hsin,hcos,dsin,dcos,wsin,wcos)))


def import_SUNCAM(path: str = None, CSI: bool = True,Time: bool = True, loc=None, freq = "15min",exclude_nights = True, include_cam = True, deduct_cs = None)-> list:
    
    if path==None:
        # assuming data is in current directory
        can = CAN_import()
        _,_,time,sun = Sunpoint_import()
        result = list()
    else:
        can = CAN_import(path)
        _,_,time,sun = Sunpoint_import(path)
        result = list()

    if freq == "15min":
        sun_values = np.zeros((20,time.shape[0]))
        for i in range(0,20):
                sun_values[i,:] = sun.loc[(slice(None), i), "rsds"]
        can_values= sliding_window_sum(can["ghi"],4,1)
        #cs,_ = get_csi(time,loc,sun_values)
        cs = sliding_window_sum(can["cs"],4,1)
        #print("This is using the old CAM CS. Proceed at your own peril.")
    elif freq == "1h":
        # Format data to be the same layout - only needed if can is downloaded every hour
        sun_values = np.zeros((20,time.shape[0]))
        for i in range(0,20):
            sun_values[i,:] = sun.loc[(slice(None), i), "rsds"]
        can_values = can["ghi"]
        cs = can["cs"]
        #assert loc,"Need to input locations for this now."
        #cs,_ = get_csi(time,loc,sun_values)
    assert sun_values.shape == can_values.shape, "Format of sun and can isnt the same."
    if exclude_nights:
        day_bool = np.array(get_day_from_cs(cs))[11]
    else:
        day_bool = np.ones(cs.shape,dtype = bool)[11]
    
    
    result.append(sun_values[:,day_bool])
    
    if include_cam:
        result.append(can_values[:,day_bool])

    if CSI:
        # COMMENT: Replacing CSI with CS here for now.
        # csi = clearsky_index(sun_values,cs)
        # assert sun_values.shape == can_values.shape == csi.shape, "Format of sun, can, or csi isnt the same."
        # result.append(csi[:,day_bool])
        result.append(cs[:,day_bool])

    ## Data shifts in station 11
    night_bool = np.array(get_day_from_cs(cs))[11] == False
    result = np.array(result)
    off1 = result[0,11,:7543]
    off2 = result[0,11,7543:]
    result[0,11,:7543] = result[0,11,:7543] - np.nanmean(off1[night_bool[:7543]]) 
    result[0,11,7543:] = result[0,11,7543:] - np.nanmean(off2[night_bool[7543:]])
    
    if deduct_cs:
        if deduct_cs == "minus":
            result[0] = result[0] - result[2] # deduct  cs
        elif deduct_cs == "divide":
             result[0] = result[0] / (result[2]+0.0001)
        else:
            print("This deduct_cs command is unkown.")
            raise NameError
    
    if Time:
        etime = embedd_time(time)
        return  time[day_bool], etime[day_bool,:], np.array(result)
    else:
        return 0,result

def shift(arr, num, fill_value=np.nan):
    """ASSUMES 1D array!! Intended to generate target by shifting observational dependent variable"""
    result = np.empty_like(arr)
    assert len(arr.shape) == 1, "assumes 1d array input. Assertion breaks otherwise."
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result

def return_cs(loc:str):
    """
    Returns cs values with 5er split from Cams. Assumes station 11 as target.
    """
    can = CAN_import(loc)
    #_,_,_,sun = Sunpoint_import(_loc_data+"/data")
    cs = can["cs"][11]
    cs = sliding_window_sum(can["cs"],4,1)[11]
    i = int((1/5)*len(cs))
    j = int((2/5)*len(cs))
    day_truth = np.array([i!=0 for i in cs ])
    cs_train = (cs[j:] - np.mean(cs[j:])) / np.std(cs[j:])
    cs_valid = (cs[i:j] - np.mean(cs[i:j])) / np.std(cs[i:j])
    cs_test = (cs[:i]- np.mean(cs[:i])) / np.std(cs[:i])
    return cs_valid, cs_test, cs_train, [day_truth[i:j],day_truth[:i],day_truth[j:]],{"valid":[np.mean(cs[i:j]),np.std(cs[i:j])], "test":[np.mean(cs[:i]),np.std(cs[:i])], "train":[np.mean(cs[j:]),np.std(cs[j:])]}