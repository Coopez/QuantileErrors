import numpy as np
import pandas as pd
#from sklearn.metrics import mean_squared_error

#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense,LSTM

#from sklearn.model_selection import TimeSeriesSplit


from pvlib.location import Location

##################### Translation to pytorch

import torch
from torch import nn

class Standartization():
    def __init__(self, input: torch.tensor, *label): 
        self.mean = input.mean(dim=1, keepdim=True)
        self.std= input.std(dim=1, keepdim=True)
        self.label = label
    def transform(self,input):
        return (input-self.mean)/self.std
    def retransform(self,input):
        return (input+self.mean)*self.std
    def show_parameter(self):
        print("Mean:"+ str(self.mean))
        print("Std:"+ str(self.std))
        print("Label:"+ str(self.label))
    

class Normalization():
    def __init__(self, input, *label): 
        self.min = input.min(dim=1, keepdim=True)
        self.max = input.max(dim=1, keepdim=True)
        self.label = label
    def transform(self,input):
        return (input-self.min)/(self.max - self.min)
    def retransform(self,input):
        return (input+self.min)*(self.max - self.min)
    def show_parameter(self):
        print("Mean:"+ str(self.mean))
        print("Std:"+ str(self.std))
        print("Label:"+ str(self.label))


#### Persistence


### Smart Persistence

class SmartPersistence(nn.Module):
    """
    I am assuming Batchsize = N
    Then, 1 forward pass processes the whole dataset(N*X)
    """
    def __init__(self, timerange, locations, shift, *args, **kwargs):
        super().__init__(SmartPersistence,*args, **kwargs)
        self.startdate = timerange[0]
        self.enddate= timerange[1]
        self.locations = locations
        self.shift = shift
        self.times = pd.date_range(start=timerange[0], end=timerange[1], freq='1h', tz="Europe/Berlin")

    def forward(self,x):
        

        cs = temp.get_clearsky(self.times).ghi.values
        csi = torch.tensor(irr/cs)
        csi = torch.roll(csi,-self.shift,0) # shifts the data towards the left
        csi[-self.shift:,:] = 0 # .roll rolls the falling off values arround to the other side. I want them to be 0
        result = torch.mul(csi,torch.tensor(cs))
        res.append(result)
        csl.append(cs)
        csil.append(csi)
        pass

### MLP


### LSTM






##################################################################




###############################
# need 
'''
- Functions which can scale with dimensionality
- assuming to work in np.arrays cuz (x,) issue
- normalization, persistence, rmse, mae, smart persistence, MLP, LSTM, Crossvalidation. 

'''

"""
NO WORKI atm :C
def normalize(input):
    #expect input in the format (xt,y) with y being a factor and xt the timeseries
    if len(np.shape(input)) == 1:
        x = len(input)
        input = input.reshape(x,1)
    else:
        x = len(input[0])     
    grd_std = 0
    grd_mean = 0
    for row in range(x):
        grd_std += np.nanstd(input[row,:])/x
        grd_mean += np.nanmean(input[row,:])/x
    return (input-grd_mean)/grd_std
"""
def persistence_model(input: np.array, shift: int, n_factor: int):  
    #expect input in the format (xt,y) with y being a factor and xt the timeseries
    if len(np.shape(input)) == 1:
        input = input.reshape(len(input),1)

    output = np.zeros((np.shape(input)[0]+shift,n_factor))
    output[shift:,:] = input
    return output[:-shift,:]


# def rmse(input,output):
#     #mask = ~np.isnan(input)
#     #mask2 =
#     #assert sum(np.isnan(input[mask])) == 0, "input has nans"
#     #print(input[mask])
#     #print(output[mask])\
#     try: 
#         input = np.transpose(input)
#         mask1 = ~np.isnan(input) 
#         mask2 = ~np.isnan(output)
#         mask=[]
#         for sub1,sub2 in zip(mask1,mask2):
#             mask.append( [a and b for a,b in zip(sub1,sub2)])
#         #print(sum(np.isnan(output[mask1+mask2])))
#     #for sub1,sub2 in  zip(mask1,mask2):
    
#     #    sub1 = [j for i,j in zip(sub1,sub2) if i!=j]
#     except:
#         input = np.transpose(input)
#         mask1 = ~np.isnan(input) 
#         mask2 = ~np.isnan(output)
#         mask=[a and b for a,b in zip(mask1,mask2)][0]
#         #mask = np.transpose(mask)
#         print(np.shape(mask))
#         print(np.shape(output))
#         output= np.transpose(output)
    
#     return mean_squared_error(input[mask],output[mask],squared=False)
    

def normalize(input):
    y = len(np.shape(input))
    x = len(input)
    grd_std = 0
    grd_mean = 0
    if y>1:
        for row in range(x):
            grd_std += np.nanstd(input[row,:])/x
            grd_mean += np.nanmean(input[row,:])/x
        return (input-grd_mean)/grd_std
    else:
        return (input-np.nanmean(input))/np.nanstd(input)
    

     # extract this: cs.ghi.values



def shift(arr, num, fill_value=np.nan):
    # apparently very efficient shift algorithm. Thanks, stack exchange.
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result

def smart_persistence(irrs: np.array,timerange: tuple, locations: np.array):
    """
    timerange expects ("YYYY-MM-DD","YYYY-MM-DD")
    location expects an array with [latitude,longitute,altitude] in every row
    Similarly, irrs has a row for every locations irradiance distribution

    """
    times = pd.date_range(start=timerange[0], end=timerange[1], freq='1h', tz="Europe/Berlin")
    print(times)
    res = list()
    csl = list()
    csil = list()
    for location,irr in zip(locations,irrs):
        temp = Location(location[0],location[1],"Europe/Berlin",location[2])
        cs = temp.get_clearsky(times)
        cs = cs.ghi.values
        print(cs)
        csi = irr/cs
        # need to shift cs
        csi = shift(csi,1,0)
        result = csi*cs
        res.append(result)
        csl.append(cs)
        csil.append(csi)
    return res, csl, csil



def get_csi(timerange:tuple, locations: np.array, irrs: np.array):
    """
    Returns clearsky and clearsky index.
    
    timerange expects ("YYYY-MM-DD","YYYY-MM-DD")
    location expects an array with [latitude,longitute,altitude] in every row
    Similarly, irrs has a row for every locations irradiance distribution
    """
    
    times = pd.date_range(start=timerange[0], end=timerange[1], freq='1h', tz="Europe/Berlin")
    csl = list()
    csil = list()
    for location,irr in zip(locations,irrs):
        temp = Location(location[0],location[1],"Europe/Berlin",location[2])
        cs = temp.get_clearsky(times)
        cs = cs.ghi.values
        if cs == 0:
            csi = cs
        else:
            csi = irr/cs
        csl.append(cs)
        csil.append(csi)
    return csl, csil




def get_day(timerange:tuple, locations: np.array):
    """
    Returns a boolean index for day/night times. -> TRUE for day, FALSE for night
    timerange expects ("YYYY-MM-DD","YYYY-MM-DD")
    location expects an array with [latitude,longitute,altitude] in every row
    """
    times = pd.date_range(start=timerange[0], end=timerange[1], freq='1h', tz="Europe/Berlin")
    dayl=list()
    for location in locations:
        temp = Location(location[0],location[1],"Europe/Berlin",location[2])
        cs = temp.get_clearsky(times)
        cs = cs.ghi.values
        day = [value>0 for value in cs]
        dayl.append(day)
    return dayl

def get_day_from_cs(cs:np.array):
    shape = cs.shape
    if shape[1] > shape[0]:
        cs = np.transpose(cs)
    dayl=list()
    for loc in range(cs.shape[1]):
        day = [value>0 for value in cs[:,loc]]
        dayl.append(day)
    return dayl


def split_sequence(sequence, n_steps,horizon):
    X, y = list(), list()
    if sequence.ndim > 1: #gotta check formating in the input
        # for now I expect for dims{a,b} a<b which needs to be switched here
        sequence = sequence.transpose()

    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-horizon:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix+horizon-1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def mlp(input: np.array, window: int, neurons: int, epochs: int, dimensions: int):
    X,y = split_sequence(input,window,1)
    if y.ndim > 1: 
        y = y[:,0]
    #print(X[2000],y[2000])
    X = X.reshape(X.shape[0],window*dimensions,1)
    model = Sequential()
    model.add(Dense(units=neurons, activation='relu', input_dim=window*dimensions))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    # fit model
    model.fit(X, y, epochs=epochs, verbose=2) # verbose is type of output, 0 is nothing
    # demonstrate prediction
    #x_input = array([70, 80, 90])
    #x_input = x_input.reshape((1, n_steps))
    #yhat = model.predict(x_input, verbose=0)
    #print(yhat)
    return model

def lstm(input: np.array, window: int, neurons: int, epochs: int, dimensions: int):
    X,y = split_sequence(input,window,1)
    if y.ndim > 1: 
        y = y[:,0]
    X = X.reshape(X.shape[0],window,dimensions)
    model = Sequential()
    #model.add(Embedding(input_dim=len(input),output_dim=window))
    model.add(LSTM(units=neurons,input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    # fit model
    model.fit(X, y, epochs=epochs, verbose=0) # verbose is type of output, 0 is nothing
    # demonstrate prediction
    #x_input = array([70, 80, 90])
    #x_input = x_input.reshape((1, n_steps))
    #yhat = model.predict(x_input, verbose=0)
    #print(yhat)
    return model


# def cross_val_split(k, type):
#     if type == "timeseries":
#         split = TimeSeriesSplit(n_splits=k)
#     assert type=="timeseries","No other functions than timeseries are implemented for now"
#     return split

# def optimize_mlp(data,k: int,windows: list,N: list, epochs: list,dimensions: int):
#     # EXPECTS: a TRAINING data set
#     cv = cross_val_split(k,"timeseries")
#     configs = [[h,i,j] for h in windows for i in N for j in epochs]
#     avg_losses=[]
#     counter = 0
#     for config in configs:
#         avg_loss = 0
#         for fold,(train_index, test_index) in enumerate(cv.split(data)):
#             model_it= mlp(data[train_index],window=config[0],neurons=config[1],epochs=config[2],dimensions=dimensions)
#             # need to split and shift the test_index
#             X,y = split_sequence(data[test_index],config[0],1)
#             forecast = model_it.predict(X,verbose=0)
#             avg_loss += mean_squared_error(y,forecast,squared=False)/k
#         counter +=1
#         print("Step " + str(counter)+ "/" + str(len(configs)))
#         avg_losses.append([avg_loss,config])
#     avg_losses.sort()
#     return avg_losses

# def optimize_lstm(data,k: int,windows: list,N: list, epochs: list,dimensions: int):
#     # EXPECTS: a TRAINING data set
#     cv = cross_val_split(k,"timeseries")
#     configs = [[h,i,j] for h in windows for i in N for j in epochs]
#     avg_losses=[]
#     counter = 0
#     for config in configs:
#         avg_loss = 0
#         for fold,(train_index, test_index) in enumerate(cv.split(data)):
#             model_it= lstm(data[train_index],window=config[0],neurons=config[1],epochs=config[2],dimensions=dimensions)
#             # need to split and shift the test_index
#             X,y = split_sequence(data[test_index],config[0],1)
#             forecast = model_it.predict(X,verbose=0)
#             avg_loss += mean_squared_error(y,forecast,squared=False)/k
#         avg_losses.append([avg_loss,config])
#         counter +=1
#         print("Step " + str(counter)+ "/" + str(len(configs)))
#     avg_losses.sort()
#     return avg_losses


def split_databytime(times):
    """
    splitting data into winter and summer via a pd.Datetimeindex.
    input is the Datetimeindex.
    output is an array of indeces which indicate where to split the data.
    """
    out = []

    winter_start = pd.Timestamp("")
    summer_start = pd.Timestamp("")

    for time in times:
        [idx for idx, date in enumerate(time) if date in specific_dates]
        indices = [time.get_loc(date) for date in (winter_start,summer_start) if date in time]
        out.append(indices)

    return np.array(out)


def nan_helper(y):
    # Directly adapted from https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array

    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()

def nan_interpolater(data):
    pass