import numpy as np
import torch
import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from res.data import data_import, Data_Normalizer
from data_provider.data_loader import return_cs

class sPersistence_Forecast():
    """
    Class for smart persistence forecast. 
    Just fetches from a precomputed pickle file
    """
    def __init__(self,normalizer,params):
        self.dat = pd.read_pickle("models/sunpoint_smart_persistence.pkl")
        self.index = self.dat.index
        self.data = np.array(self.dat["Value"].tolist(),dtype=np.float32)
        self.data = torch.tensor(self.data).float()
        self.min = normalizer.min_target
        self.max = normalizer.max_target
        self.data_normalized = (self.data-self.min)/(self.max-self.min)

    def forecast_raw(self,timestamp):
        persistence = self.data[timestamp,:]
        # persistence = self._rolling_mean(input_tensor=persistence)[:,self.target_summary-1::self.target_summary]
        # persistence = persistence.view(-1, self.target_summary).mean(dim=-1, keepdim=True).view(-1,self.current_horizon)
        return persistence
    def forecast(self,timestamp):
        """
        Forecasts the smart persistence for a given timestamp.
        """
        persistence = self.data_normalized[timestamp,:]
        # persistence = self._rolling_mean(input_tensor=persistence)[:,self.target_summary-1::self.target_summary]
        # persistence = persistence.view(-1, self.data.shape[-1]).mean(dim=-1, keepdim=True).view(-1,self.data.shape[-1])
        return persistence

def smart_persistence(input,clearsky,output_size=36,datatype="numpy"):
    """
    Day-ahead smart persistence. Assumed as input the last 24h of irradiance and extrapolates from there.
    Needs clearsky for target window though.
    """
    assert len(clearsky) == output_size, "clearsky values are assumed corresponding to output_size"
    assert len(input) == 24, "assuming last 24h of irradiance as input"
    step_number = output_size/len(input)
    step_size = 24
    if datatype == "numpy":
        csi = np.zeros((output_size))
    elif datatype == "torch":
        device = input.device
        csi = torch.zeros((output_size)).to(device)
    else:
        raise ValueError("datatype must be either numpy or torch")
    # for step in np.arange(step_number):
    #     if output_size - (step_size*(step+1)) < 0:
    #         stop = 36
    #     else:
    #         stop = 24
        # if sum([i==0 for i in clearsky[int(step*step_size):int(step_size*(step+1))] ])== 0:

        #csi[int(step*step_size):int(step_size*(step+1))] = input[:stop]/clearsky[int(step*step_size):int(step_size*(step+1))]
        # else:
            # csi[int(step*step_size):int(step_size*(step+1))] = 0 

    csi[:24] = input/(clearsky[:24]+0.00000001)
    csi[24:]= input[:12]/(clearsky[24:]+0.00000001)
    result = csi*clearsky
    return result


def generate_persistence_data(horizon_size=36,window_size=96):
    persistence_size = 24 # window size is just used to determine the start and end index
    # import sunpoint and cs 
    train,train_target,valid,valid_target,_,test_target= data_import() #dtype="float64"
    _loc_data = os.getcwd()
    cs_valid, cs_test, cs_train, day_mask,cs_de_norm = return_cs(os.path.join(_loc_data,"data"))
    # make time index
    start_date = "2016-01-01 00:30:00"
    end_date = "2020-12-31 23:30:00"    
    index = pd.date_range(start=start_date, end = end_date, freq = '1h', tz='CET')
    i_series = np.arange(0, len(index), 1)
    train_index = i_series[len(test_target)+len(valid_target):]
    valid_index = i_series[len(test_target):len(test_target)+len(valid_target)]
    overall_time = index.values
    
    ghi = np.concatenate([test_target, valid_target, train_target], axis=0)
    cs_ghi = np.concatenate([cs_test ,cs_valid , cs_train],axis = 0)
 

    # make persistence data as forecast from this current timestamp
    start = 0 #+ window_size
    end = len(overall_time) - (horizon_size + window_size)
    pers_list =list()
    for idx,stamp in enumerate(overall_time[start:end], start=start):
        pers = smart_persistence(ghi[idx+(window_size-persistence_size):idx+window_size],cs_ghi[idx+ window_size:idx+horizon_size+ window_size],output_size=horizon_size)
        pers_list.append({"idx": idx, "Time": stamp, "Value": pers})
    pers_df = pd.DataFrame(pers_list)
    pers_df.set_index("idx",inplace=True)

    # save to pickle
    pers_df.to_pickle("models/sunpoint_smart_persistence.pkl")
    return

if __name__ == "__main__":
    generate_persistence_data(horizon_size=36,window_size=96)
