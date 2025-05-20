import torch
import pandas as pd
import numpy as np
class Persistence_Model():
    def __init__(self,normalizer,params,data_from="Station 11 Irradiance Sunpoint"):
        self.data = pd.read_pickle('models/persistence.pkl')
        # if data_from == "IFE Skycam":
        #     self.data = pd.read_pickle('models/persistence.pkl')
        # elif data_from == "Station 11 Irradiance Sunpoint":
        #     self.data = pd.read_pickle('models/sunpoint_persistence.pkl')
        self.index = self.data.index
        self.data = torch.tensor(self.data.values).float()
        self.min = normalizer.min_target
        self.max = normalizer.max_target
        self.data_normalized = (self.data-self.min)/(self.max-self.min)
        self.params = params

        self.target_summary = params["target_summary"]
        self.current_horizon = params["horizon_size"]
    
    def forecast_raw(self,time):
        persistence = self.data[time,:]
        # persistence = self._rolling_mean(input_tensor=persistence)[:,self.target_summary-1::self.target_summary]
        persistence = persistence.view(-1, self.target_summary).mean(dim=-1, keepdim=True).view(-1,self.current_horizon)
        return persistence
        # persistence = np.zeros((time.shape[0],self.params['horizon_size']))
        # for idx,instance in enumerate(time):
        #     temp_pers =self.data.loc[instance].rolling(window=self.target_summary, min_periods=1).mean()[self.target_summary-1::self.target_summary]
        #     persistence[idx] = temp_pers.values

        # return persistence#self.data.loc[time]
    
    def forecast(self,time):
        persistence = self.data_normalized[time,:]
        # persistence = self._rolling_mean(input_tensor=persistence)[:,self.target_summary-1::self.target_summary]
        persistence = persistence.view(-1, self.target_summary).mean(dim=-1, keepdim=True).view(-1,self.current_horizon)
        return persistence
    
    def evaluate(self,forecast,actual):
        mae =  np.mean(np.abs(forecast-actual))
        tau = 0.5
        fake_pinball = np.mean(np.maximum(tau*(actual-forecast), (tau-1)*(actual-forecast)))
        return mae,fake_pinball
    
  