import torch
device = torch.device(f'cuda:{torch.cuda.current_device()}') if torch.cuda.is_available() else 'cpu'
torch.set_default_device(device)
#TODO GPU performance is worse than CPU unless batch size is increased. Maybe need better data loading.
### Debug tool################
from debug.plot import Debug_model
##############################
from metrics.metric_plots import MetricPlots
from res.data import data_import, Data_Normalizer
from res.ife_data import import_ife_data 
from dataloader.calibratedDataset import CalibratedDataset

from losses.qr_loss import SQR_loss

from utils.helper_func import generate_surrogate_quantiles, return_features, return_Dataframe
from config import _LOG_NEPTUNE, _VERBOSE, params, _DATA_DESCRIPTION
from models.persistence import Persistence_Model
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def train():

    if _LOG_NEPTUNE:
        import neptune
        from neptune.utils import stringify_unsupported
        from api_key import _NEPTUNE_API_TOKEN
        run = neptune.init_run(
            project="n1kl4s/QuantileError",
            name = "smart_persistence",
            api_token=_NEPTUNE_API_TOKEN,
            tags= "smart_persistence"
        )
        run['data/type'] = _DATA_DESCRIPTION
    else:
        run = None


    # pytorch random seed
    torch.manual_seed(params['random_seed'])

    if _LOG_NEPTUNE:
        run['parameters'] = stringify_unsupported(params) # neptune only supports float and string

    if _DATA_DESCRIPTION ==  "Station 11 Irradiance Sunpoint":
        train,train_target,valid,valid_target,_,_ = data_import()
        if params['_INPUT_SIZE_LSTM'] == 1:
            train = train[:,11] # disable if training on all features/stations
        valid = valid[:,11]
    elif _DATA_DESCRIPTION == "IFE Skycam":
        train,train_target,valid,valid_target,cs_train, cs_valid, overall_time, train_index, valid_index= import_ife_data(params) # has 22 features now, 11 without preprocessing
        train,train_target,valid,valid_target, cs_train, cs_valid, overall_time= train.values,train_target.values,valid.values,valid_target.values, cs_train.values, cs_valid.values, overall_time.values
    else:
        raise ValueError("Data description not implemented")

    # # normalize train and valid
    Normalizer = Data_Normalizer(train,train_target,valid,valid_target)
    train,train_target,valid,valid_target = Normalizer.transform_all()
    # Training set
    quantiles = generate_surrogate_quantiles(len(train),params)
    y = train_target
    X = return_Dataframe(train)
    # Validation set
    # quantiles_valid = generate_surrogate_quantiles(len(valid),params)

    Xv = return_Dataframe(valid)
    #features = return_features(quantiles,params,data=train)
    data = CalibratedDataset(X, y,cs_train, train_index, device=device,params=params) 
    dataloader = torch.utils.data.DataLoader(data, batch_size=params['batch_size'], shuffle=params['train_shuffle'], generator=torch.Generator(device=device))

    data_valid = CalibratedDataset(Xv, valid_target,cs_valid,valid_index, device=device,params=params)
    data_loader_valid = torch.utils.data.DataLoader(data_valid, batch_size=params['batch_size'], shuffle=params['valid_shuffle'], generator=torch.Generator(device=device))

    
    metric_plots = MetricPlots(params,Normalizer,sample_size=params["valid_plots_sample_size"],log_neptune=_LOG_NEPTUNE)

    persistence = Persistence_Model(Normalizer,params)
    
    mses = []
    maes = []
    plot_ids = sorted(list(set([int(x * (512 / params["batch_size"])) for x in [7,24,14,22,37,8,3]])))
    for  idx,batch in enumerate(data_loader_valid):
        data, target, cs, time = batch
        pers_index = persistence.data.index[time[:,params["window_size"]-1].detach().cpu().numpy()] # -1 because it is an index
        out= torch.tensor(persistence.forecast(pers_index)).unsqueeze(-1)
        out_raw = torch.tensor(persistence.forecast_raw(pers_index)).unsqueeze(-1)
        target_raw = Normalizer.inverse_transform(target,"target")
        data_raw = Normalizer.inverse_transform(data.detach().cpu().numpy(),"train")
        mse = torch.sqrt(torch.nn.functional.mse_loss(out_raw,target_raw))
        mae = torch.nn.functional.l1_loss(out_raw,target_raw)
        mses.append(mse)
        maes.append(mae)
        if idx in plot_ids:
            target_max = Normalizer.max_target
            gen_plot(data_raw,target_raw,out_raw,overall_time[time.detach().cpu().numpy()],neptune_run=run,sample_num=idx, target_max=target_max)
    print(f"Mean MSE: {torch.mean(torch.tensor(mses))}")
    print(f"Mean MAE: {torch.mean(torch.tensor(maes))}")
    if _LOG_NEPTUNE:
        run["valid/MSE"] = torch.mean(torch.tensor(mses))
        run["valid/MAE"] = torch.mean(torch.tensor(maes))
   

    if _LOG_NEPTUNE:
        run.stop()

def gen_plot(data,target,out,time,neptune_run,sample_num,target_max):
    data = data[0]#.detach().cpu().numpy()
    target = target[0].detach().cpu().numpy()
    out = out[0].detach().cpu().numpy()
    time = time[0]
    x_idx = np.arange(0,len(data),1)
    y_idx = np.arange(len(data),len(data)+len(target),1)
    seaborn_style = "whitegrid"
    sns.set_theme(style=seaborn_style, palette="colorblind")
    
    plt.ioff()
    plt.figure(figsize=(10, 4))
    colors = sns.color_palette("colorblind")
    plt.plot(time[x_idx], data[:,0], label='Input Data', color=colors[0])
    plt.plot(time[y_idx], target[:,0], label='Ground Truth', color=colors[2])
    plt.plot(time[y_idx], out[:,0], label='Prediction', linestyle='--', color=colors[1])
    plt.xlabel('Time (DD HH:MM)')
    plt.ylabel('GHI (W/m^2)')
    plt.legend()
    plt.grid(True)
    plt.yticks(np.linspace(0, target_max, 10))
    # plt.xticks(time)
    plt.tight_layout()
    plt.savefig(f"{params['valid_plots_save_path']}/persistence_{sample_num}.png")
    plt_fig = plt.gcf()  # Get the current figure

    if neptune_run is not None:
            neptune_run[f"valid/persistence_plot_{sample_num}"].append(plt_fig)
    plt.close()


    return 0

class MetricPlots:
    def __init__(self, params, normalizer, sample_size=1, log_neptune=False,trial_num = 1):
        self.params = params
        self.sample_size = sample_size
        self.log_neptune = log_neptune
        self.save_path = params['valid_plots_save_path']
        self.metrics = params['array_metrics']
        self.normalizer = normalizer
        self.range_dict = {"PICP": None, "PINAW": None, "Cali_PICP": None}
        self.trial_num = trial_num
        seaborn_style = "whitegrid"
        sns.set_theme(style=seaborn_style, palette="colorblind")
    

   
    def generate_result_plots(self,data,pred,truth,quantile,cs,time,sample_num,neptune_run=None):
        """
        Plotting the prediction performance of the model.
        Saves the plots to save_path and logs them to neptune if needed.
        """
        sample_mid = data.shape[0] // 2
        sample_start =0 # or sample-mid 
        sample_idx = range(1)#np.arange(sample_start, sample_start + self.sample_size)
        data = data.detach().cpu().numpy()
        pred = pred.detach().cpu().numpy()
        truth = truth.detach().cpu().numpy()
        quantile = quantile.detach().cpu().numpy()

        data_denorm = self.normalizer.inverse_transform(data,"train")
        pred_denorm = self.normalizer.inverse_transform(pred,"target")
        truth_denorm = self.normalizer.inverse_transform(truth,"target")

        if cs is not None and self.params["target"] == "CSI":
            # cs = cs[:self.sample_size].detach().cpu().numpy()
            pred_denorm = pred_denorm * cs.detach().cpu().numpy()
            truth_denorm = truth_denorm * cs.detach().cpu().numpy()

        # Plotting
        target_max = self.normalizer.max_target
         
        for i in sample_idx:
            self._plot_results(data_denorm[i],pred_denorm[i],truth_denorm[i],quantile[i],time[i],target_max=target_max,sample_num=sample_num,neptune_run=neptune_run)

    
    def _plot_results(self,data,pred,truth,quantile,time,target_max,sample_num,neptune_run=None):
        x_idx = np.arange(0,len(data),1)
        y_idx = np.arange(len(data),len(data)+len(pred),1)
        pred_idx = int(quantile.shape[-1] / 2)
        
        plt.ioff()
        plt.figure(figsize=(10, 4))
        colors = sns.color_palette("colorblind")
        plt.plot(time[x_idx], data[:,0], label='Input Data', color=colors[0])
        plt.plot(time[y_idx], truth[:,0], label='Ground Truth', color=colors[2])
        plt.plot(time[y_idx], pred[:,pred_idx], label='Prediction', linestyle='--', color=colors[1])
        plt.fill_between(time[y_idx], pred[:, 0], pred[:, -1], alpha=0.2, label='Prediction Interval', color=colors[1])
        plt.xlabel('Time (DD HH:MM)')
        plt.ylabel('GHI (W/m^2)')
        plt.legend()
        plt.grid(True)
        plt.yticks(np.linspace(0, target_max, 10))
        # plt.xticks(time)
        plt.tight_layout()
        plt.savefig(f"{self.save_path}/timeseries_plot_{sample_num}.png")
        plt_fig = plt.gcf()  # Get the current figure

        if neptune_run is not None:
            neptune_run[f"valid/distribution_trial{self.trial_num}_{sample_num}"].append(plt_fig)
        plt.close()

if __name__ == "__main__":
    train()

