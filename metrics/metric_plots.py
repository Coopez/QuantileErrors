import torch
from .metrics import PICP, PINAW, PICP_quantile
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd




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
    def accumulate_array_metrics(self,metrics,pred,truth,quantile):
        
        picp, picp_interval = PICP(pred,truth,quantiles=quantile, return_counts=False,return_array=True)
        pinaw, pinaw_interval =PINAW(pred,truth,quantiles=quantile, return_counts=False,return_array=True)
        picp_c, picp_c_quantiles = PICP_quantile(pred,truth,quantiles=quantile, return_counts=False,return_array=True)   
        if self.range_dict["PICP"] is None:
            self.range_dict["PICP"] = picp_interval
            self.range_dict["PINAW"] = pinaw_interval
            self.range_dict["Cali_PICP"] = picp_c_quantiles
        
        metrics["PICP"].append(picp)
        metrics["PINAW"].append(pinaw)
        metrics["Cali_PICP"].append(picp_c)

        return metrics
    """
    rewrite the code to just take the very last batch of an epoch and calculate the metrics on that. 

    """

    def generate_metric_plots(self,metrics,neptune_run=None, dataloader_length=None):
        
        plot_data = self._summarize_array_metrics(metrics)
        for name, x_values in self.range_dict.items():
            self._plot_metric(name,plot_data[name],x_values,neptune_run=neptune_run)

    def _summarize_array_metrics(self,metrics):
        summary = self.metrics.copy()
        summary["PICP"] = np.mean(np.array(metrics["PICP"]),axis = 0)
        summary["PINAW"] = np.mean(np.array(metrics["PINAW"]),axis = 0)
        summary["Cali_PICP"] = np.mean(np.array(metrics["Cali_PICP"]),axis = 0)
        return summary

    
    def _plot_metric(self,name, value, ideal = None, neptune_run=None):
        if ideal is not None:
            sorted_indices = np.argsort(ideal)
            value = np.array(value)[sorted_indices]
            ideal = np.array(ideal)[sorted_indices]
            # value = np.insert(value, 0, 0)
            # ideal = np.insert(ideal, 0, 0)
        x = np.linspace(ideal[0], ideal[-1], len(value))
        x_label = "Quantiles" if name == "Cali_PICP" else "Intervals"
        plt.ioff()  # Turn off interactive mode
        plt.figure(figsize=(4, 3))
        plt.plot(x, value, label=name, linewidth=3)
        if name != "PINAW":           
            plt.plot(x, ideal, label="Ideal", linewidth=3)
            plt.yticks(np.linspace(0, 1, 5))
        plt.xlabel(x_label)  # should be label quantiles for calibration, intervals for picp and pinaw
        plt.ylabel(name)
        plt.legend()
        plt.grid(True)
        
        plt.xticks(np.linspace(0,1,5))
        plt.tight_layout()  # Adjust layout to prevent label cutoff
        plt.savefig(f"{self.save_path}/{name}_plot.png")
        plt_fig = plt.gcf()  # Get the current figure
        
        
        if neptune_run is not None:
            neptune_run[f"valid/distribution_{name}"].append(plt_fig)
        plt.close()

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