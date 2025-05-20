
""" 
Adapted from LarsBentsen  https://github.com/LarsBentsen/Paper5_ProbFor/blob/master/utils/metrics.py

"""
import numpy as np
import torch
import torch.distributions as distribution

from torch.nn import MSELoss, L1Loss

def RSE(pred, true): # why is this scaled by the difference in true to true mean - Some kind of Normalization
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))

@torch.no_grad()
def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)

@torch.no_grad()
def MAE(pred, true, return_mean=True, data_scaler=None, return_logits=False):
    if data_scaler is not None: 
        assert len(true.shape) == 2 and true.shape[-1] == 1
        assert len(pred.shape) == 2 and pred.shape[-1] == 1
        true = data_scaler(true)
        pred = data_scaler(pred)
    if torch.is_tensor(pred):
        _logits = (pred - true).abs()
    else:
        _logits = np.abs(pred - true)
    if return_logits:
        return [_logits]
    else:
        if return_mean:
            return torch.mean(_logits)
        else:
            if torch.is_tensor(_logits):
                return np.array([_logits.sum(), _logits.numel()])
            else: 
                return np.array([_logits.sum(), _logits.size])        

@torch.no_grad()
def MSE(pred, true, return_mean=True, data_scaler=None, return_logits=False):
    if data_scaler is not None: 
        assert len(true.shape) == 2 and true.shape[-1] == 1
        assert len(pred.shape) == 2 and pred.shape[-1] == 1
        true = data_scaler(true)
        pred = data_scaler(pred)
    _logits = (pred - true) ** 2
    if return_logits:
        return [_logits]
    else:
        if return_mean:
            return torch.mean(_logits)
        else:
            if torch.is_tensor(_logits):
                return np.array([_logits.sum(), _logits.numel()])
            else: 
                return np.array([_logits.sum(), _logits.size])

@torch.no_grad()
def RMSE(pred, true, return_mean=True):
    return torch.sqrt(MSE(pred, true, return_mean=return_mean))

@torch.no_grad()
def MAPE(pred, true, eps=1e-07, return_mean=True):
    _logits = torch.abs((pred - true) / (true + eps))
    if return_mean:
        return torch.mean(_logits)
    else:
        return _logits

@torch.no_grad()
def MSPE(pred, true, eps=1e-07, return_mean=True):
    _logits = torch.square((pred - true) / (true + eps))
    if return_mean:
        return torch.mean(_logits)
    else:
        return _logits


@torch.no_grad()
def PINAW(pred, truth, intervals=[0.2, 0.5, 0.9], quantiles=None, return_counts=True, johnson_flag=False, arbitrary_flag=False, data_scaler=None, return_logits=False,return_array=False):
    if len(truth.shape) == 3: 
        truth = truth.view(-1, truth.shape[-1])
        pred = pred.view(-1, pred.shape[-1])

    # range_samples = torch.max(truth)- torch.min(truth)
    # min_samples = torch.min(truth)
    # if quantiles is not None: 
    assert len(quantiles) % 2 == 1
    quantiles = torch.sort(quantiles)[0]
    intervals = [quantiles[-(i + 1)] - quantiles[i] for i in range(int((len(quantiles) - 1)/2))]
    _scores = {}
    _arrary_scores = []
    _items = []
    # if arbitrary_flag:
    #     assert len(pred.shape) == 2
    #     pred = pred.sort(0)[0]

    for i, interval_i in enumerate(intervals): 
        # if quantiles is not None:
            # quantile prediction. Assumes that the quantiles are in 
            # ascending order and correspond to the intervals
        ci_l = pred[..., i][..., None]
        ci_u = pred[..., -(i+1)][..., None]


        if return_logits:                
            _scores[np.round(interval_i, 5)] =  np.abs(ci_u - ci_l)
        else:        
            if torch.is_tensor(ci_l):
                avg_w = (ci_u - ci_l).abs().mean() 
            else: 
                avg_w = np.mean(np.abs(ci_u - ci_l))
            if return_counts: 
                # _scores[np.round(interval_i.item(), 5)] = np.array([avg_w, num_samples])   # Include the sum of the truth for normalising... 
                _scores[np.round(interval_i.item(), 5)] = torch.stack([avg_w, torch.tensor(1.0)]) # 1 instead of range_samples
            elif return_array:
                _arrary_scores.append(avg_w.item() / 1.0) # instead of range_samples.item())
                _items.append(np.round(interval_i.item(), 5))
            else: 
                _scores[np.round(interval_i.item(), 5)] = avg_w.item() / range_samples.item()
    if return_array:
        return _arrary_scores, _items
    return _scores

@torch.no_grad()
def PICP_quantile(pred, truth, intervals=None, quantiles=[0.05,0.125,0.25,0.375,0.45,0.5,0.55,0.625,0.75,0.875,0.95], return_counts=True, loss_type=None, data_scaler=None, return_logits=False, return_array=False):
    if len(truth.shape) == 3: 
        truth = truth.view(-1, truth.shape[-1])
        pred = pred.view(-1, pred.shape[-1])
    
    # if data_scaler is not None: 
    #     assert len(truth.shape) == 2 and truth.shape[-1] == 1
    #     truth = data_scaler(truth)
   
    # if intervals is not None:
    #     intervals = np.sort(intervals)
    #     quantiles = [0.5 - intervals[i]/2 for i in range((len(intervals) ))] + [0.5] 
    #     quantiles += [0.5 + intervals[i]/2 for i in range((len(intervals) ))]
    #     quantiles = np.sort(quantiles)
    # if loss_type=="Pinnball":
    #     assert pred.shape[-1] == len(quantiles)
    # else:
    #     assert pred.shape[-1] == 4
    #     norm_dist = distribution.Normal(pred[:, 0], pred[:, 1]) ### adjusted std here from idx 4 to 2
    _scores = {}
    _array_scores = []
    _items = []
    quantiles = torch.sort(quantiles)[0]
    for i, quantile_i in enumerate(quantiles): 
        # if loss_type=="Pinnball":
        ci = pred[..., i][..., None]

        
        if return_logits:                
            _scores[np.round(quantile_i.item(), 5)] = np.concatenate([(truth <= ci)], -1).all(-1).astype(int)[:, None]
            
        else:
            if torch.is_tensor(truth):
                count_correct = torch.cat([ (truth <= ci)], -1).sum() 
            else: 
                count_correct = np.concatenate([(truth <= ci)], -1).all(-1).sum()
            if return_counts: 
                # _scores[np.round(quantile_i.item(), 5)] = np.array([count_correct, truth.shape[0]]) 
                _scores[np.round(quantile_i.item(), 5)] = torch.stack([count_correct, torch.tensor(truth.shape[0]).float()])
            elif return_array:
                _array_scores.append((count_correct.item() / truth.shape[0]))
                _items.append(np.round(quantile_i.item(), 5))
            else: 
                _scores[np.round(quantile_i.item(), 5)] = count_correct.item() / truth.shape[0]
    if return_array:
        return _array_scores, _items
    return _scores

@torch.no_grad()
def PICP(pred, truth, intervals=[0.1,0.25, 0.5, 0.75, 0.9], quantiles=None, return_counts=True, loss_type = None, arbitrary_flag=False, data_scaler=None, return_logits=False, return_array=False):
    if len(truth.shape) == 3: 
        truth = truth.view(-1, truth.shape[-1])
        pred = pred.view(-1, pred.shape[-1])
    # if data_scaler is not None: 
    #     assert len(truth.shape) == 2 and truth.shape[-1] == 1
    #     truth = data_scaler(truth)
    if quantiles is not None: 
        assert len(quantiles) % 2 == 1
        quantiles = torch.sort(quantiles)[0]
        intervals = [quantiles[-(i + 1)] - quantiles[i] for i in range(int((len(quantiles) - 1)/2))]
    _scores = {}
    _arrary_scores = []
    _items = []
    for i, interval_i in enumerate(intervals): 
        # if quantiles is not None:
            # quantile prediction. Assumes that the quantiles are in 
            # ascending order and correspond to the intervals
        ci_l = pred[..., i][..., None]
        ci_u = pred[..., -(i+1)][..., None]

        # else: 
        #     #WHAT DOES THIS DO AND WHY WOULD IT DO IT TODO
        #     # Assumes normal distribution... 
        #     # for interval_i in interval_i:
        #     z_val = torch.erfinv(torch.tensor(interval_i)) * np.sqrt(2)
        #     mean_i = pred.mean(0)
        #     std_i = pred.std(0)
        #     ci_l = mean_i - std_i * z_val       #/ (pred.shape[0] ** 0.5)
        #     ci_u = mean_i + std_i * z_val

        # if data_scaler is not None: 
        #     assert len(ci_l.shape) == 2 and ci_l.shape[-1] == 1
        #     ci_l = data_scaler(ci_l)
        #     ci_u = data_scaler(ci_u)
        if return_logits:                
            _scores[np.round(interval_i, 5)] = np.concatenate([(truth >= ci_l), (truth <= ci_u)], -1).all(-1).astype(int)[:, None]
            
        else:
            
            count_correct = torch.cat([(truth >= ci_l), (truth <= ci_u)], -1).all(-1).sum().float() 

            if return_counts: 
                _scores[np.round(interval_i.item(), 5)] = torch.stack([count_correct, torch.tensor(truth.shape[0]).float()])
            elif return_array:
                _arrary_scores.append(count_correct.item() / truth.shape[0])
                _items.append(np.round(interval_i.item(), 5))
            else: 
                _scores[np.round(interval_i.item(), 5)] = count_correct.item() / truth.shape[0]
    
    if return_array:
        return _arrary_scores, _items
    return _scores

@torch.no_grad()
def ACE(picp):
    # ace = torch.mean(torch.stack([torch.abs(k - v) for k, v in picp.values()]))
    ace = torch.mean(torch.stack([torch.abs(label - (values[0]/values[1])) for label,values in picp.items()]))
    return ace

    
# @torch.no_grad()
# def crps_empirical(pred, truth):
#     """
#     Computes negative Continuous Ranked Probability Score CRPS* [1] between a
#     set of samples ``pred`` and true data ``truth``. This uses an ``n log(n)``
#     time algorithm to compute a quantity equal that would naively have
#     complexity quadratic in the number of samples ``n``::
#         CRPS* = E|pred - truth| - 1/2 E|pred - pred'|
#               = (pred - truth).abs().mean(0)
#               - (pred - pred.unsqueeze(1)).abs().mean([0, 1]) / 2
#     Note that for a single sample this reduces to absolute error.
#     **References**
#     [1] Tilmann Gneiting, Adrian E. Raftery (2007)
#         `Strictly Proper Scoring Rules, Prediction, and Estimation`
#         https://www.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf
#     :param torch.Tensor pred: A set of sample predictions batched on rightmost dim.
#         This should have shape ``(num_samples,) + truth.shape``.
#     :param torch.Tensor truth: A tensor of true observations.
#     :return: A tensor of shape ``truth.shape``.
#     :rtype: torch.Tensor
#     """
#     if pred.shape[1:] != (1,) * (pred.dim() - truth.dim() - 1) + truth.shape:
#         raise ValueError(
#             "Expected pred to have one extra sample dim on left. "
#             "Actual shapes: {} versus {}".format(pred.shape, truth.shape)
#         )
#     opts = dict(device=pred.device, dtype=pred.dtype)
#     num_samples = pred.size(0)
#     if num_samples == 1:
#         return (pred[0] - truth).abs()
    
#     # Sort the different samples for each prediction from smallest to largest
#     pred = pred.sort(dim=0).values
#     diff = pred[1:] - pred[:-1]
#     weight = torch.arange(1, num_samples, **opts) * torch.arange(
#         num_samples - 1, 0, -1, **opts
#     )
#     weight = weight.reshape(weight.shape + (1,) * (diff.dim() - 1))
#     # TODO: For analytical distributions we could maybe calculate exactly 
#     return (pred - truth).abs().mean(0) - (diff * weight).sum(0) / num_samples**2

# @torch.no_grad()
# def crps_analytic(pred, truth, quantiles, return_mean=True, return_logits=False): 
#     """
#     Assumes that the predictions are quantile predictions. If having predicted distributions, 
#     the respective quantiles should be computed before passing to this function. 
#     truth: [N, 1]
#     pred: [N, Q], where Q are the number of quantiles
#     quantiles: [Q,], i.e. the quantile probabilites, e.g. [0.05, 0.5, 0.95]
#     """
#     quantiles = torch.tensor(quantiles)
#     sorted_indices = torch.argsort(quantiles)
#     quantiles = quantiles[sorted_indices]
#     pred = pred[..., sorted_indices]

#     # Calculate the indicator function (truth <= pred)
#     indicator = (truth <= pred).float()
    
#     # Calculate the integrand
#     integrand = (quantiles - indicator) #** 2
    
#     # Integrate using trapezoid rule
#     crps = torch.trapezoid(integrand, pred, dim=-1)

#     #crps = np.trapz(np.array((torch.tensor(quantiles,device="cpu") - 1*(truth.clone().detach().cpu() <= pred.clone().detach().cpu())) ** 2), pred.clone().detach().cpu(), axis=-1)
#     #crps = torch.trapezoid((quantiles - 1*(truth <= pred))**2, pred, dim = -1)
#     #crps = torch.trapezoid( (pred- 1*(truth <= pred))**2, torch.tensor(quantiles), axis=-1)
#     if return_logits: 
#         return [crps]
#     else:
#         if return_mean:
#             #return np.nanmean(crps)
#             return crps.mean()
#         else: 
#             return np.array([np.nansum(crps), crps.size])

# @torch.no_grad()
# def eval_crps(pred, truth, data_scaler=None, analytic=False, quantiles=None, return_mean=True, return_logits=False, distribution=None):
#     # if data_scaler is not None:
#     #     assert len(truth.shape) == 2 and truth.shape[-1] == 1
#     #     if not analytic:
#     #         pred = torch.clip(pred, -1e3, 1e3)
#     #         pred = np.stack([data_scaler(pred_i) for pred_i in pred], 0)           
#     #     else:            
#     #         pred = np.stack([data_scaler(pred[..., i][..., None])[..., 0] for i in range(pred.shape[-1])], -1)
#     #     truth = data_scaler(truth)
#     #     pred = torch.from_numpy(pred)
#     #     truth = torch.from_numpy(truth)
#     """ For my own purposes, assuming that we put MLE outputs or quantile predictions into the pred tensor.
#     I.e. we use analytical only as then we can just generate quantiles from MLE and plug and play. 
#     Empirical is a bit opaque to me. Is the input a sample from a predicted distribution??  
#     """
#     if distribution == "Gaussian" or distribution == "JohnsonsSU" or distribution == "JohnsonsSB" or distribution == "Weibull":
#         dist = prob_dist(pred, distribution)
#         pred_quantiles = torch.stack([dist.icdf(q) for q in quantiles],axis = -1)
#         # neeed to inverse scale
#         pred_quantiles  = data_scaler.inverse_transform(pred_quantiles.detach().cpu()).to(truth.device)
#         #TODO debug test here that the truth is not scaled. otherwise need scaler here
#         return crps_analytic(pred_quantiles, truth, quantiles, return_mean=return_mean)
#     else:
#         return crps_analytic(pred, truth, quantiles, return_mean=return_mean)
    
#     if return_logits:
#         return crps_analytic(pred, truth, quantiles, return_logits=True)
#     else:
#         if not analytic:
#             return crps_empirical(pred, truth).mean().cpu().item()
#         else: 
#             return crps_analytic(pred, truth, quantiles, return_mean=return_mean)

# @torch.no_grad()
# def prob_metric(pred,truth,interval=[0.1,0.25,0.5,0.75,0.9], loss_type=None, reliability=False):
#     # blank for correct insertion #TODO needs setup
#     if loss_type=="Pinnball":
#         qr= [0.05,0.125,0.25,0.375,0.45,0.5,0.55,0.625,0.75,0.875,0.95]
#         picp = PICP(pred,truth,quantiles=qr,return_counts=False,loss_type=loss_type)
#         ace = ACE(picp) # needs to be before the rounding
#         picp = [np.round(item.numpy().item(),5) for interval,item in picp.items()]
#         #ace = ACE(pred,truth,quantiles=qr)
#     if reliability:
#         qr= [0.05,0.125,0.25,0.375,0.45,0.5,0.55,0.625,0.75,0.875,0.95]
#         r = PICP_quantile(pred,truth,quantiles=qr,return_counts=False,loss_type=loss_type)
#         r = [np.round(item.numpy().item(),5) for interval,item in r.items()]
#         return picp, ace, r
#     return picp, ace #, crps

# @torch.no_grad()
# def point_metric_for_prob(pred,truth,z,scaler=None,daytime=None,z_minus=None,loss_type=None):
#     # if z_minus ==None or z_minus == False:
#     #     z_minus = torch.zeros_like(z)
    
#     if loss_type=="Pinnball": # qr doesnt need scaler and daytime
#         middle_quantile = int(pred.shape[-1]/2)
#         pred = pred[:,:,middle_quantile].unsqueeze(-1)

#     return det_metric(pred,truth+ (z if z_minus else 0),daytime=daytime)
import warnings

def pinball_loss(pred, truth, quantiles):
    if not (len(pred.shape) == len(truth.shape) == len(quantiles.shape)):
        warnings.warn('All inputs should have the same number of dimensions')    
    return torch.mean(torch.max((truth - pred) * quantiles, (pred - truth) * (1 - quantiles)))

def diff_expected_observed_coverage(pinaw,intervals):
    #TODO this no longer works. Do we need it?
    raise NotImplementedError
    return np.mean(intervals-pinaw)

def approx_crps(pred, truth, quantiles):
    pinball_list = []
    for i in range(quantiles.size()[-1]):
        quantile = quantiles[..., i].unsqueeze(-1)
        pinball_list.append(pinball_loss(pred, truth, quantile).detach().cpu().numpy())
    return np.mean(pinball_list) #TODO check if this is correct 



from losses.qr_loss import SQR_loss

class Metrics():
    @torch.no_grad()
    def __init__(self,params,normalizer,data_source):
        self.metrics = params["metrics"]
        self.params = params
        self.lambda_ = params["loss_calibration_lambda"] 
        self.batchsize = params["batch_size"]
        self.horizon = params["horizon_size"]
        if params["input_model"] == "dnn":
            self.input_size = params["dnn_input_size"]
        elif params["input_model"] == "lstm":
            self.input_size = params["lstm_input_size"]
        else:
            raise ValueError("Metrics: Unknown input model type")  
        self.normalizer = normalizer 
        self.quantile_dim = params['metrics_quantile_dim'] 

        self.cs_multiplier = True if self.params["target"] == "CSI" else False
    @torch.no_grad()
    def __call__(self, pred, truth,quantile,cs, metric_dict,q_range,pers=None):
        results = metric_dict.copy()
        pred_denorm = self.normalizer.inverse_transform(pred,"target")
        truth_denorm = self.normalizer.inverse_transform(truth,"target")
        # we are approx crps with pinball so we do not need pinball loss
        # pred_denorm = pred_denorm *  cs if self.cs_multiplier else pred_denorm
        # truth_denorm = truth_denorm * cs if self.cs_multiplier else truth_denorm
        median = pred_denorm[...,int(self.quantile_dim/2)].unsqueeze(-1)
        
        if self.params["valid_clamp_output"]: # Clamping the output to be >= 0.0
            pred_denorm = torch.clamp(pred_denorm, min=0.0)
            median = torch.clamp(median, min=0.0)
        
        for metric in self.metrics:
            
            if metric == 'CS_L':
                sqr = SQR_loss(type='calibration_sharpness_loss', lambda_=self.lambda_)
                results['CS_L'].append(sqr(pred_denorm, truth_denorm, quantile).item()) 
            ### - Deterministic metrics
            elif metric == 'MAE':
                mae = L1Loss()
                results['MAE'].append(mae(median,truth_denorm).item())
            elif metric == 'MSE':
                results['MSE'].append(MSE(median, truth_denorm).item())
            elif metric == 'RMSE':
                rmse = MSELoss()
                results['RMSE'].append(torch.sqrt(rmse(median,truth_denorm)).item())
                #RMSE(median, truth)
            elif metric == 'MAPE':
                results['MAPE'].append(MAPE(median, truth_denorm).item())
            elif metric == 'MSPE':
                results['MSPE'].append(MSPE(median, truth_denorm).item()) 
            elif metric == 'RSE':
                results['RSE'].append(RSE(median, truth_denorm).item())
            elif metric == 'CORR':
                results['CORR'].append(CORR(median, truth_denorm).item()) 
            elif metric == 'SS':
                ss = Skill_score(truth = truth, y = pred[...,int(self.quantile_dim/2)].unsqueeze(-1), p = pers)
                # if ss.evaluate().item() < -10:
                #     print("SS:",ss.evaluate().item())
                results['SS'].append(ss.evaluate().item())

                # rmse_y = torch.sqrt(MSELoss()(median,truth_denorm))
                # rmse_p = torch.sqrt(MSELoss()(pers,truth_denorm))
                # results['SS'].append(Skill_score(rmse_y,rmse_p).item())
                # assert input is not None, "Input (X) is required for skill score's persistence model"
                # assert 'horizon' in options, "Horizon is required for skill score"
                # assert 'lookback' in options, "Lookback is required for skill score"
                # results['skill_score'] = Skill_score(options["horizon"],options["lookback"])(median, truth_denorm,input)
            elif metric == "SS_filt":
                ss = Skill_score(truth = truth,y = pred[...,int(self.quantile_dim/2)].unsqueeze(-1), p = pers)
                cut_off = self.params["horizon_size"] // 15
                results['SS_filt'].append(ss.evaluate_timestep_mean(exclude=cut_off).item())
            ###
            ### - Probabilistic metrics
            elif metric == 'ACE':
                picp = PICP(pred, truth,quantiles=q_range)
                results['ACE'].append((ACE(picp)).item())    #/(self.batchsize*self.horizon)
            elif metric == 'CRPS':
                results['CRPS'].append(approx_crps(pred_denorm, truth_denorm,quantiles=quantile).item())
            elif metric == 'COV': # for coverage if wanted/relevant TODO
                pass
            else:
                raise ValueError(f"Unknown metric: {metric}")
        return results
    
    def summarize_metrics(self, results,verbose=True,neptune=False,neptune_run=None):
        scheduler_metrics = dict()
        for metric, value in results.items():           
            if isinstance(value, (list, tuple, np.ndarray)):
                value_str = np.mean(np.array(value))
            else:
                value_str = value
            scheduler_metrics[metric] = value_str
            if metric == "Time":
                if verbose:
                    print(f"{value_str:.1f}s".ljust(8)+"-|", end=' ')
            elif metric == "Epoch":
                if verbose:
                    print("Epoch:" + value_str, end=' ')
            elif metric == "PICP" or metric == "PINAW":
                pass # we don't want to print this
            else:
                if verbose:
                    print((f"{metric}: {value_str:.4f}").ljust(15), end=' ')            
                if neptune:
                    neptune_run[f'valid/{metric}'].log(value_str)
        print(f" ")
        return scheduler_metrics
    # def approx_cdf(self,input,model):
    #     quantiles = [0.05,0.125,0.25,0.375,0.45,0.5,0.55,0.625,0.75,0.875,0.95]
    #     cdf = []
    #     Batch_norm = Batch_Normalizer(input)
    #     for q in quantiles:
    #         q_in = torch.tensor(q).repeat(input.shape[0],1).to(input.device)
    #         # input = Batch_norm.transform(input)
    #         pred = model(input,q_in,valid_run=True)
    #         # if self.input_size == 1:
    #         #     pred = Batch_norm.inverse_transform(pred)
    #         # else:
    #         #     pred = Batch_norm.inverse_transform(pred,pos=11)
    #         cdf.append(pred)
    #     cdf = torch.stack(cdf,dim=-1).squeeze(-2)
    #     cdf_denorm = self.normalizer.inverse_transform(cdf)
    #     return cdf,cdf_denorm, quantiles
    

  
# def Skill_score(rmse_y,rmse_p):
#     return 1 - (rmse_y/rmse_p)

class Skill_score():
    def __init__(self,truth,y,p):
        mse = MSELoss()
        self.epsilon = 1.0
        self.truth = truth
        self.y = y
        self.p = p
        self.rmse_y = torch.mean(((truth - y)**2)) + self.epsilon#torch.sqrt(mse(y,truth))
        self.rmse_p = torch.mean(((truth - p)**2)) + self.epsilon#torch.sqrt(mse(p,truth))
        # self.test_y = torch.mean(torch.sqrt(torch.mean(((truth - y)**2),dim=0)))
        # self.test_p = torch.sqrt(torch.mean(torch.mean(((truth - p)**2),dim=0)))
        self.rmse_timestep_y = torch.mean(((truth - y)**2),dim=0) +self.epsilon
        self.rmse_timestep_p = torch.mean(((truth - p)**2),dim=0) +self.epsilon
    def evaluate(self):
        # if 1 - (self.rmse_y/self.rmse_p) < -10.0:
        #     print("RMSE_y:",self.rmse_y)
        #     print("RMSE_p:",self.rmse_p)
        return 1 - (self.rmse_y/self.rmse_p)
        # return torch.mean(1- (self.rmse_timestep_y/self.rmse_timestep_p))
    
    def evaluate_timestep_mean(self,exclude=0):
        return torch.mean(1 - (self.rmse_timestep_y[exclude:]/self.rmse_timestep_p[exclude:]))

    def evaluate_timestep(self,exclude=0):
        
        return 1 - (self.rmse_timestep_y[exclude:]/self.rmse_timestep_p[exclude:])
# Ramp score is no good.
# class Ramp_score():
#     @torch.no_grad()
#     def __init__(self,res,LT,epsilon,delta_t,settings):
#         self.res = res # time resolution
#         self.LT = LT # Lead time - useless, because output will alrad ybe led in the future
#         self.epsilon = epsilon # threshold
#         self.settings = settings # settings for the model
#         self.delta_t = delta_t # tolerance window for the ramp event
#     @torch.no_grad()
#     def ramp_event(self, y_pred,y):
#         for i in range(y.shape[-1]):

#         ORE = torch.max(y[...,])
#         return 