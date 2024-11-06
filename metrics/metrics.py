
""" 
Adapted from LarsBentsen  https://github.com/LarsBentsen/Paper5_ProbFor/blob/master/utils/metrics.py

"""
import numpy as np
import torch
import torch.distributions as distribution



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
def det_metric(pred, true, daytime=None):
    if daytime is not None:
        pred = pred*daytime
        true = true*daytime
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe


class nnl_loss:
    @torch.no_grad()
    def __init__(self, dist_type='Gauss'):
        self.dist_type = dist_type
    @torch.no_grad()
    def __call__(self, pred, truth, return_mean=True, data_scaler=None, return_logits=False):
        # only implemented for univariate predictions so far... 

        if self.dist_type == 'Gauss': 
            # Assumes pred[..., 0] and pred[..., -1] contains loc and scale preds, respectively
            assert pred.shape[-1] == 2
            assert pred[..., :1].shape == truth.shape
            _logs = distribution.Normal(pred[..., :1], pred[..., -1:]).log_prob(truth)
        # elif self.dist_type == 'Johns':
        #     assert pred.shape[-1] == 4
        #     assert pred[..., 0].shape == truth[..., 0].shape
        #     _logs = JohnsonSUDistribution(pred[..., 0], pred[..., 1], pred[..., 2], pred[..., 3]).log_prob(truth[..., 0])
        #     _logs = torch.clip(_logs, -1e3, 1e3)
        #     # _logs.requires_grad = True

        # def pdf(x, xi, lamb, gamma, delta):
        #     # delta = scale
        #     # lambda = b
        #     # gamma = loc
        #     # xi = a
        #     return delta/(lamb * (2*torch.pi)**0.5) * 1/(torch.sqrt(1 + ((x - xi)/lamb)**2)) * torch.exp(-0.5 * (gamma + delta*torch.arcsinh(((x - xi)/lamb))) ** 2)

        else: 
            raise NotImplementedError
        if return_logits:
            return [_logs]
        else:
            if return_mean:
                nnl = - _logs.mean()
            else: 
                if torch.is_tensor(pred):
                    nnl = np.array([-_logs.sum(), _logs.numel()])
                else: 
                    nnl = np.array([-_logs.sum(), _logs.size])

            return nnl

def gradient_ascent(func, start, learn_rate, iter_max=100):
    assert len(start.shape) <= 2
    if len(start.shape) == 2: 
        assert start.shape[-1] == 1
        start = start[:, 0]
    best_guess = start
    n_iter = 0
    diff = torch.ones_like(start)
    while n_iter < iter_max:
        n_iter += 1
        # if (2 * abs(diff) * learn_rate) < 1e-100: 
        #     break
        mask = (torch.abs(diff)) < 1e-10
        if mask.all(): 
            break
        step = learn_rate * torch.abs(diff)
        diff[~mask] = -(func(best_guess + step) - func(best_guess - step))[~mask] / (2 * step[~mask])
        best_guess[~mask] = (best_guess - learn_rate * diff)[~mask]
    return best_guess


@torch.no_grad()
def PINAW(pred, truth, intervals=[0.2, 0.5, 0.9], quantiles=None, return_counts=True, johnson_flag=False, arbitrary_flag=False, data_scaler=None, return_logits=False):
    if len(truth.shape) == 3: 
        truth = truth.view(-1, truth.shape[-1])
        pred = pred.view(-1, pred.shape[-1])

    num_samples = truth.shape[0]

    if quantiles is not None: 
        assert len(quantiles) % 2 == 1
        quantiles = np.sort(quantiles)
        intervals = [quantiles[-(i + 1)] - quantiles[i] for i in range(int((len(quantiles) - 1)/2))]
    _scores = {}

    if arbitrary_flag:
        assert len(pred.shape) == 2
        pred = pred.sort(0)[0]

    for i, interval_i in enumerate(intervals): 
        if quantiles is not None:
            # quantile prediction. Assumes that the quantiles are in 
            # ascending order and correspond to the intervals
            ci_l = pred[..., i][..., None]
            ci_u = pred[..., -(i+1)][..., None]

        elif arbitrary_flag:
            int_l = int((0.5 - interval_i / 2) * pred.shape[0])
            int_u = int((0.5 + interval_i / 2) * pred.shape[0])
            ci_l = pred[int_l][:, None]
            ci_u = pred[int_u][:, None]
        else: 
            # Assumes normal distribution... 
            # for interval_i in interval_i:
            z_val = torch.erfinv(torch.tensor(interval_i)) * np.sqrt(2)
            mean_i = pred.mean(0)
            std_i = pred.std(0)
            ci_l = mean_i - std_i * z_val       #/ (pred.shape[0] ** 0.5)
            ci_u = mean_i + std_i * z_val

        if data_scaler is not None: 
            assert len(ci_l.shape) == 2 and ci_l.shape[-1] == 1
            ci_l = data_scaler(ci_l)
            ci_u = data_scaler(ci_u)

        if return_logits:                
            _scores[np.round(interval_i, 5)] =  np.abs(ci_u - ci_l)
        else:        
            if torch.is_tensor(ci_l):
                avg_w = (ci_u - ci_l).abs().sum() 
            else: 
                avg_w = np.sum(np.abs(ci_u - ci_l))
            if return_counts: 
                _scores[np.round(interval_i, 5)] = np.array([avg_w, num_samples])   # Include the sum of the truth for normalising... 
            else: 
                _scores[np.round(interval_i, 5)] = avg_w / num_samples 

    return _scores

@torch.no_grad()
def PICP_quantile(pred, truth, intervals=None, quantiles=[0.05,0.125,0.25,0.375,0.45,0.5,0.55,0.625,0.75,0.875,0.95], return_counts=True, loss_type=None, data_scaler=None, return_logits=False):
    if len(truth.shape) == 3: 
        truth = truth.view(-1, truth.shape[-1])
        pred = pred.view(-1, pred.shape[-1])
    
    if data_scaler is not None: 
        assert len(truth.shape) == 2 and truth.shape[-1] == 1
        truth = data_scaler(truth)
   
    if intervals is not None:
        intervals = np.sort(intervals)
        quantiles = [0.5 - intervals[i]/2 for i in range((len(intervals) ))] + [0.5] 
        quantiles += [0.5 + intervals[i]/2 for i in range((len(intervals) ))]
        quantiles = np.sort(quantiles)
    if loss_type=="Pinnball":
        assert pred.shape[-1] == len(quantiles)
    else:
        assert pred.shape[-1] == 4
        norm_dist = distribution.Normal(pred[:, 0], pred[:, 1]) ### adjusted std here from idx 4 to 2
    _scores = {}
    for i, quantile_i in enumerate(quantiles): 
        if loss_type=="Pinnball":
            ci = pred[..., i][..., None]
        else:
            raise NotImplementedError
        
        if return_logits:                
            _scores[np.round(quantile_i, 5)] = np.concatenate([(truth <= ci)], -1).all(-1).astype(int)[:, None]
            
        else:
            if torch.is_tensor(truth):
                count_correct = torch.cat([ (truth <= ci)], -1).all(-1).sum() 
            else: 
                count_correct = np.concatenate([(truth <= ci)], -1).all(-1).sum()
            if return_counts: 
                _scores[np.round(quantile_i, 5)] = np.array([count_correct, truth.shape[0]])
            else: 
                _scores[np.round(quantile_i, 5)] = count_correct / truth.shape[0]

    return _scores

@torch.no_grad()
def PICP(pred, truth, intervals=[0.1,0.25, 0.5, 0.75, 0.9], quantiles=None, return_counts=True, loss_type = None, arbitrary_flag=False, data_scaler=None, return_logits=False):
    if len(truth.shape) == 3: 
        truth = truth.view(-1, truth.shape[-1])
        pred = pred.view(-1, pred.shape[-1])
    if data_scaler is not None: 
        assert len(truth.shape) == 2 and truth.shape[-1] == 1
        truth = data_scaler(truth)
    if quantiles is not None: 
        assert len(quantiles) % 2 == 1
        quantiles = np.sort(quantiles)
        intervals = [quantiles[-(i + 1)] - quantiles[i] for i in range(int((len(quantiles) - 1)/2))]
    _scores = {}
    
    for i, interval_i in enumerate(intervals): 
        if quantiles is not None:
            # quantile prediction. Assumes that the quantiles are in 
            # ascending order and correspond to the intervals
            ci_l = pred[..., i][..., None]
            ci_u = pred[..., -(i+1)][..., None]

        else: 
            #WHAT DOES THIS DO AND WHY WOULD IT DO IT TODO
            # Assumes normal distribution... 
            # for interval_i in interval_i:
            z_val = torch.erfinv(torch.tensor(interval_i)) * np.sqrt(2)
            mean_i = pred.mean(0)
            std_i = pred.std(0)
            ci_l = mean_i - std_i * z_val       #/ (pred.shape[0] ** 0.5)
            ci_u = mean_i + std_i * z_val

        if data_scaler is not None: 
            assert len(ci_l.shape) == 2 and ci_l.shape[-1] == 1
            ci_l = data_scaler(ci_l)
            ci_u = data_scaler(ci_u)
        if return_logits:                
            _scores[np.round(interval_i, 5)] = np.concatenate([(truth >= ci_l), (truth <= ci_u)], -1).all(-1).astype(int)[:, None]
            
        else:
            if torch.is_tensor(truth):
                count_correct = torch.cat([(truth >= ci_l), (truth <= ci_u)], -1).all(-1).sum() 
            else: 
                count_correct = np.concatenate([(truth >= ci_l), (truth <= ci_u)], -1).all(-1).sum()
            if return_counts: 
                _scores[np.round(interval_i, 5)] = np.array([count_correct, truth.shape[0]])
            else: 
                _scores[np.round(interval_i, 5)] = count_correct / truth.shape[0]

    return _scores

@torch.no_grad()
def ACE(picp):
    ace = np.mean([np.abs(k - v) for k, v in picp.items()])
    return np.round(ace,5)

@torch.no_grad()
def sharpness(pred, intervals=[0.2, 0.5, 0.9], quantiles=None, return_counts=True):
    raise NotImplementedError
    # Returns the widhts of the different quantiles.
    # If return_counts, then it will sum the different widths and return this along with 
    # the number of data points to perform averaging later. Otherwise it will average over the pred
    if quantiles is not None: 
        assert len(quantiles) % 2 == 1
        quantiles = np.sort(quantiles)
        intervals = [quantiles[-(i + 1)] - quantiles[i] for i in range((len(quantiles) - 1)/2)]
    
    _scores = {}
    for i, interval_i in enumerate(intervals): 
        if quantiles is not None:
            # quantile prediction. Assumes that the quantiles are in 
            # ascending order and correspond to the intervals
            ci_l = pred[:, i][:, None]
            ci_u = pred[:, -(i+1)][:, None]

        elif pred.shape[-1] == 2 and len(pred.shape) == 2: 
            # Assumes normal distribution... 
                z_val = torch.erfinv(torch.tensor(interval_i)) * np.sqrt(2)
                ci_l = pred[:, :1] - pred[:, -1:] * z_val
                ci_u = pred[:, :1] + pred[:, -1:] * z_val

        else: 
            # Assumes normal distribution... 
            for interval_i in interval_i:
                z_val = torch.erfinv(torch.tensor(interval_i)) * np.sqrt(2)
                mean_i = pred.mean(0)
                std_i = pred.std(0)
                ci_l = mean_i - std_i * z_val       #/ (pred.shape[0] ** 0.5)
                ci_u = mean_i + std_i * z_val
        
        # We just take the absolute values to not give better scores for QR which might wrongly place quantiles. 
        # This is still not ideal and we could instead just not consider such non-senible scenarious... 
        if torch.is_tensor(pred):
            _widths = torch.abs(ci_u - ci_l).sum()
        else: 
            _widths = np.abs(ci_u - ci_l).sum()

        if return_counts: 
            _scores[interval_i] = np.array([_widths, ci_l.shape[0]])
        else: 
            _scores[interval_i] = _widths / ci_l.shape[0]

    return _scores
    
@torch.no_grad()
def crps_empirical(pred, truth):
    """
    Computes negative Continuous Ranked Probability Score CRPS* [1] between a
    set of samples ``pred`` and true data ``truth``. This uses an ``n log(n)``
    time algorithm to compute a quantity equal that would naively have
    complexity quadratic in the number of samples ``n``::
        CRPS* = E|pred - truth| - 1/2 E|pred - pred'|
              = (pred - truth).abs().mean(0)
              - (pred - pred.unsqueeze(1)).abs().mean([0, 1]) / 2
    Note that for a single sample this reduces to absolute error.
    **References**
    [1] Tilmann Gneiting, Adrian E. Raftery (2007)
        `Strictly Proper Scoring Rules, Prediction, and Estimation`
        https://www.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf
    :param torch.Tensor pred: A set of sample predictions batched on rightmost dim.
        This should have shape ``(num_samples,) + truth.shape``.
    :param torch.Tensor truth: A tensor of true observations.
    :return: A tensor of shape ``truth.shape``.
    :rtype: torch.Tensor
    """
    if pred.shape[1:] != (1,) * (pred.dim() - truth.dim() - 1) + truth.shape:
        raise ValueError(
            "Expected pred to have one extra sample dim on left. "
            "Actual shapes: {} versus {}".format(pred.shape, truth.shape)
        )
    opts = dict(device=pred.device, dtype=pred.dtype)
    num_samples = pred.size(0)
    if num_samples == 1:
        return (pred[0] - truth).abs()
    
    # Sort the different samples for each prediction from smallest to largest
    pred = pred.sort(dim=0).values
    diff = pred[1:] - pred[:-1]
    weight = torch.arange(1, num_samples, **opts) * torch.arange(
        num_samples - 1, 0, -1, **opts
    )
    weight = weight.reshape(weight.shape + (1,) * (diff.dim() - 1))
    # TODO: For analytical distributions we could maybe calculate exactly 
    return (pred - truth).abs().mean(0) - (diff * weight).sum(0) / num_samples**2

@torch.no_grad()
def crps_analytic(pred, truth, quantiles, return_mean=True, return_logits=False): 
    """
    Assumes that the predictions are quantile predictions. If having predicted distributions, 
    the respective quantiles should be computed before passing to this function. 
    truth: [N, 1]
    pred: [N, Q], where Q are the number of quantiles
    quantiles: [Q,], i.e. the quantile probabilites, e.g. [0.05, 0.5, 0.95]
    """
    
    crps = np.trapz(np.array((torch.tensor(quantiles) - 1*(truth.clone().detach().cpu() <= pred.clone().detach().cpu())) ** 2), pred.clone().detach().cpu(), axis=-1)
    if return_logits: 
        return [crps]
    else:
        if return_mean:
            return np.nanmean(crps)
        else: 
            return np.array([np.nansum(crps), crps.size])

@torch.no_grad()
def eval_crps(pred, truth, data_scaler=None, analytic=False, quantiles=None, return_mean=True, return_logits=False, distribution=None):
    # if data_scaler is not None:
    #     assert len(truth.shape) == 2 and truth.shape[-1] == 1
    #     if not analytic:
    #         pred = torch.clip(pred, -1e3, 1e3)
    #         pred = np.stack([data_scaler(pred_i) for pred_i in pred], 0)           
    #     else:            
    #         pred = np.stack([data_scaler(pred[..., i][..., None])[..., 0] for i in range(pred.shape[-1])], -1)
    #     truth = data_scaler(truth)
    #     pred = torch.from_numpy(pred)
    #     truth = torch.from_numpy(truth)
    """ For my own purposes, assuming that we put MLE outputs or quantile predictions into the pred tensor.
    I.e. we use analytical only as then we can just generate quantiles from MLE and plug and play. 
    Empirical is a bit opaque to me. Is the input a sample from a predicted distribution??  
    """
    if distribution == "Gaussian" or distribution == "JohnsonsSU" or distribution == "JohnsonsSB" or distribution == "Weibull":
        dist = prob_dist(pred, distribution)
        pred_quantiles = torch.stack([dist.icdf(q) for q in quantiles],axis = -1)
        # neeed to inverse scale
        pred_quantiles  = data_scaler.inverse_transform(pred_quantiles.detach().cpu()).to(truth.device)
        #TODO debug test here that the truth is not scaled. otherwise need scaler here
        return crps_analytic(pred_quantiles, truth, quantiles, return_mean=return_mean)
    else:
        return crps_analytic(pred, truth, quantiles, return_mean=return_mean)
    
    if return_logits:
        return crps_analytic(pred, truth, quantiles, return_logits=True)
    else:
        if not analytic:
            return crps_empirical(pred, truth).mean().cpu().item()
        else: 
            return crps_analytic(pred, truth, quantiles, return_mean=return_mean)

@torch.no_grad()
def prob_metric(pred,truth,interval=[0.1,0.25,0.5,0.75,0.9], loss_type=None, reliability=False):
    # blank for correct insertion #TODO needs setup
    if loss_type=="Pinnball":
        qr= [0.05,0.125,0.25,0.375,0.45,0.5,0.55,0.625,0.75,0.875,0.95]
        picp = PICP(pred,truth,quantiles=qr,return_counts=False,loss_type=loss_type)
        ace = ACE(picp) # needs to be before the rounding
        picp = [np.round(item.numpy().item(),5) for interval,item in picp.items()]
        #ace = ACE(pred,truth,quantiles=qr)
    if reliability:
        qr= [0.05,0.125,0.25,0.375,0.45,0.5,0.55,0.625,0.75,0.875,0.95]
        r = PICP_quantile(pred,truth,quantiles=qr,return_counts=False,loss_type=loss_type)
        r = [np.round(item.numpy().item(),5) for interval,item in r.items()]
        return picp, ace, r
    return picp, ace #, crps

@torch.no_grad()
def point_metric_for_prob(pred,truth,z,scaler=None,daytime=None,z_minus=None,loss_type=None):
    # if z_minus ==None or z_minus == False:
    #     z_minus = torch.zeros_like(z)
    
    if loss_type=="Pinnball": # qr doesnt need scaler and daytime
        middle_quantile = int(pred.shape[-1]/2)
        pred = pred[:,:,middle_quantile].unsqueeze(-1)

    return det_metric(pred,truth+ (z if z_minus else 0),daytime=daytime)



class Metrics():
    @torch.no_grad()
    def __init__(self, metrics):
        self.metrics = metrics
    @torch.no_grad()
    def __call__(self, pred, truth,input=None, options={}):
        results = {}
        for metric in self.metrics:
            if metric == 'MAE':
                results['MAE'] = MAE(pred, truth, options)
            elif metric == 'MSE':
                results['MSE'] = MSE(pred, truth, options)
            elif metric == 'RMSE':
                results['RMSE'] = RMSE(pred, truth, options)
            elif metric == 'MAPE':
                results['MAPE'] = MAPE(pred, truth, options)
            elif metric == 'MSPE':
                results['MSPE'] = MSPE(pred, truth, options)
            elif metric == 'RSE':
                results['RSE'] = RSE(pred, truth)
            elif metric == 'CORR':
                results['CORR'] = CORR(pred, truth)
            elif metric == 'PINAW':
                results['PINAW'] = PINAW(pred, truth, options)
            elif metric == 'PICP':
                results['PICP'] = PICP(pred, truth, options)
            elif metric == 'ACE':
                picp = PICP(pred, truth, options)
                results['ACE'] = ACE(picp)
            elif metric == 'CRPS':
                results['CRPS'] = eval_crps(pred, truth, options)
            elif metric == 'prob_metric':
                results['prob_metric'] = prob_metric(pred, truth, options)
            elif metric == 'point_metric_for_prob':
                results['point_metric_for_prob'] = point_metric_for_prob(pred, truth, options)
            elif metric == 'nnl_loss':
                results['nnl_loss'] = nnl_loss(options)(pred, truth, options)
            elif metric == 'skill_score':
                assert input is not None, "Input (X) is required for skill score's persistence model"
                assert 'horizon' in options, "Horizon is required for skill score"
                assert 'lookback' in options, "Lookback is required for skill score"
                results['skill_score'] = Skill_score(options["horizon"],options["lookback"])(pred, truth,input)
            else:
                raise ValueError(f"Unknown metric: {metric}")
        return results
    def print_metrics(self, results):
        for metric, value in results.items():
            if isinstance(value, (list, tuple, np.ndarray)):
                value_str = np.mean(value)
            else:
                value_str = value
            if metric == "Epoch":
                print(f"{metric}-{value_str}".ljust(8)+"-|", end=' ')
            else:
                print((f"{metric}: {value_str:.4f}").ljust(15), end=' ')
        print(f" ")

def persistence(x):
    return x[...,-1,0]

class Skill_score():
    @torch.no_grad()
    def __init__(self,horizon,lookback):
        self.horizon = horizon
        self.lookback = lookback
    @torch.no_grad()
    def __call__(self, y_pred,y,x):
        pers = persistence(x)

        return 1 - (torch.sum((y - y_pred)**2)/torch.sum((y - pers.unsqueeze(-1))**2))
        


# Ramp score is no good.
# class Ramp_score():
#     @torch.no_grad()
#     def __init__(self,res,LT,epsilon,delta_t,settings):
#         self.res = res # time resolution
#         self.LT = LT # Lead time - useless, becasue output will alrad ybe led in the future
#         self.epsilon = epsilon # threshold
#         self.settings = settings # settings for the model
#         self.delta_t = delta_t # tolerance window for the ramp event
#     @torch.no_grad()
#     def ramp_event(self, y_pred,y):
#         for i in range(y.shape[-1]):

#         ORE = torch.max(y[...,])
#         return 