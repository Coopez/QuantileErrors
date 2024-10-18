import torch
import warnings

def sqr_loss(y_pred,y_true,quantile,type='pinball_loss'):
    if not (len(y_pred.shape) == len(y_true.shape) == len(quantile.shape)):
        warnings.warn('All inputs should have the same number of dimensions')
    if type == 'pinball_loss':
        return torch.mean(torch.max(torch.mul(quantile,(y_true-y_pred)),torch.mul((quantile-1),(y_true-y_pred))))
    elif type == 'calibration_sharpness_loss':
        # Expects y_pred to be of size (batch_size,window_size,2) for quantile and 1-quantile
        assert y_pred.shape[2] == 2, 'y_pred should have quantile and 1-quantile'
        lambda_ = 0.5
        calibration = calibration_loss(y_pred[...,0].view(y_pred.shape[0],y_pred.shape[1],1),y_true,quantile)
        sharpness = sharpness_loss(y_pred,quantile)
        loss = lambda_*calibration + (1-lambda_)*sharpness
        return loss
    else:
        raise ValueError('Unknown loss type')
    return None

def calibration_loss(y_pred,y_true,quantile):
    pp = predicted_probability(y_pred,y_true)
    loss = torch.mean(
        identifier(quantile.view(quantile.shape[0],1),pp) * torch.mean((y_true-y_pred) * identifier_matrix(y_true,y_pred),dim=0) + 
        identifier(pp,quantile.view(quantile.shape[0],1)) * torch.mean((y_pred-y_true) * identifier_matrix(y_pred,y_true),dim=0) 
        ) #TODO check if dim is correct
    return loss


def identifier(x,y):
    """
    implements I(x<y)
    output is likely a vector of batchsize
    """
    return (x>y).type(torch.float32)

def identifier_matrix(x,y):
    """
    returns a matrix of identifiers x<y
    """
    assert torch.is_tensor(x) and torch.is_tensor(y), 'Both inputs should be tensors'
    return (x>y).type(torch.float32)

def predicted_probability(y_pred,y_true):
    """
    P(y_true<=y_pred) = E[I(y_true<=y_pred)]
    output has to be a vector of batch size
    """
    assert torch.is_tensor(y_pred) and torch.is_tensor(y_true), 'Both inputs should be tensors'
    y = y_true - y_pred
    return torch.mean((y<=0).type(torch.float32),dim=0) #TODO check if dim is correct


def sharpness_loss(y_pred,quantile):
    """
    sharpness component checks how far apart sister quantiles are and penalizes in turn
    """
    p = (quantile <= 0.5).type(torch.float32)
    # needs abs because of quantile crossover probability
    # not in original paper. !TODO: check if this is in code of  Beyond quantile loss paper
    # !TODO: if not can make reference to this in own paper
    return torch.mean((p*(y_pred[...,1]-y_pred[...,0]) + (1-p)*(y_pred[...,0]-y_pred[...,1]))**2) 
# Sharpness seems to only look at positive difference in quantiles. The formula could be simplified to an absolute difference anyways if we are to assume that there wont be quantile crossovers.
#  If there are, this loss form is not helpful either, as that would be awarded 0 loss. 


#!TODO: Why not use normal MSE or MAE as sharpness component? We only care how far apart quantiles are from the truth. We could even use something more sophisticated for the sharpness component.
# Question: Does passing the qunatiles as input vector feature make sense? If so, should 1-p also be passed as input?