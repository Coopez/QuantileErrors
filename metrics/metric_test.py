
import torch

def calculate_crps(predictions, quantiles, truth):
    """
    Calculate Continuous Ranked Probability Score (CRPS)
    
    Args:
        predictions: Tensor of predicted values at each quantile
        quantiles: Tensor of quantile values (between 0 and 1)
        truth: Tensor of true values
        
    Returns:
        CRPS score (non-negative float)
    """
    # Ensure inputs are tensors
    predictions = torch.as_tensor(predictions)
    quantiles = torch.as_tensor(quantiles)
    truth = torch.as_tensor(truth)
    
    # Ensure quantiles are sorted
    sorted_indices = torch.argsort(quantiles)
    quantiles = quantiles[sorted_indices]
    predictions = predictions[..., sorted_indices]
    
    # Calculate the indicator function (truth <= pred)
    indicator = (truth.unsqueeze(-1) <= predictions).float()
    
    # Calculate the integrand
    integrand = (quantiles - indicator) ** 2
    
    # Integrate using trapezoid rule
    crps = torch.trapezoid(integrand, quantiles, dim=-1)
    
    return crps

# Test the implementation
def test_crps():
    # Test case 1: Perfect prediction
    quantiles = torch.linspace(0, 1, 100)
    truth = torch.tensor(5.0)
    perfect_pred = torch.full((100,), 5.0)
    perfect_score = calculate_crps(perfect_pred, quantiles, truth)
    print(f"Perfect prediction CRPS: {perfect_score.item():.6f}")
    
    # Test case 2: Biased prediction
    biased_pred = torch.full((100,), 6.0)
    biased_score = calculate_crps(biased_pred, quantiles, truth)
    print(f"Biased prediction CRPS: {biased_score.item():.6f}")
    
    # Test case 3: Uncertain prediction
    uncertain_pred = torch.linspace(4.0, 6.0, 100)
    uncertain_score = calculate_crps(uncertain_pred, quantiles, truth)
    print(f"Uncertain prediction CRPS: {uncertain_score.item():.6f}")

if __name__ == "__main__":
    test_crps()