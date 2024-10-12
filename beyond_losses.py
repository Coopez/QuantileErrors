import torch

"""
Loss functions for quantile regression adapted from:
https://github.com/YoungseogChung/calibrated-quantile-uq/tree/master
Beyond Pinball Loss: Quantile Methods for Calibrated Uncertainty Quantification

"""


def interval_loss(model, y, x, q, device, args):
    """
    implementation of interval score

    q: scalar
    """

    num_pts = y.size(0)

    with torch.no_grad():
        lower = torch.min(torch.stack([q, 1 - q], dim=0), dim=0)[0]
        upper = 1.0 - lower
        # l_list = torch.min(torch.stack([q_list, 1-q_list], dim=1), dim=1)[0]
        # u_list = 1.0 - l_list

    l_rep = lower.view(-1, 1).repeat(1, num_pts).view(-1, 1).to(device)
    u_rep = upper.view(-1, 1).repeat(1, num_pts).view(-1, 1).to(device)

    if x is None:
        model_in = torch.cat([lower, upper], dim=0)
    else:
        l_in = torch.cat([x, l_rep], dim=1)
        u_in = torch.cat([x, u_rep], dim=1)
        model_in = torch.cat([l_in, u_in], dim=0)

    pred_y = model(model_in)
    pred_l = pred_y[:num_pts].view(-1)
    pred_u = pred_y[num_pts:].view(-1)

    below_l = (pred_l - y.view(-1)).gt(0)
    above_u = (y.view(-1) - pred_u).gt(0)

    loss = (
        (pred_u - pred_l)
        + (1.0 / lower) * (pred_l - y.view(-1)) * below_l
        + (1.0 / lower) * (y.view(-1) - pred_u) * above_u
    )

    return torch.mean(loss)


def cali_loss(model, y, x, q, device, args):
    """
    original proposed loss function:
        when coverage is low, pulls from above
        when coverage is high, pulls from below

    q: scalar
    """
    num_pts = y.size(0)
    q = float(q)

    if x is None:
        q_tensor = torch.Tensor([q]).to(device)
        pred_y = model(q_tensor)
    else:
        q_tensor = q * torch.ones(num_pts).view(-1, 1).to(device)
        pred_y = model(torch.cat([x, q_tensor], dim=1))

    idx_under = y <= pred_y
    idx_over = ~idx_under
    coverage = torch.mean(idx_under.float()).item()

    if coverage < q:
        loss = torch.mean((y - pred_y)[idx_over])
    else:
        loss = torch.mean((pred_y - y)[idx_under])

    if hasattr(args, "scale") and args.scale:
        loss = torch.abs(q - coverage) * loss

    if hasattr(args, "sharp_penalty"):
        import pudb

        pudb.set_trace()
        assert isinstance(args.sharp_penalty, float)

        if x is None:
            opp_q_tensor = torch.Tensor([1 - q]).to(device)
            opp_pred_y = model(opp_q_tensor)
        else:
            opp_q_tensor = (1 - q) * torch.ones(num_pts).view(-1, 1).to(device)
            opp_pred_y = model(torch.cat([x, opp_q_tensor], dim=1))

        with torch.no_grad():
            below_med = q <= 0.5
            above_med = not below_med # changed from ~ below_med

        sharp_penalty = below_med * (opp_pred_y - pred_y) + above_med * (
            pred_y - opp_pred_y
        )

        if sharp_penalty <= 0.0:
            sharp_penalty = 0.0

        loss = (1 - args.sharp_penalty) * loss + (
            args.sharp_penalty * sharp_penalty
        )

    return loss