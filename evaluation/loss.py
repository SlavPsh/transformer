import torch
import torch.nn as nn
import torch.nn.functional as F


def supcon_loss_flat(emb_flat,       # (N,D) normalised
                     pid_flat,       # (N,)   unique within event
                     ev_flat,        # (N,)   event_id
                     pt_flat,        # (N,)
                     pt_threshold=0.9,
                     temperature=0.05):
    """
    Contrastive InfoNCE for a *flattened* mini‑batch.
    Events are isolated by `ev_flat`.
    Hits with pt < pt_threshold are fully ignored.
    """
    keep = pt_flat >= pt_threshold
    emb  = emb_flat[keep]            # (N_keep,D)
    pid  = pid_flat[keep]
    ev   = ev_flat[keep]

    if emb.size(0) < 2:
        return emb.new_tensor(0.0)   

    # -------- similarity ------------------------------------
    sim = emb @ emb.T / temperature          
    diag = torch.eye(len(emb), device=emb.device).bool()
    sim.masked_fill_(diag, -9e15)

    # positives: same (event, particle)
    pos_mask = (pid[:, None] == pid[None, :]) & \
               (ev [:, None] == ev [None, :]) & ~diag

    # mask out hits from different events in the denominator
    same_event = ev[:, None] == ev[None, :]
    sim.masked_fill_(~same_event, -9e15)

    sim_max, _ = sim.max(dim=1, keepdim=True)
    log_prob = sim - sim_max.detach() - torch.log(
                 torch.exp(sim - sim_max.detach()).sum(dim=1, keepdim=True))

    pos_cnt = pos_mask.sum(dim=1)
    valid   = pos_cnt > 0
    if not valid.any():
        return emb.new_tensor(0.0)

    loss_i  = -(log_prob * pos_mask).sum(dim=1) / pos_cnt.clamp_min(1)
    return loss_i[valid].mean()


def mse_per_feature_loss(pred, target, feature_names):
    losses = {}
    for i, feature in enumerate(feature_names):
        loss_feature = F.mse_loss(pred[:, i], target[:, i])
        losses[f'loss_{feature}'] = loss_feature
    total_loss = torch.stack(list(losses.values())).mean()
    return total_loss, losses


def mse_per_feature_loss_weighted(
        pred,            # (N, F)  float32/float16
        target,          # (N, F)  same dtype; may contain NaNs
        feature_names,   # list[str]  len == F
        pt=None,         # (N,) optional, pt per hit
        *,
        pt_threshold=0.9,
        pt_weighting='none',   # 'none' | 'zero' | 'linear' | 'exp'
        exp_beta=5.0           # controls steepness for exp fall‑off
):
    """
    Weighted MSE per feature.
    ----------------------------------------------------------
    • If target == NaN   → weight 0 for that element.
    • If pt < threshold  → extra weight according to pt_weighting:
          'none'   : 1     (no change)
          'zero'   : 0     (ignore hit)
          'linear' : pt / pt_threshold   (0…1)
          'exp'    : exp(‑β · (Δpt)/pt_threshold)
    Returns
    -------
    total_loss  : scalar
    losses_dict : {f'loss_{name}': tensor}
    """

    if pt is None:
        pt = torch.ones(pred.size(0), device=pred.device, dtype=pred.dtype)

    # ------- pt weights --------------------------------------------------
    if pt_weighting == 'none':
        w_pt = torch.ones_like(pt)
    elif pt_weighting == 'zero':
        w_pt = (pt >= pt_threshold).float()
    elif pt_weighting == 'linear':
        w_pt = torch.where(pt >= pt_threshold,
                           torch.ones_like(pt),
                           pt / pt_threshold)
    elif pt_weighting == 'exp':
        w_pt = torch.where(pt >= pt_threshold,
                           torch.ones_like(pt),
                           torch.exp(-exp_beta * (pt_threshold - pt) / pt_threshold))
    else:
        raise ValueError("pt_weighting must be 'none'|'zero'|'linear'|'exp'")

    # broadcast to (N, F)
    w_pt = w_pt.unsqueeze(1).expand_as(pred)

    # ------- NaN mask ----------------------------------------------------
    nan_mask = ~torch.isnan(target)         # True where value is *valid*
    target_filled = torch.nan_to_num(target, nan=0.0)

    # squared error per entry
    sq_err = (pred - target_filled).pow(2)

    # total weight per entry
    w = nan_mask.float() * w_pt

    # ------- per‑feature losses -----------------------------------------
    losses = {}
    for i, name in enumerate(feature_names):
        num = (sq_err[:, i] * w[:, i]).sum()
        den = w[:, i].sum().clamp_min(1e-8)   # avoid /0
        losses[f'loss_{name}'] = num / den

    # scalar total (mean of feature losses)
    total_loss = torch.stack(list(losses.values())).mean()

    return total_loss, losses