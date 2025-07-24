import torch
import numpy as np
from hdbscan import HDBSCAN
from sklearn.cluster import DBSCAN

from typing import List, Sequence, Tuple


@torch.no_grad()
def cluster_event_by_similarity_per_cluster(
        emb:              torch.Tensor,   # (N,D)  float16/32, on GPU
        cluster_ids_in:   torch.Tensor,   # (N,)   int32/64,  on GPU/CPU
        num_points:       int,
        temperature:      float,
        min_cluster_size: int,
        give_remainder_own_id: bool
) -> torch.Tensor:
    """
    Same output dtype / semantics as the original routine,
    but *respecting* pre-computed pixel clusters.
    """
    device       = emb.device
    N            = emb.size(0)


    track_ids = torch.full((N,), -1, dtype=torch.int32, device=device)


    cluster_ids = cluster_ids_in.to(device, non_blocking=True)
    unique_cids = torch.unique(cluster_ids)
    unique_cids = unique_cids[unique_cids >= 0]       

    next_track_id = 0

    for cid in unique_cids:
        mask   = (cluster_ids == cid)
        idxs   = torch.nonzero(mask, as_tuple=False).squeeze(1)
        sub_emb = emb[idxs]


        sub_labels = _cluster_subset(sub_emb,
                                     num_points=num_points,
                                     temperature=temperature,
                                     min_cluster_size=min_cluster_size,
                                     give_remainder_own_id=give_remainder_own_id)

        # offset to remain unique across clusters
        valid = sub_labels >= 0
        sub_labels[valid] += next_track_id
        next_track_id += int(sub_labels.max().item()) + 1 if valid.any() else 0

        track_ids[idxs] = sub_labels

    return track_ids.cpu()          



def _cluster_subset(emb_subset: torch.Tensor,
                    *,
                    num_points: int,
                    temperature: float,
                    min_cluster_size: int,
                    give_remainder_own_id: bool) -> torch.Tensor:

    num_valid = emb_subset.size(0)
    cid_local = torch.full((num_valid,), -1, dtype=torch.int32, device=emb_subset.device)
    if num_valid < min_cluster_size:
        return cid_local                           # all noise

    #emb_subset = torch.nn.functional.normalize(emb_subset, dim=-1)
    emb_subset = torch.nn.functional.normalize(emb_subset.to(torch.float), dim=-1)
    sim        = emb_subset @ emb_subset.T / temperature
    diag_idx   = torch.arange(num_valid, device=emb_subset.device)
    sim[diag_idx, diag_idx] = -1.

    sim_max, sim_arg  = sim.max(dim=1)
    assigned = torch.zeros(num_valid, dtype=torch.bool, device=emb_subset.device)

    current = 0
    while True:
        remaining = ~assigned
        if not remaining.any():
            break
        anchor = torch.nonzero(remaining)[sim_max[remaining].argmax()].item()
        rem    = int(remaining.sum())
        if rem < min_cluster_size:
            if give_remainder_own_id:
                cid_local[remaining] = current
            break

        sims_anchor = sim[anchor]
        sims_anchor[assigned] = -1.
        k = min(num_points, rem)
        _, top_idx = sims_anchor.topk(k=k-1)
        members = torch.cat([torch.tensor([anchor], device=emb_subset.device), top_idx])

        cid_local[members] = current
        assigned[members] = True
        current          += 1


        bad_rows = (~assigned) & assigned[sim_arg]
        if bad_rows.any():
            rows     = torch.nonzero(bad_rows).squeeze(1)
            new_sim  = sim[rows][:, ~assigned]
            new_max, new_arg   = new_sim.max(dim=1)
            sim_max[rows]      = new_max
            abs_idx            = torch.nonzero(~assigned).squeeze(1)[new_arg]
            sim_arg[rows]      = abs_idx

    return cid_local


@torch.no_grad()
def _cluster_event_by_similarity(emb: torch.Tensor,
                                 num_points: int,
                                 temperature: float,
                                 min_cluster_size: int,
                                 give_remainder_own_id: bool
                                 ) -> torch.Tensor:
    """
    Parameters
    ----------
    emb : (N, D) **float32 or float16**, CUDA or CPU
    Returns
    -------
    cluster_ids : (N,) int32  (‑1 == noise)
    """
    device = emb.device
    N = emb.shape[0]

    # (1) remove obviously broken rows (NaN in any component)  → noise
    nan_mask      = torch.isnan(emb).any(dim=1)
    valid_mask    = ~nan_mask
    num_valid     = int(valid_mask.sum())
    cid_full      = torch.full((N,), -1, dtype=torch.int32, device=device)

    if num_valid < min_cluster_size:
        return cid_full                               # all noise

    emb_valid = emb[valid_mask]
    emb_valid = torch.nn.functional.normalize(emb_valid.to(torch.float), dim=-1)
    #emb_valid = torch.nn.functional.normalize(emb_valid, dim=-1)

    # (2) cosine‑similarity matrix once
    sim = emb_valid @ emb_valid.T / temperature
    idx = torch.arange(num_valid, device=device)
    sim[idx, idx] = -1.                              # prevent self‑match

    # (3) per‑row maxima and argmax
    sim_max, sim_arg = sim.max(dim=1)
    assigned   = torch.zeros(num_valid, dtype=torch.bool, device=device)
    cid_local  = torch.full((num_valid,), -1, dtype=torch.int32, device=device)

    current = 0
    while True:
        remaining_mask = ~assigned
        if not remaining_mask.any():
            break

        # pick anchor with highest still‑available similarity value
        anchor = torch.nonzero(remaining_mask)[sim_max[remaining_mask].argmax()].item()
        rem    = int(remaining_mask.sum())
        if rem < min_cluster_size:
            if give_remainder_own_id:
                cid_local[remaining_mask] = current
            break

        sims_anchor = sim[anchor]
        sims_anchor[assigned] = -1.
        k = min(num_points, rem)
        _, top_idx = sims_anchor.topk(k=k-1)
        members = torch.cat([torch.tensor([anchor], device=device), top_idx])

        cid_local[members] = current
        assigned[members] = True
        current += 1

        # update sim_max only where previous best got consumed
        bad = (~assigned) & assigned[sim_arg]
        if bad.any():
            rows     = torch.nonzero(bad).squeeze(1)
            new_sim  = sim[rows][:, ~assigned]
            new_max, new_arg = new_sim.max(dim=1)
            sim_max[rows]  = new_max
            abs_idx        = torch.nonzero(~assigned).squeeze(1)[new_arg]
            sim_arg[rows]  = abs_idx

    # (4) merge back into full‑event tensor and move to CPU
    cid_full[valid_mask] = cid_local
    return cid_full.cpu()               


def clustering_similarity(pred_embeds  : Sequence[torch.Tensor],
                          *,
                          cluster_ids_in: Sequence[torch.Tensor] = None,
                          num_points         : int      = 4,
                          temperature        : float    = 0.05,
                          min_cluster_size   : int      = 3,
                          give_remainder_own_id: bool   = True,
                          save_similarity_for_event: bool = False 
                          ) -> List[torch.Tensor]:
    """
    Parameters
    ----------
    pred_embeds : list/tuple of (N_i, D) tensors – **already on CUDA if possible**
    num_points  : desired hits per cluster (≈ detector layers)
    Returns
    -------
    cluster_labels : list of torch.int32 tensors, one per event (‑1 noise)
    """

    clusters = []
    similarity_matrix = None  # default is None unless explicitly saved
    for i, emb in enumerate(pred_embeds):
        if emb.numel() == 0:
            clusters.append(torch.empty((0,), dtype=torch.int32))
            continue
        
        sim_matrix = None

        if save_similarity_for_event and i == 0:
            sim_matrix = torch.matmul(emb, emb.T) / temperature
            similarity_matrix = sim_matrix.detach().cpu()

        
        if cluster_ids_in is not None:
            cluster_ids = cluster_ids_in[i].to(emb.device, non_blocking=True)
            # respect pre-computed  clusters
            cid = cluster_event_by_similarity_per_cluster(
                emb, cluster_ids_in=cluster_ids,
                num_points=num_points,
                temperature=temperature,
                min_cluster_size=min_cluster_size,
                give_remainder_own_id=give_remainder_own_id
            )
        else:
            cid = _cluster_event_by_similarity(
                    emb, num_points=num_points,
                    temperature=temperature,
                    min_cluster_size=min_cluster_size,
                    give_remainder_own_id=give_remainder_own_id
                )
        clusters.append(cid)

    return clusters, similarity_matrix

def clustering_HDBSCAN(pred_params, min_cl_size, min_samples):
    '''
    Function to perform DBSCAN clustering on the predicted track parameters, with specified
    DBSCAN hyperparameters. Returns the associated cluster IDs.
    '''
    clustering_algorithm = HDBSCAN(min_cluster_size=min_cl_size, min_samples=min_samples)
    cluster_labels = []
    
    for event_prediction in pred_params:
        regressed_params = np.array(event_prediction.tolist())

        # Explicitly remove NaNs before clustering
        valid_mask = ~np.isnan(regressed_params).any(axis=1)
        valid_params = regressed_params[valid_mask]

        # Check if sufficient valid data points remain after removing NaNs
        if len(valid_params) == 0:
            event_cluster_labels = -np.ones(len(regressed_params), dtype=int)
        else:
            event_cluster_labels = -np.ones(len(regressed_params), dtype=int)
            # Perform clustering only on valid data points
            valid_labels = clustering_algorithm.fit_predict(valid_params)
            event_cluster_labels[valid_mask] = valid_labels

        cluster_labels.append(torch.from_numpy(event_cluster_labels).int())

    return cluster_labels

def clustering(pred_params, epsilon, min_samples):
    '''
    Function to perform DBSCAN clustering on the predicted track parameters, with specified
    DBSCAN hyperparameters. Returns the associated cluster IDs.
    '''
    clustering_algorithm = DBSCAN(eps=epsilon, min_samples=min_samples)
    cluster_labels = []
    
    for event_prediction in pred_params:
        regressed_params = np.array(event_prediction.tolist())

        # Explicitly remove NaNs before clustering
        valid_mask = ~np.isnan(regressed_params).any(axis=1)
        valid_params = regressed_params[valid_mask]

        # Check if sufficient valid data points remain after removing NaNs
        if len(valid_params) == 0:
            event_cluster_labels = -np.ones(len(regressed_params), dtype=int)
        else:
            event_cluster_labels = -np.ones(len(regressed_params), dtype=int)
            # Perform clustering only on valid data points
            valid_labels = clustering_algorithm.fit_predict(valid_params)
            event_cluster_labels[valid_mask] = valid_labels

        cluster_labels.append(torch.from_numpy(event_cluster_labels).int())

    return cluster_labels




def clustering_inception(pred_params, existing_cluster_ids, epsilon, min_samples, cluster_noise=False):
    '''
    Performs clustering separately within each existing cluster.
    
    Parameters:
    - pred_params: List of tensors, each containing predictions for an event.
    - existing_cluster_ids: List of tensors, each containing existing cluster IDs for an event.
    - min_cl_size: Minimum cluster size for HDBSCAN.
    - min_samples: Minimum samples per cluster for HDBSCAN.
    - cluster_noise: If True, re-cluster original noise points (existing label = -1).
    
    Returns:
    - cluster_labels: List of tensors with new cluster labels per event.
    '''
    #clustering_algorithm = HDBSCAN(min_cluster_size=min_cl_size, min_samples=min_samples)
    clustering_algorithm = DBSCAN(eps=epsilon, min_samples=min_samples)
    cluster_labels = []

    for event_preds, event_existing_clusters in zip(pred_params, existing_cluster_ids):
        event_preds_np = event_preds.cpu().float().numpy()
        existing_clusters_np = event_existing_clusters.cpu().numpy()

        unique_existing_labels = np.unique(existing_clusters_np)
        new_event_labels = -np.ones_like(existing_clusters_np, dtype=int)
        next_label = 0

        for existing_label in unique_existing_labels:
            if existing_label == -1 and not cluster_noise:
                continue  # Skip clustering original noise points unless explicitly specified

            indices = np.where(existing_clusters_np == existing_label)[0]
            sub_params = event_preds_np[indices]

            # Remove rows containing NaNs
            valid_mask = ~np.isnan(sub_params).any(axis=1)
            clean_sub_params = sub_params[valid_mask]
            clean_indices = indices[valid_mask]

            # Check again if we have enough points for HDBSCAN after removing NaNs
            if len(clean_sub_params) < min_samples:
                continue  # Too few valid points remain after removing NaNs

            # Cluster only valid (non-NaN) data points
            sub_labels = clustering_algorithm.fit_predict(clean_sub_params)

            # Assign new labels back to original indices, excluding HDBSCAN-detected noise
            valid_sub_labels = np.unique(sub_labels[sub_labels != -1])
            for sub_label in valid_sub_labels:
                mask = (sub_labels == sub_label)
                new_event_labels[clean_indices[mask]] = next_label
                next_label += 1

        cluster_labels.append(torch.from_numpy(new_event_labels).int())

    return cluster_labels
