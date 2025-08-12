import torch
from typing import List, Sequence, Tuple
from concurrent.futures import ThreadPoolExecutor

@torch.no_grad()
def parallel_cluster_event_by_similarity(
        emb: torch.Tensor,
        cluster_ids_in: torch.Tensor,
        num_points: int,
        temperature: float,
        min_cluster_size: int,
        give_remainder_own_id: bool,
        num_workers: int = 64
) -> torch.Tensor:

    device = emb.device
    N = emb.size(0)
    track_ids = torch.full((N,), -1, dtype=torch.int32, device=device)

    cluster_ids = cluster_ids_in.to(device, non_blocking=True)
    unique_cids = torch.unique(cluster_ids)
    unique_cids = unique_cids[unique_cids >= 0]

    emb = torch.nn.functional.normalize(emb.float(), dim=-1)

    next_track_id = torch.tensor([0], dtype=torch.int32)

    def process_single_cluster(cid):
        mask = cluster_ids == cid
        idxs = mask.nonzero(as_tuple=True)[0]
        sub_emb = emb[idxs]

        sub_labels = _cluster_subset(
            sub_emb,
            num_points=num_points,
            temperature=temperature,
            min_cluster_size=min_cluster_size,
            give_remainder_own_id=give_remainder_own_id
        )

        valid = sub_labels >= 0
        if valid.any():
            local_max = sub_labels[valid].max().item() + 1
            sub_labels[valid] += next_track_id.item()
            next_track_id.add_(local_max)

        return idxs, sub_labels

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_single_cluster, cid) for cid in unique_cids]

        for future in futures:
            idxs, sub_labels = future.result()
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

import threading
from concurrent.futures import ThreadPoolExecutor

@torch.no_grad()
def clustering_similarity(pred_embeds: Sequence[torch.Tensor],
                          *,
                          cluster_ids_in: Sequence[torch.Tensor] = None,
                          num_points: int = 4,
                          temperature: float = 0.05,
                          min_cluster_size: int = 3,
                          give_remainder_own_id: bool = True,
                          save_similarity_for_event: bool = False,
                          num_workers: int = 8
                          ) -> Tuple[List[torch.Tensor], torch.Tensor]:

    clusters = []
    similarity_matrix = None
    track_id_lock = threading.Lock()  # Protects next_track_id increment

    for i, emb in enumerate(pred_embeds):
        if emb.numel() == 0:
            clusters.append(torch.empty((0,), dtype=torch.int32))
            continue

        if save_similarity_for_event and i == 0:
            normalized_emb = torch.nn.functional.normalize(emb.float(), dim=-1)
            similarity_matrix = (normalized_emb @ normalized_emb.T).div(temperature).cpu()

        if cluster_ids_in is not None:
            cluster_ids = cluster_ids_in[i].to(emb.device, non_blocking=True)
        else:
            cluster_ids = torch.zeros(emb.size(0), dtype=torch.int64, device=emb.device)

        device = emb.device
        N = emb.size(0)
        track_ids = torch.full((N,), -1, dtype=torch.int32, device=device)
        emb = torch.nn.functional.normalize(emb.float(), dim=-1)

        unique_cids = torch.unique(cluster_ids[cluster_ids >= 0])
        next_track_id = [0]

        def process_single_cluster(cid):
            mask = cluster_ids == cid
            idxs = mask.nonzero(as_tuple=True)[0]
            sub_emb = emb[idxs]

            sub_labels = _cluster_subset(
                sub_emb,
                num_points=num_points,
                temperature=temperature,
                min_cluster_size=min_cluster_size,
                give_remainder_own_id=give_remainder_own_id
            )

            valid = sub_labels >= 0
            if valid.any():
                local_max = int(sub_labels[valid].max().item()) + 1
                with track_id_lock:  # Ensuring thread-safety here
                    offset = next_track_id[0]
                    next_track_id[0] += local_max
                sub_labels[valid] += offset

            return idxs, sub_labels

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_single_cluster, cid) for cid in unique_cids]

            for future in futures:
                idxs, sub_labels = future.result()
                track_ids[idxs] = sub_labels

        clusters.append(track_ids.cpu())

    return clusters, similarity_matrix
