
import torch, torch_cluster
from typing import Sequence, List

# ----------  GPU Union‑Find  -------------------------------
class _UnionFind(torch.autograd.Function):
    @staticmethod
    def forward(ctx, edges, n):
        """
        edges : (2,E) int64
        n     : scalar int64 (number of vertices)
        returns: (n,) int64 parent array with path‑compression applied
        """
        # parents initialised to self
        parent = torch.arange(n, dtype=torch.int64, device=edges.device)

        @torch.jit.script
        def _find(p: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            # recursive find with path compression
            while p[x] != x:
                p[x] = p[p[x]]
                x = p[x]
            return x

        # union every edge
        for u, v in edges.t():
            # ϒ compile hoists the loop into a single block
            root_u = _find(parent, u)
            root_v = _find(parent, v)
            if root_u != root_v:
                # union by smaller index to keep deterministic
                if root_u < root_v:
                    parent[root_v] = root_u
                else:
                    parent[root_u] = root_v
        return parent

# ------------------------------------------------------------
@torch.no_grad()
def _cluster_event_knn(
        emb: torch.Tensor, *,
        k: int,
        temperature: float,
        min_cluster_size: int,
        give_remainder_own_id: bool
) -> torch.Tensor:
    """
    New, fully‑GPU, O(Nk) clustering.
    emb  : (N,D) fp16 / fp32 on CUDA
    k    : num_points from the original API
    """
    N = emb.size(0)
    if N < min_cluster_size:
        return torch.full((N,), -1, dtype=torch.int32, device='cpu')

    # (1) build k‑NN graph         ---------------------------
    edge_index = torch_cluster.knn_graph(
        emb.to(torch.float32), k=k, batch=None, loop=False
    )                               # shape (2, E)

    # (2) cosine similarity filter  --------------------------
    emb_n = torch.nn.functional.normalize(emb.to(torch.float32), dim=-1)
    sim = (emb_n[edge_index[0]] * emb_n[edge_index[1]]).sum(dim=1)
    keep = (sim / temperature) > 1.0          # τ = 1 ≈ original anchor ≥ neighbour rule
    edge_index = edge_index[:, keep]

    # (3) GPU Union‑Find           ---------------------------
    parents = _UnionFind.apply(edge_index, torch.tensor(N, device=emb.device))

    # canonical component labels
    comp = torch.where(
        parents == torch.arange(N, device=emb.device),
        torch.arange(N, device=emb.device),
        parents
    )

    # (4) re‑map to compact 0…C‑1 and size filter
    #     – we have to count component sizes first
    uniq, inv = torch.unique(comp, return_inverse=True)
    sizes = torch.bincount(inv, minlength=uniq.size(0))
    is_big = sizes >= min_cluster_size
    big_map = torch.full_like(uniq, -1)
    big_map[is_big] = torch.arange(int(is_big.sum()), device=emb.device)

    labels = big_map[inv]                             # (N,)
    if give_remainder_own_id:
        # each small component gets its own id starting after big ones
        small_comp = (~is_big)[inv]
        labels[small_comp] = (small_comp.nonzero().squeeze(1)
                              + int(is_big.sum())).to(labels.dtype)

    return labels.to(torch.int32)

# ------------------------------------------------------------
#  Main public wrapper  (keeps original signature)
# ------------------------------------------------------------
def clustering_similarity(
        pred_embeds: Sequence[torch.Tensor],
        *,
        cluster_ids_in: Sequence[torch.Tensor] = None,
        num_points: int = 5,
        temperature: float = 0.05,
        min_cluster_size: int = 3,
        give_remainder_own_id: bool = True,
        save_similarity_for_event: bool = False
) -> List[torch.Tensor]:
    out, sim_dump = [], None
    for i, emb in enumerate(pred_embeds):
        if emb.numel() == 0:
            out.append(torch.empty((0,), dtype=torch.int32))
            continue

        # -------- partition by pre‑clusters, but stay on GPU
        if cluster_ids_in is not None:
            cid_all  = cluster_ids_in[i].to(emb.device, non_blocking=True)
            uniq_cid = cid_all.unique()
            labels   = torch.full_like(cid_all, -1, dtype=torch.int32)

            base = 0
            for cid in uniq_cid[uniq_cid >= 0]:
                sel  = (cid_all == cid)
                sub  = _cluster_event_knn(
                         emb[sel], k=num_points,
                         temperature=temperature,
                         min_cluster_size=min_cluster_size,
                         give_remainder_own_id=give_remainder_own_id
                       )
                good = sub >= 0
                sub[good] += base
                base += sub.max().item() + 1 if good.any() else 0
                labels[sel] = sub
        else:
            labels = _cluster_event_knn(
                        emb, k=num_points,
                        temperature=temperature,
                        min_cluster_size=min_cluster_size,
                        give_remainder_own_id=give_remainder_own_id)

        # optional similarity dump (first event only)
        if save_similarity_for_event and i == 0:
            sim_dump = (torch.matmul(emb, emb.T) / temperature).float().cpu()

        out.append(labels)

    out = [lbl.cpu() for lbl in out]

    return out, sim_dump
