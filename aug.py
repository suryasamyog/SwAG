import torch
import torch.nn.functional as F 
from torch_geometric.utils import to_undirected

def drop_node_features(x: torch.Tensor, drop_prob: float = 0.2) -> torch.Tensor:
    if drop_prob == 0.0:
        return x
        
    drop_mask = torch.empty((x.size(1),), dtype=torch.float32, device=x.device).uniform_(0, 1) > drop_prob
    
    x_aug = x.clone()
    x_aug[:, ~drop_mask] = 0.0
    
    return x_aug

def stalk_masking(x: torch.Tensor, final_d: int, emb_dim: int, multi_view: bool = False, n_multiview: int = 2, epsilon: float = 0.05):
    N = x.size(0)

    x_reshaped = x.view(N, final_d, emb_dim)
    views = []

    if final_d < 2:
        return [x.view(N, -1), x.view(N, -1)]
    
    masked_dims = torch.randperm(final_d)[:2] 

    for d_idx in masked_dims:
        v_main = x_reshaped.clone()
        v_main[:, d_idx, :] = 0 
        views.append(v_main.view(N, -1))

    if multi_view:
        for _ in range(n_multiview):
            noise = torch.randn_like(x_reshaped) * epsilon
            v_eps = x_reshaped + noise
            views.append(v_eps.view(N, -1)) 
            
    return views


from torch_geometric.utils import dropout_edge

def drop_graph_edges(edge_index, drop_prob: float = 0.2, training: bool = True):
    if not training or drop_prob == 0.0:
        return edge_index

    edge_index_dropped, _ = dropout_edge(edge_index, p=drop_prob, force_undirected=True)
    return edge_index_dropped

def generate_mean_pool_views(x, edge_index, num_nodes):
    src, tgt = edge_index
    
    v2 = torch.zeros_like(x)
    
    idx = src.view(-1, 1).expand_as(x[tgt])
    v2.scatter_add_(0, idx, x[tgt])
    
    deg = torch.zeros(num_nodes, device=x.device, dtype=x.dtype)
    deg.scatter_add_(0, src, torch.ones_like(src, dtype=x.dtype))
    
    v2 = v2 / (deg.view(-1, 1) + 1e-8)
    
    return [x, v2]

def khop_mean_aggregate(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
    k: int = 2,
    include_self: bool = True,
    to_undirected_graph: bool = True,
    exact_k: bool = False,
    combine: str = "last",  
    normalize_each_step: bool = False,
):

    assert k >= 1, "k must be >= 1"
    assert combine in ["last", "mean"]

    device = x.device
    dtype = x.dtype

    if to_undirected_graph:
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)

    src, dst = edge_index[0], edge_index[1]

    h = x
    hops = []

    use_self = include_self and not exact_k

    for _ in range(k):
        out = torch.zeros_like(h)

        out.index_add_(0, dst, h[src])

        deg = torch.bincount(dst, minlength=num_nodes).to(device=device, dtype=dtype)

        if use_self:
            out = out + h
            deg = deg + 1.0

        out = out / deg.clamp_min(1.0).view(-1, 1)

        if normalize_each_step:
            out = F.normalize(out, dim=1, p=2)

        h = out
        hops.append(h)

    if combine == "last":
        return hops[-1]

    return torch.stack(hops, dim=0).mean(dim=0)
