import torch
import torch.nn.functional as F  # <--- Add this line
from torch_geometric.utils import to_undirected

def compute_forman_ricci_prior(edge_index, num_nodes):
    """
    Computes a blazing-fast approximation of Forman-Ricci curvature for each edge.
    F(u,v) = 2 - deg(u) - deg(v) + 3 * triangles(u,v)
    For speed, we use the 1st-order approximation (ignoring triangles).
    Returns a normalized [0, 1] gating tensor to pre-condition transport.
    """
    src, tgt = edge_index
    device = edge_index.device
    
    # 1. Compute node degrees
    deg = torch.zeros(num_nodes, device=device, dtype=torch.float)
    deg.scatter_add_(0, src, torch.ones_like(src, dtype=torch.float))
    
    # 2. Base Forman Curvature (Negative = Bottleneck/Heterophilic bridge)
    forman_curvature = 2.0 - deg[src] - deg[tgt]
    
    # 3. Normalize into a [0, 1] gate. 
    # Edges with highly negative curvature get pushed closer to 0 (suppressed).
    # We scale by the mean degree to keep the sigmoid responsive across datasets.
    mean_deg = deg.mean() + 1e-5
    ricci_gate = torch.sigmoid(forman_curvature / mean_deg)
    # transport_weight = transport_weight_raw * torch.exp(ricci_gate)
    
    return ricci_gate.unsqueeze(-1)

# =============================================================================
# Strategy 1: Standard Feature Masking
# =============================================================================
def drop_node_features(x: torch.Tensor, drop_prob: float = 0.2) -> torch.Tensor:
    """
    Randomly masks a percentage of node features to 0.
    Applied BEFORE the encoder to create distinct views of the graph.
    """
    if drop_prob == 0.0:
        return x
        
    # Generate a random mask for the feature dimension
    drop_mask = torch.empty((x.size(1),), dtype=torch.float32, device=x.device).uniform_(0, 1) > drop_prob
    
    x_aug = x.clone()
    x_aug[:, ~drop_mask] = 0.0
    
    return x_aug

# =============================================================================
# Strategy 2: Targeted Stalk Masking
# =============================================================================
def stalk_masking(x: torch.Tensor, final_d: int, emb_dim: int, multi_view: bool = False, n_multiview: int = 2, epsilon: float = 0.05):
    """
    Masks out specific stalk dimensions. 
    Requires the input `x` to have a feature dimension equal to final_d * emb_dim.
    """
    N = x.size(0)
    
    # Reshape to access the 'd' stalk dimension: [N, final_d, emb_dim]
    x_reshaped = x.view(N, final_d, emb_dim)
    views = []

    if final_d < 2:
        # Fallback: Just return identical views or apply dropout
        return [x.view(N, -1), x.view(N, -1)]
    
    # Randomly select 2 'd' dimensions to mask out to create 2 distinct primary views
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

# def generate_sheaf_views(x, final_d, emb_dim, maps, edge_index, num_nodes):
#     """
#     Creates a second view by transporting neighbor embeddings 
#     to the central node using the learned sheaf restriction maps.
#     """
#     src, tgt = edge_index
#     E = src.shape[0] // 2
#     N = x.size(0)
    
#     src_ = src[:E]
#     tgt_ = tgt[:E]
    
#     x_reshaped = x.view(N, final_d, emb_dim)
    
#     r_src = maps[:E]
#     r_tgt = maps[E:]
    
#     z_src = x_reshaped[src_]
#     z_tgt = x_reshaped[tgt_]
    
#     # Apply restriction maps
#     s_src = r_src.unsqueeze(-1) * z_src        
#     s_tgt = r_tgt.unsqueeze(-1) * z_tgt
    
#     t_src = r_src.unsqueeze(-1) * s_tgt        
#     t_tgt = r_tgt.unsqueeze(-1) * s_src
    
#     # Calculate disagreement (optional, useful for separate regularizers)
#     disagreement = (s_src - s_tgt).norm(dim=(-2, -1))
    
#     v2 = torch.zeros(num_nodes, final_d, emb_dim, device=x.device, dtype=x.dtype)
#     deg = torch.zeros(num_nodes, device=x.device, dtype=x.dtype)
    
#     idx = src_.view(-1, 1, 1).expand_as(t_src)
#     v2.scatter_add_(0, idx, t_src)

#     idx = tgt_.view(-1, 1, 1).expand_as(t_tgt)
#     v2.scatter_add_(0, idx, t_tgt)

#     deg.scatter_add_(0, src_, torch.ones(E, device=x.device))
#     deg.scatter_add_(0, tgt_, torch.ones(E, device=x.device))

#     v2 = v2 / (deg.view(-1, 1, 1) + 1e-8)
#     v2 = v2.view(num_nodes, -1)
    
#     # View 1 is the original latent encoding, View 2 is the transported neighborhood
#     return [x, v2], disagreement

# def generate_sheaf_views(x, final_d, emb_dim, maps, edge_index, num_nodes):
#     """
#     Creates a second view by transporting neighbor embeddings 
#     to the central node using the learned sheaf restriction maps.
#     Universally supports DiagSheaf (2D maps) and Bundle/General Sheaf (3D maps).
#     """
#     src, tgt = edge_index
#     E = src.shape[0] // 2
#     N = x.size(0)
    
#     src_ = src[:E]
#     tgt_ = tgt[:E]
    
#     x_reshaped = x.view(N, final_d, emb_dim) # [N, d, f]
    
#     r_src = maps[:E]
#     r_tgt = maps[E:]
    
#     # ---------------------------------------------------------
#     # 1. Dimension Check & Padding for add_lp / add_hp
#     # ---------------------------------------------------------
#     d_maps = r_src.shape[1]
#     is_matrix_sheaf = (r_src.dim() == 3) # True for Bundle/General, False for Diag
    
#     if final_d > d_maps:
#         pad_size = final_d - d_maps
#         if is_matrix_sheaf:
#             # Pad 3D maps with Identity blocks for the extra dimensions
#             new_r_src = torch.zeros(E, final_d, final_d, device=maps.device, dtype=maps.dtype)
#             new_r_tgt = torch.zeros(E, final_d, final_d, device=maps.device, dtype=maps.dtype)
#             new_r_src[:, :d_maps, :d_maps] = r_src
#             new_r_tgt[:, :d_maps, :d_maps] = r_tgt
#             idx = torch.arange(d_maps, final_d)
#             new_r_src[:, idx, idx] = 1.0
#             new_r_tgt[:, idx, idx] = 1.0
#             r_src, r_tgt = new_r_src, new_r_tgt
#         else:
#             # Pad 2D Diag maps with 1.0s
#             pad_ones = torch.ones((E, pad_size), device=maps.device, dtype=maps.dtype)
#             r_src = torch.cat([r_src, pad_ones], dim=1)
#             r_tgt = torch.cat([r_tgt, pad_ones], dim=1)

#     # ---------------------------------------------------------
#     # 2. Geometric Transport
#     # ---------------------------------------------------------
#     if is_matrix_sheaf:
#         # Full Matrix Transport (BundleSheaf / GeneralSheaf)
#         # Using Batch Matrix Multiply: [E, d, d] @ [E, d, f] -> [E, d, f]
#         s_src = torch.bmm(r_src, x_reshaped[src_])
#         s_tgt = torch.bmm(r_tgt, x_reshaped[tgt_])
        
#         # r_src.transpose(1, 2) acts as the transpose of the restriction map
#         t_src = torch.bmm(r_src.transpose(1, 2), s_tgt)
#         t_tgt = torch.bmm(r_tgt.transpose(1, 2), s_src)
        
#         # Free heavy intermediate tensors
#         del s_src, s_tgt
#     else:
#         # Fast Diagonal Transport (DiagSheaf)
#         transport_weight = (r_src * r_tgt).unsqueeze(-1) # [E, d, 1]
#         t_src = transport_weight * x_reshaped[tgt_]
#         t_tgt = transport_weight * x_reshaped[src_]
#         del transport_weight

#     # ---------------------------------------------------------
#     # 3. Aggregation & Degree Normalization
#     # ---------------------------------------------------------
#     v2 = torch.zeros(num_nodes, final_d, emb_dim, device=x.device, dtype=x.dtype)
    
#     v2.scatter_add_(0, src_.view(-1, 1, 1).expand_as(t_src), t_src)
#     v2.scatter_add_(0, tgt_.view(-1, 1, 1).expand_as(t_tgt), t_tgt)
#     del t_src, t_tgt # Free memory

#     deg = torch.zeros(num_nodes, device=x.device, dtype=x.dtype)
#     deg.scatter_add_(0, src_, torch.ones(E, device=x.device))
#     deg.scatter_add_(0, tgt_, torch.ones(E, device=x.device))

#     v2 = v2 / (deg.view(-1, 1, 1) + 1e-8)
#     v2 = v2.view(num_nodes, -1)
    
#     return [x, v2], None

def generate_sheaf_views(x, final_d, emb_dim, maps, edge_index, num_nodes, use_laplacian_diagonal=False, use_ricci_prior=False):
    """
    Creates structural views using the learned sheaf restriction maps.
    If use_laplacian_diagonal=True, View 1 is the Sheaf Degree scaling.
    Otherwise, View 1 is the raw ego-node embedding.
    View 2 is always the transported neighborhood.
    """
    src, tgt = edge_index
    E = src.shape[0] // 2
    N = x.size(0)

    if maps is None:
        # Identity Sheaf Fallback: All edge restrictions are 1.0
        maps = torch.ones((src.shape[0], final_d), device=x.device, dtype=x.dtype)
    
    src_ = src[:E]
    tgt_ = tgt[:E]
    
    x_reshaped = x.view(N, final_d, emb_dim) # [N, d, f]
    
    r_src = maps[:E]
    r_tgt = maps[E:]
    
    # 1. Dimension Check & Padding (Keep your existing padding logic here)
    d_maps = r_src.shape[1]
    is_matrix_sheaf = (r_src.dim() == 3)
    
    if final_d > d_maps:
        pad_size = final_d - d_maps
        if is_matrix_sheaf:
            new_r_src = torch.zeros(E, final_d, final_d, device=maps.device, dtype=maps.dtype)
            new_r_tgt = torch.zeros(E, final_d, final_d, device=maps.device, dtype=maps.dtype)
            new_r_src[:, :d_maps, :d_maps] = r_src
            new_r_tgt[:, :d_maps, :d_maps] = r_tgt
            idx = torch.arange(d_maps, final_d)
            new_r_src[:, idx, idx] = 1.0
            new_r_tgt[:, idx, idx] = 1.0
            r_src, r_tgt = new_r_src, new_r_tgt
        else:
            pad_ones = torch.ones((E, pad_size), device=maps.device, dtype=maps.dtype)
            r_src = torch.cat([r_src, pad_ones], dim=1)
            r_tgt = torch.cat([r_tgt, pad_ones], dim=1)

    # 2. Geometric Transport & Identity Penalty
    ortho_penalty = 0.0

    if is_matrix_sheaf:
        transport_map = torch.bmm(r_src.transpose(1, 2), r_tgt)
        identity = torch.eye(final_d, device=maps.device, dtype=maps.dtype).unsqueeze(0).expand(E, -1, -1)
        ortho_penalty = F.mse_loss(transport_map, identity)

        # View 2: Transported Neighborhood (Off-diagonal)
        s_src = torch.bmm(r_src, x_reshaped[src_])
        s_tgt = torch.bmm(r_tgt, x_reshaped[tgt_])
        t_src = torch.bmm(r_src.transpose(1, 2), s_tgt)
        t_tgt = torch.bmm(r_tgt.transpose(1, 2), s_src)
        
        # View 1: Sheaf Degree Scaling (Diagonal)
        if use_laplacian_diagonal:
            t_self_src = torch.bmm(r_src.transpose(1, 2), s_src)
            t_self_tgt = torch.bmm(r_tgt.transpose(1, 2), s_tgt)
            
        del s_src, s_tgt, transport_map, identity
    else:
        # Fast Diagonal Transport
        transport_weight_raw = (r_src * r_tgt)
        identity = torch.ones_like(transport_weight_raw)
        ortho_penalty = F.mse_loss(transport_weight_raw, identity)

        if use_ricci_prior:
            # ---> NEW: THE RICCI PRIOR <---
            # Compute the geometric prior gate based on graph topology
            ricci_gate = compute_forman_ricci_prior(edge_index, N)
            
            # Modulate the learned restriction maps with the topological prior!
            transport_weight_raw = transport_weight_raw * ricci_gate[:E]
            # ------------------------------

        # View 2: Transported Neighborhood
        transport_weight = transport_weight_raw.unsqueeze(-1)
        t_src = transport_weight * x_reshaped[tgt_]
        t_tgt = transport_weight * x_reshaped[src_]
        
        # View 1: Sheaf Degree Scaling
        if use_laplacian_diagonal:
            weight_self_src = (r_src * r_src).unsqueeze(-1)
            weight_self_tgt = (r_tgt * r_tgt).unsqueeze(-1)
            t_self_src = weight_self_src * x_reshaped[src_]
            t_self_tgt = weight_self_tgt * x_reshaped[tgt_]
            
        del transport_weight, transport_weight_raw, identity

    # 3. Aggregation & Degree Normalization
    v2 = torch.zeros(num_nodes, final_d, emb_dim, device=x.device, dtype=x.dtype)
    v2.scatter_add_(0, src_.view(-1, 1, 1).expand_as(t_src), t_src)
    v2.scatter_add_(0, tgt_.view(-1, 1, 1).expand_as(t_tgt), t_tgt)
    
    if use_laplacian_diagonal:
        v1 = torch.zeros(num_nodes, final_d, emb_dim, device=x.device, dtype=x.dtype)
        v1.scatter_add_(0, src_.view(-1, 1, 1).expand_as(t_self_src), t_self_src)
        v1.scatter_add_(0, tgt_.view(-1, 1, 1).expand_as(t_self_tgt), t_self_tgt)
    else:
        v1 = x_reshaped

    deg = torch.zeros(num_nodes, device=x.device, dtype=x.dtype)
    deg.scatter_add_(0, src_, torch.ones(E, device=x.device))
    deg.scatter_add_(0, tgt_, torch.ones(E, device=x.device))

    v2 = v2 / (deg.view(-1, 1, 1) + 1e-8)
    v2 = v2.view(num_nodes, -1)
    
    if use_laplacian_diagonal:
        v1 = v1 / (deg.view(-1, 1, 1) + 1e-8)
        v1 = v1.view(num_nodes, -1)
    else:
        v1 = x # Return original flat vector

    return [v1, v2], ortho_penalty

from torch_geometric.utils import dropout_edge

def drop_graph_edges(edge_index, drop_prob: float = 0.2, training: bool = True):
    """
    Randomly drops edges from the adjacency matrix. 
    Applied BEFORE the encoder.
    """
    if not training or drop_prob == 0.0:
        return edge_index
        
    # PyG utility automatically handles undirected graph logic if needed
    edge_index_dropped, _ = dropout_edge(edge_index, p=drop_prob, force_undirected=True)
    return edge_index_dropped

def generate_mean_pool_views(x, edge_index, num_nodes):
    """
    Creates a second view by computing the unweighted 1-hop neighborhood average.
    Acts as a strong homophilic/low-pass filter without needing learned maps.
    """
    src, tgt = edge_index
    
    # We aggregate features from neighbors (tgt) to the central node (src)
    v2 = torch.zeros_like(x)
    
    # Expand src to match the feature dimension for scatter_add
    idx = src.view(-1, 1).expand_as(x[tgt])
    v2.scatter_add_(0, idx, x[tgt])
    
    # Calculate node degrees for normalization
    deg = torch.zeros(num_nodes, device=x.device, dtype=x.dtype)
    deg.scatter_add_(0, src, torch.ones_like(src, dtype=x.dtype))
    
    # Mean pool (add eps to prevent division by zero for isolated nodes)
    v2 = v2 / (deg.view(-1, 1) + 1e-8)
    
    # View 1 is the raw embedding, View 2 is the smoothed neighborhood
    return [x, v2]

def khop_mean_aggregate(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
    k: int = 2,
    include_self: bool = True,
    to_undirected_graph: bool = True,
    exact_k: bool = False,
    combine: str = "last",   # "last" or "mean"
    normalize_each_step: bool = False,
):
    """
    Computes k-hop mean aggregation in latent space.

    x: [N, D]
    edge_index: [2, E]

    If exact_k=False and include_self=True:
        H^{t+1} = mean over N(v) union {v}.
        This is a k-step diffusion / up-to-k smoothing due to self-loops.

    If exact_k=True:
        no self-loops are used inside propagation, so H^k represents
        walk-length-k neighborhood aggregation.

    combine:
        "last": return H^k
        "mean": return average of H^1, ..., H^k
    """
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

        # Aggregate neighbor messages: src -> dst.
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


def generate_khop_aggregate_views(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
    k: int = 2,
    include_self: bool = True,
    to_undirected_graph: bool = True,
    exact_k: bool = False,
    combine: str = "last",
    residual_alpha: float = 0.0,
    normalize_each_step: bool = False,
):
    """
    Returns:
        view1 = original latent node embedding
        view2 = k-hop neighborhood aggregate of view1

    residual_alpha:
        view2 = alpha * view1 + (1-alpha) * khop_agg
    """
    view1 = x

    view2 = khop_mean_aggregate(
        x=x,
        edge_index=edge_index,
        num_nodes=num_nodes,
        k=k,
        include_self=include_self,
        to_undirected_graph=to_undirected_graph,
        exact_k=exact_k,
        combine=combine,
        normalize_each_step=normalize_each_step,
    )

    if residual_alpha > 0.0:
        view2 = residual_alpha * view1 + (1.0 - residual_alpha) * view2

    return [view1, view2]