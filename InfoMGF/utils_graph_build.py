import numpy as np
from sklearn.metrics import mutual_info_score
from scipy.signal import coherence
import scipy.sparse as sp
from numba import jit, prange
import torch
from tslearn.metrics import cdist_dtw
from scipy.spatial.distance import euclidean as _euclidean
from torch_geometric.utils import to_scipy_sparse_matrix,degree
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse, to_dense_adj
brain_regions = [
    [1, 10, 4],          # Region 1
    [2, 3, 5, 7, 8, 13],   # Region 2
    [12, 24],            # Region 3
    [6, 11, 14, 17, 18, 20], # Region 4
    [9, 15, 16, 19, 21, 22, 23, 27], # Region 5
    [30, 33, 35, 36, 37, 41, 43, 48], # Region 6 (Hemisphere 2)
    [31, 32, 34, 39, 42, 45],         # Region 7 (Hemisphere 2)
    [26, 38],                       # Region 8 (Hemisphere 2)
    [44, 46, 49, 50, 51, 53],       # Region 9 (Hemisphere 2)
    [40, 47, 52]
]

def make_region_map(brain_regions, num_channels):
    region_map = np.full(num_channels, -1, dtype=int) 
    for region_id, channels in enumerate(brain_regions):
        for ch in channels:
            if ch - 1 < num_channels:
                region_map[ch-1] = region_id
            else:
                print(f"Warning: Channel {ch} is out of range (num_channels={num_channels})")
    
    assigned_count = np.sum(region_map != -1)
    unassigned_count = np.sum(region_map == -1)
    
    print(f"通道分配统计:")
    print(f"  总通道数: {num_channels}")
    print(f"  已分配到脑区的通道: {assigned_count}")
    print(f"  未分配的通道: {unassigned_count}")
    if unassigned_count > 0:
        unassigned_channels = np.where(region_map == -1)[0]
        print(f"  未分配的通道索引: {[i+1 for i in unassigned_channels]}") 
    return region_map

def compute_global_edges(train_dataset, n_bins=16, strategy="uniform", eps=1e-8):
    C = train_dataset[0].shape[1]
    all_data = np.concatenate(train_dataset, axis=0)
    edges_list = []
    constant_channels = []
    
    for i in range(C):
        x = all_data[:, i]

        if x.max() - x.min() < eps:
            edges_list.append(None) 
            constant_channels.append(i)
            continue
        
        if strategy == "uniform":
            edges = np.linspace(x.min(), x.max(), n_bins + 1)[1:-1]
        elif strategy == "quantile":
            edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))[1:-1]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # 验证edges的有效性
        unique_edges = np.unique(edges)
        if len(unique_edges) < len(edges) * 0.5:  # 如果超过一半的边界值相同
            print(f"Warning: Channel {i} has low diversity edges (only {len(unique_edges)} unique values)")
        
        edges_list.append(edges)
    
    if constant_channels:
        print(f"Found {len(constant_channels)} constant channels: {constant_channels[:10]}{'...' if len(constant_channels) > 10 else ''}")
    
    return edges_list

def discretize_data_strict(data, n_bins=16, strategy="uniform", eps=1e-8,
                           global_edges=None):
    T, C = data.shape
    binned = np.zeros((T, C), dtype=np.int32)
    edges_list = [] if global_edges is None else global_edges

    for i in range(C):
        x = data[:, i]
        if global_edges is None:
            if x.max() - x.min() < eps:
                edges = None
                binned[:, i] = 0
                edges_list.append(edges)
                continue
            if strategy == "uniform":
                edges = np.linspace(x.min(), x.max(), n_bins + 1)[1:-1]
            elif strategy == "quantile":
                edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))[1:-1]
            else:
                raise ValueError("Unknown strategy")
            edges_list.append(edges)
        else:
            edges = global_edges[i]

        if edges is not None:
            binned[:, i] = np.digitize(x, edges, right=False)

    return binned, edges_list

def _mi_matrix_from_binned(binned_labels):
    W, C = binned_labels.shape
    M = np.zeros((C, C), dtype=float)
    for i in range(C):
        xi = binned_labels[:, i]
        for j in range(i, C):
            xj = binned_labels[:, j]
            val = mutual_info_score(xi, xj)
            M[i, j] = val
            M[j, i] = val
    return M

def build_intra_region_view_mi(
    time_series_data, region_map, global_edges,
    n_bins=16, strategy="uniform",
    window_size=400, stride=200,
    zero_diag=True, eps=1e-8
):
    T, C = time_series_data.shape

    max_windows = 20 
    window_starts = np.linspace(0, T - window_size, min(max_windows, (T - window_size) // stride + 1), dtype=int)
    
    A_mean = np.zeros((C, C), dtype=float)
    n_windows = len(window_starts)
    
    for start in window_starts:
        end = start + window_size
        window = time_series_data[start:end, :]

        binned, _ = discretize_data_strict(
            window, n_bins=n_bins, strategy=strategy, eps=eps, global_edges=global_edges
        )

        M_n = _mi_matrix_from_binned(binned)
        A_mean += M_n

    A_mean /= n_windows
    A_channel = A_mean

    intra_mask = np.zeros((C, C), dtype=float)
    assigned_indices = np.where(region_map != -1)[0]
    
    for i in assigned_indices:
        for j in assigned_indices:
            if i != j and region_map[i] == region_map[j]:
                intra_mask[i, j] = 1.0
    
    G_intra = A_channel * intra_mask

    if zero_diag:
        np.fill_diagonal(G_intra, 0.0)
        np.fill_diagonal(A_channel, 0.0)

    return G_intra, A_channel, n_windows

def build_global_view(times_series, w1,w2,fs=1.0):
    T, C = times_series.shape
    D_dtw=cdist_dtw(times_series.T)
    sigma = np.median(D_dtw[D_dtw > 0])
    S_dtw = np.exp(- (D_dtw ** 2) / (sigma ** 2 + 1e-8))
    S_coh = np.zeros((C, C))
    for i in range(C):
        for j in range(i, C):
            f, Cxy = coherence(times_series[:, i], times_series[:, j], fs=fs, nperseg=min(256, T))
            val = np.mean(Cxy) 
            S_coh[i, j] = val
            S_coh[j, i] = val
    S_global = w1 * S_dtw + w2 * S_coh


    return S_global, S_dtw, S_coh
def topk_sparsify_sym_row_normalize(adj_matrix, k):
    C = adj_matrix.shape[0]

    adj_sparse = adj_matrix.copy()

    np.fill_diagonal(adj_sparse, 0)

    if k < C:
        thresholds = np.partition(adj_sparse, -k, axis=1)[:, -k]
        mask = adj_sparse >= thresholds[:, np.newaxis]
        adj_sparse = adj_sparse * mask
    adj_sparse = np.maximum(adj_sparse, adj_sparse.T)

    row_sums = adj_sparse.sum(axis=1)
    row_sums[row_sums == 0] = 1
    adj_normalized = adj_sparse / row_sums[:, np.newaxis]
    
    return adj_normalized

def compute_degree_encoding(g, n_dg=16):
    g_dg = degree(g.edge_index[0], num_nodes=g.num_nodes).numpy().clip(1, n_dg)
    SE_dg = torch.zeros([g.num_nodes, n_dg])
    for i in range(len(g_dg)):
        SE_dg[i, int(g_dg[i]-1)] = 1
    return SE_dg.numpy()

def compute_random_walk_encoding(g, n_rw=16):
    A = to_scipy_sparse_matrix(g.edge_index, num_nodes=g.num_nodes)
    D = (degree(g.edge_index[0], num_nodes=g.num_nodes) ** -1.0).numpy()
    D_inv = sp.diags(D)

    RW = A @ D_inv
    M = RW

    SE_rw = [torch.from_numpy(M.diagonal()).float()]
    M_power = M
    for _ in range(n_rw - 1):
        M_power = M_power @ M
        SE_rw.append(torch.from_numpy(M_power.diagonal()).float())

    SE_rw = torch.stack(SE_rw, dim=-1)
    return SE_rw.numpy() 

def compute_combined_encoding(adj_matrix, n_rw=16, n_dg=16):

    degree_enc = compute_degree_encoding(adj_matrix, n_dg)
    rw_enc = compute_random_walk_encoding(adj_matrix, n_rw)

    combined_enc = np.concatenate([degree_enc, rw_enc], axis=1)
    print(f"Combined encoding stats - shape: {combined_enc.shape}, mean: {np.mean(combined_enc):.6f}, std: {np.std(combined_enc):.6f}")
    return combined_enc

def compute_structure_encodings(adj_input, encoding_type='rw_dg', n_rw=16, n_dg=16):
    if isinstance(adj_input, np.ndarray):
        adj_t = torch.from_numpy(adj_input).float()
        edge_index, edge_weight = dense_to_sparse(adj_t)
        g = Data(edge_index=edge_index, edge_weight=edge_weight, num_nodes=adj_t.shape[0])
    elif isinstance(adj_input, torch.Tensor):
        adj_t = adj_input.float()
        edge_index, edge_weight = dense_to_sparse(adj_t)
        g = Data(edge_index=edge_index, edge_weight=edge_weight, num_nodes=adj_t.size(0))
    elif isinstance(adj_input, Data):
        g = adj_input
        if not hasattr(g, 'num_nodes') or g.num_nodes is None:
            if hasattr(g, 'edge_index') and g.edge_index is not None:
                g.num_nodes = int(g.edge_index.max().item()) + 1
            else:
                raise ValueError("Input Data has neither num_nodes nor edge_index.")
        if not hasattr(g, 'edge_weight') or g.edge_weight is None:
            if hasattr(g, 'edge_attr') and g.edge_attr is not None:
                g.edge_weight = g.edge_attr
            else:
                if hasattr(g, 'edge_index') and g.edge_index is not None:
                    g.edge_weight = torch.ones(g.edge_index.size(1), dtype=torch.float32)
                else:
                    g.edge_weight = torch.tensor([], dtype=torch.float32)
    else:
        raise TypeError(f"Unsupported adj_input type: {type(adj_input)}")

    try:
        ei = g.edge_index
        ew = g.edge_weight if hasattr(g, 'edge_weight') else None
        dense = to_dense_adj(ei, edge_attr=ew)[0].cpu().numpy()
        matrix_size = dense.shape
        input_mean = float(np.mean(dense))
        input_std = float(np.std(dense))
        input_nonzero = int(np.count_nonzero(dense))
        print(f"Computing {encoding_type} encoding for matrix with shape {matrix_size}")
        print(f"Input matrix stats - mean: {input_mean:.6f}, std: {input_std:.6f}, non-zero: {input_nonzero}")
    except Exception as e:
        print("Warning: failed to densify for stats:", e)
        matrix_size = (int(g.num_nodes), int(g.num_nodes))
        print(f"Computing {encoding_type} encoding for matrix with shape {matrix_size}")

    num_edges = int(g.edge_index.size(1)) if hasattr(g, 'edge_index') and g.edge_index is not None else 0
    if num_edges == 0 or (hasattr(g, 'edge_weight') and g.edge_weight.numel() == 0):
        print("Warning: Zero adjacency matrix detected (no edges), using random encoding")
        n_nodes = int(g.num_nodes)
        if encoding_type == 'dg':
            return np.random.normal(0, 1, (n_nodes, n_dg))
        elif encoding_type == 'rw':
            return np.random.normal(0, 1, (n_nodes, n_rw))
        else:
            return np.random.normal(0, 1, (n_nodes, n_rw + n_dg))

    if encoding_type == 'dg':
        result = compute_degree_encoding(g, n_dg)
    elif encoding_type == 'rw':
        result = compute_random_walk_encoding(g, n_rw)
    elif encoding_type == 'rw_dg':
        result = compute_combined_encoding(g, n_rw, n_dg)
    else:
        raise ValueError(f"Unknown encoding type: {encoding_type}")

    if isinstance(result, torch.Tensor):
        result = result.cpu().numpy()

    print(f"Final encoding stats - shape: {result.shape}, mean: {np.mean(result):.6f}, std: {np.std(result):.6f}")
 
    return result
