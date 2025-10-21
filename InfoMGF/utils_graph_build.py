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

def discretize_data_per_sample(data, n_bins=16, strategy="uniform", eps=1e-8):
    T, C = data.shape
    binned = np.zeros((T, C), dtype=np.int32)
    
    constant_channels = 0
    for i in range(C):
        x = data[:, i]
        x_std = np.std(x)
        x_range = np.max(x) - np.min(x)
            
        # 为当前样本的当前通道单独计算分箱边界
        if strategy == "uniform":
            edges = np.linspace(x.min(), x.max(), n_bins + 1)[1:-1]
        elif strategy == "quantile":
            edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))[1:-1]
        else:
            raise ValueError("Unknown strategy")
        
        # 检查边界有效性
        unique_edges = np.unique(edges)
        if len(unique_edges) < len(edges) * 0.5:
            print(f"Warning: Sample channel {i} has low diversity edges")
        binned[:, i] = np.digitize(x, edges, right=False)
    return binned

def _mi_matrix_from_binned(binned_labels):
    W, C = binned_labels.shape
    M = np.zeros((C, C), dtype=float)

    valid_channels = []
    for i in range(C):
        xi = binned_labels[:, i]
        unique_vals = len(np.unique(xi))
        if unique_vals > 1:
            valid_channels.append(i)
 
    if len(valid_channels) < 2:
        print(f"互信息计算失败: 只有 {len(valid_channels)} 个有效通道")
        return M

    for i_idx, i in enumerate(valid_channels):
        xi = binned_labels[:, i]
        for j in valid_channels[i_idx:]:
            xj = binned_labels[:, j]
            try:
                val = mutual_info_score(xi, xj)
                M[i, j] = val
                M[j, i] = val
            except Exception:
                M[i, j] = 0
                M[j, i] = 0
    
    return M


def build_intra_region_view_mi(
    time_series_data, region_map, 
    n_bins=16, strategy="uniform",
    window_size=400, stride=200,
    zero_diag=True, eps=1e-8
):
    T, C = time_series_data.shape

    max_windows = 20 
    window_starts = np.linspace(0, T - window_size, min(max_windows, (T - window_size) // stride + 1), dtype=int)
    
    A_mean = np.zeros((C, C), dtype=float)
    n_windows = len(window_starts)
    
    valid_windows = 0
    for start in window_starts:
        end = start + window_size
        window = time_series_data[start:end, :]
        
        # 检查窗口数据质量
        window_std = np.std(window)
        if window_std < eps:
            continue
            
        valid_windows += 1

        # 使用简化的离散化
        binned= discretize_data_per_sample(
            window, n_bins=n_bins, strategy=strategy, eps=eps
        )

        M_n = _mi_matrix_from_binned(binned)
        A_mean += M_n

    if valid_windows == 0:
        return np.zeros((C, C)), np.zeros((C, C)), 0
        
    A_mean /= valid_windows
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

    return G_intra, A_channel, valid_windows

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
