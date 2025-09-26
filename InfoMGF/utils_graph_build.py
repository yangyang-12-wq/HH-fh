import numpy as np
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score
from tslearn.metrics import cdist_dtw
from scipy.signal import coherence
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
    [40, 47, 52],
    [25, 28, 29] # Region 10 (Hemisphere 2)
]
def make_region_map(brain_regions,num_channels):
    region_map=np.zeros(num_channels,dtype=int)-1
    for region_id,channels in enumerate(brain_regions):
        for ch in channels:
            region_map[ch-1]=region_id
    if np.any(region_map==-1):
        raise ValueError("某些通道没有被分配到任何脑区")
    return region_map
#这个是在训练集上学习对于时间窗口的划分 这个就是之后在用到时间窗口划分的时候用到的
def compute_global_edges(train_dataset, n_bins=16, strategy="uniform"):

    C = train_dataset[0].shape[1]
    all_data = np.concatenate(train_dataset, axis=0)  # 拼接所有训练样本
    edges_list = []
    for i in range(C):
        x = all_data[:, i]
        if strategy == "uniform":
            edges = np.linspace(x.min(), x.max(), n_bins + 1)[1:-1]
        elif strategy == "quantile":
            edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))[1:-1]
        edges_list.append(edges)
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
                # 常量通道：全 0
                edges = None
                binned[:, i] = 0
                edges_list.append(edges)
                continue
            if strategy == "uniform":
                # 只取内部切点，保证恰好 n_bins 个箱（标签 0..n_bins-1）
                edges = np.linspace(x.min(), x.max(), n_bins + 1)[1:-1]
            elif strategy == "quantile":
                edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))[1:-1]
            else:
                raise ValueError("Unknown strategy")
            edges_list.append(edges)
        else:
            edges = global_edges[i]

        if edges is not None:
            binned[:, i] = np.digitize(x, edges, right=False)  # 0..n_bins-1

    return binned, edges_list

def build_intra_region_view_mi(
    time_series_data, region_map, global_edges,
    n_bins=16, strategy="uniform",
    window_size=400, stride=200,
    use_nmi=False, base="nat",
    zero_diag=True, eps=1e-8
):
    T, C = time_series_data.shape
    assert len(region_map) == C
    assert len(global_edges) == C

    A_mean = np.zeros((C, C), dtype=float)
    n_windows = 0

    for start in range(0, T - window_size + 1, stride):
        end = start + window_size
        window = time_series_data[start:end, :]

        #  固定全局切点进行离散化（避免数据泄漏 & 跨窗口可比）
        binned, _ = discretize_data_strict(
            window, n_bins=n_bins, strategy=strategy, eps=eps, global_edges=global_edges
        )

        # 单窗口互信息矩阵（公式 2）
        M_n = _mi_matrix_from_binned(binned, use_nmi=use_nmi, base=base)

        # 在线均值 = 跨窗口平均（公式 3）
        n_windows += 1
        A_mean += (M_n - A_mean) / n_windows

    if n_windows == 0:
        raise ValueError("数据长度不足以创建一个窗口，请减小 window_size 或 stride。")

    A_channel = A_mean

    # 仅保留同脑区边
    intra_mask = (region_map[:, None] == region_map[None, :]).astype(float)
    G_intra = A_channel * intra_mask

    if zero_diag:
        np.fill_diagonal(G_intra, 0.0)
        np.fill_diagonal(A_channel, 0.0)

    return G_intra, A_channel, n_windows

#这个就是公式二的具体实现这里默认是归一化了方便比较 返回值是一个包含所有顶点的互信值矩阵（处理的是一个时间窗口内 通道之间的关联度）
def _mi_matrix_from_binned(binned_labels, use_nmi=False, base="nat"):
    W, C = binned_labels.shape
    M = np.zeros((C, C), dtype=float)

    for i in range(C):
        xi = binned_labels[:, i]
        for j in range(i, C):
            xj = binned_labels[:, j]
            if use_nmi:
                val = normalized_mutual_info_score(xi, xj)
            else:
                val = mutual_info_score(xi, xj)
                if base == "bit":
                    val = val / np.log(2)
            M[i, j] = val
            M[j, i] = val
    return M



#下面是关于利用频率和时间上变化的相似度去构建全局的连接矩阵视图的
def build_global_view(times_series, w1,w2,fs=1.0):
    '''

    :param times_series: 这个是 ndarray, shape (T, C)
        T: 时间长度, C: 通道数
    :param w1:DTW的权重
    :param w2:coh频域的权重
    :param fs:这个是fnirs数据集在去数据的时候的频率
    :return:S_global (C, C)融合后的全局脑邻接矩阵
    '''
    T, C = times_series.shape
    #这个就是所有通道之间的DTW距离矩阵
    D_dtw=cdist_dtw(times_series.T)
    #这个算的是σDTW 是所有值中的中位数
    sigma = np.median(D_dtw[D_dtw > 0])
    #最终得到的 S_dtw 矩阵，其值介于 0 到 1 之间
    S_dtw = np.exp(- (D_dtw ** 2) / (sigma ** 2 + 1e-8))
    S_coh = np.zeros((C, C))
    for i in range(C):
        for j in range(i, C):
            f, Cxy = coherence(times_series[:, i], times_series[:, j], fs=fs, nperseg=min(256, T))
            val = np.mean(Cxy)  # 平均相干性
            S_coh[i, j] = val
            S_coh[j, i] = val
    S_global = w1 * S_dtw + w2 * S_coh


    return S_global, S_dtw, S_coh