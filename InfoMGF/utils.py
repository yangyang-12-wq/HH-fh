import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.neighbors import kneighbors_graph
import dgl
from sklearn import metrics
from munkres import Munkres
import os
import pickle

EOS = 1e-10


def apply_non_linearity(tensor, non_linearity, i):
    if non_linearity == 'elu':
        return F.elu(tensor * i - i) + 1
    elif non_linearity == 'relu':
        return F.relu(tensor)
    elif non_linearity == 'none':
        return tensor
    else:
        raise NameError('We dont support the non-linearity yet')


def split_batch(init_list, batch_size):
    groups = zip(*(iter(init_list),) * batch_size)
    end_list = [list(i) for i in groups]
    count = len(init_list) % batch_size
    end_list.append(init_list[-count:]) if count != 0 else end_list
    return end_list


def edge_deletion(adj, drop_r):
    edge_index = np.array(np.nonzero(adj))
    half_edge_index = edge_index[:, edge_index[0,:] < edge_index[1,:]]
    num_edge = half_edge_index.shape[1]
    samples = np.random.choice(num_edge, size=int(drop_r * num_edge), replace=False)
    dropped_edge_index = half_edge_index[:, samples].T
    adj[dropped_edge_index[:,0],dropped_edge_index[:,1]] = 0.
    adj[dropped_edge_index[:,1],dropped_edge_index[:,0]] = 0.
    return adj

def edge_addition(adj, add_r):
    edge_index = np.array(np.nonzero(adj))
    half_edge_index = edge_index[:, edge_index[0,:] < edge_index[1,:]]
    num_edge = half_edge_index.shape[1]
    num_node = adj.shape[0]
    added_edge_index_in = np.random.choice(num_node, size=int(add_r * num_edge), replace=True)
    added_edge_index_out = np.random.choice(num_node, size=int(add_r * num_edge), replace=True)
    adj[added_edge_index_in,added_edge_index_out] = 1.
    adj[added_edge_index_out,added_edge_index_in] = 1.
    return adj


def get_feat_mask(features, mask_rate):
    feat_node = features.shape[1]
    mask = torch.zeros(features.shape)
    samples = np.random.choice(feat_node, size=int(feat_node * mask_rate), replace=False)
    mask[:, samples] = 1
    return mask.to(torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu'))


def accuracy(preds, labels):
    pred_class = torch.max(preds, 1)[1]
    return torch.sum(torch.eq(pred_class, labels)).float() / labels.shape[0]


def nearest_neighbors(X, k, metric):
    adj = kneighbors_graph(X, k, metric=metric)
    adj = np.array(adj.todense(), dtype=np.float32)
    adj += np.eye(adj.shape[0])
    return adj


def nearest_neighbors_sparse(X, k, metric):
    adj = kneighbors_graph(X, k, metric=metric)
    loop = np.arange(X.shape[0])
    [s_, d_, val] = sp.find(adj)
    s = np.concatenate((s_, loop))
    d = np.concatenate((d_, loop))
    return s, d


def nearest_neighbors_pre_exp(X, k, metric, i):
    adj = kneighbors_graph(X, k, metric=metric)
    adj = np.array(adj.todense(), dtype=np.float32)
    adj += np.eye(adj.shape[0])
    adj = adj * i - i
    return adj


def nearest_neighbors_pre_elu(X, k, metric, i):
    adj = kneighbors_graph(X, k, metric=metric)
    adj = np.array(adj.todense(), dtype=np.float32)
    adj += np.eye(adj.shape[0])
    adj = adj * i - i
    return adj


def normalize(adj, mode, sparse=False):
    if not sparse:
        if mode == "sym":
            inv_sqrt_degree = 1. / (torch.sqrt(adj.sum(dim=1, keepdim=False)) + EOS)
            return inv_sqrt_degree[:, None] * adj * inv_sqrt_degree[None, :]
        elif mode == "row":
            inv_degree = 1. / (adj.sum(dim=1, keepdim=False) + EOS)
            return inv_degree[:, None] * adj
        else:
            exit("wrong norm mode")
    else:
        adj = adj.coalesce()
        if mode == "sym":
            inv_sqrt_degree = 1. / (torch.sqrt(torch.sparse.sum(adj, dim=1).values()) + EOS)
            D_value = inv_sqrt_degree[adj.indices()[0]] * inv_sqrt_degree[adj.indices()[1]]

        elif mode == "row":
            aa = torch.sparse.sum(adj, dim=1)
            bb = aa.values()
            inv_degree = 1. / (torch.sparse.sum(adj, dim=1).values() + EOS)
            D_value = inv_degree[adj.indices()[0]]
        else:
            exit("wrong norm mode")
        new_values = adj.values() * D_value

        return torch.sparse.FloatTensor(adj.indices(), new_values, adj.size()).coalesce()



def symmetrize(adj):  # only for non-sparse
    return (adj + adj.T) / 2


def cal_similarity_graph(node_embeddings):
    similarity_graph = torch.mm(node_embeddings, node_embeddings.t())
    return similarity_graph


def top_k(raw_graph, K):
    values, indices = raw_graph.topk(k=int(K), dim=-1)
    assert torch.max(indices) < raw_graph.shape[1]
    device = raw_graph.device if hasattr(raw_graph, 'device') else (torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu'))
    mask = torch.zeros(raw_graph.shape).to(device)
    mask[torch.arange(raw_graph.shape[0]).view(-1, 1), indices] = 1.

    mask.requires_grad = False
    sparse_graph = raw_graph * mask
    return sparse_graph


def knn_fast(X, k, b):
    X = F.normalize(X, dim=1, p=2)
    index = 0
    device = X.device if hasattr(X, 'device') else (torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu'))
    values = torch.zeros(X.shape[0] * (k + 1)).to(device)
    rows = torch.zeros(X.shape[0] * (k + 1)).to(device)
    cols = torch.zeros(X.shape[0] * (k + 1)).to(device)
    norm_row = torch.zeros(X.shape[0]).to(device)
    norm_col = torch.zeros(X.shape[0]).to(device)
    while index < X.shape[0]:
        if (index + b) > (X.shape[0]):
            end = X.shape[0]
        else:
            end = index + b
        sub_tensor = X[index:index + b]
        similarities = torch.mm(sub_tensor, X.t())
        vals, inds = similarities.topk(k=k + 1, dim=-1)
        values[index * (k + 1):(end) * (k + 1)] = vals.view(-1)
        cols[index * (k + 1):(end) * (k + 1)] = inds.view(-1)
        rows[index * (k + 1):(end) * (k + 1)] = torch.arange(index, end).view(-1, 1).repeat(1, k + 1).view(-1)
        norm_row[index: end] = torch.sum(vals, dim=1)
        norm_col.index_add_(-1, inds.view(-1), vals.view(-1))
        index += b
    norm = norm_row + norm_col
    rows = rows.long()
    cols = cols.long()
    values *= (torch.pow(norm[rows], -0.5) * torch.pow(norm[cols], -0.5))
    return rows, cols, values



def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def torch_sparse_to_dgl_graph(torch_sparse_mx):
    torch_sparse_mx = torch_sparse_mx.coalesce()
    indices = torch_sparse_mx.indices()
    values = torch_sparse_mx.values()
    rows_, cols_ = indices[0,:], indices[1,:]
    device = values.device if hasattr(values, 'device') else (torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu'))
    dgl_graph = dgl.graph((rows_, cols_), num_nodes=torch_sparse_mx.shape[0], device=device)
    dgl_graph.edata['w'] = values.detach().to(device)
    return dgl_graph


def dgl_graph_to_torch_sparse(dgl_graph):
    values = dgl_graph.edata['w'].cpu().detach()
    rows_, cols_ = dgl_graph.edges()
    indices = torch.cat((torch.unsqueeze(rows_, 0), torch.unsqueeze(cols_, 0)), 0).cpu()
    torch_sparse_mx = torch.sparse.FloatTensor(indices, values)
    return torch_sparse_mx


def remove_self_loop(adjs):
    adjs_ = []
    for i in range(len(adjs)):
        adj = adjs[i].coalesce()
        non_diag_index = torch.nonzero(adj.indices()[0] != adj.indices()[1]).flatten()
        adj = torch.sparse.FloatTensor(adj.indices()[:, non_diag_index], adj.values()[non_diag_index], adj.shape).coalesce()
        adjs_.append(adj)
    return adjs_


def get_sparse_diag(adj):
    adj = adj.coalesce()
    diag_index = torch.nonzero(adj.indices()[0] == adj.indices()[1]).flatten()
    values = adj.values()[diag_index]
    indices = adj.indices()[:, diag_index]
    return indices, values


def sparse_tensor_add_self_loop(adj):
    adj = adj.coalesce()
    node_num = adj.shape[0]
    index = torch.stack((torch.tensor(range(node_num)), torch.tensor(range(node_num))), dim=0).to(adj.device)
    values = torch.ones(node_num).to(adj.device)

    adj_new = torch.sparse.FloatTensor(torch.cat((index, adj.indices()), dim=1), torch.cat((values, adj.values()),dim=0), adj.shape)
    return adj_new.coalesce()

def torch_sparse_eye(num_nodes):
    indices = torch.arange(num_nodes).repeat(2, 1)
    values = torch.ones(num_nodes)
    return torch.sparse.FloatTensor(indices, values)

def adj_values_one(adj):
    adj = adj.coalesce()
    index = adj.indices()
    return torch.sparse.FloatTensor(index, torch.ones(len(index[0])).to(adj.device), adj.shape).coalesce()


class clustering_metrics():
    def __init__(self, true_label, predict_label):
        self.true_label = true_label
        self.pred_label = predict_label

    def clusteringAcc(self):
        # best mapping between true_label and predict label
        l1 = list(set(self.true_label))
        numclass1 = len(l1)

        l2 = list(set(self.pred_label))
        numclass2 = len(l2)
        if numclass1 != numclass2:
            print(numclass1, numclass2)
            print(self.pred_label)
            print(self.pred_label.max(), self.pred_label.min())
            print('Class Not equal, Error!!!!')
            return 0, 0, 0, 0, 0, 0, 0

        cost = np.zeros((numclass1, numclass2), dtype=int)
        for i, c1 in enumerate(l1):
            mps = [i1 for i1, e1 in enumerate(self.true_label) if e1 == c1]
            for j, c2 in enumerate(l2):
                mps_d = [i1 for i1 in mps if self.pred_label[i1] == c2]

                cost[i][j] = len(mps_d)

        # match two clustering results by Munkres algorithm
        m = Munkres()
        cost = cost.__neg__().tolist()

        indexes = m.compute(cost)

        # get the match results
        new_predict = np.zeros(len(self.pred_label))
        for i, c in enumerate(l1):
            # correponding label in l2:
            c2 = l2[indexes[i][1]]

            # ai is the index with label==c2 in the pred_label list
            ai = [ind for ind, elm in enumerate(self.pred_label) if elm == c2]
            new_predict[ai] = c

        acc = metrics.accuracy_score(self.true_label, new_predict)
        f1_macro = metrics.f1_score(self.true_label, new_predict, average='macro')
        precision_macro = metrics.precision_score(self.true_label, new_predict, average='macro')
        recall_macro = metrics.recall_score(self.true_label, new_predict, average='macro')
        f1_micro = metrics.f1_score(self.true_label, new_predict, average='micro')
        precision_micro = metrics.precision_score(self.true_label, new_predict, average='micro')
        recall_micro = metrics.recall_score(self.true_label, new_predict, average='micro')
        return acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro

    def evaluationClusterModelFromLabel(self, print_results=True):
        nmi = metrics.normalized_mutual_info_score(self.true_label, self.pred_label)
        adjscore = metrics.adjusted_rand_score(self.true_label, self.pred_label)
        acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro = self.clusteringAcc()

        if print_results:
            print('ACC={:.4f}, f1_macro={:.4f}, precision_macro={:.4f}, recall_macro={:.4f}, f1_micro={:.4f}, '
                  .format(acc, f1_macro, precision_macro, recall_macro, f1_micro) +
                  'precision_micro={:.4f}, recall_micro={:.4f}, NMI={:.4f}, ADJ_RAND_SCORE={:.4f}'
                  .format(precision_micro, recall_micro, nmi, adjscore))

        return acc, nmi, f1_macro, adjscore
#这个是在处理的时候KNN稀疏化 加上对称归一处理 最后中间对角线上的都化为0
def topk_sparsify_sym_row_normalize(S: np.ndarray, k: int):
    C = S.shape[0]
    if k >= C:
        S = 0.5 * (S + S.T)
        np.fill_diagonal(S, 0.0)
        S = np.maximum(S, 0.0)
        row_sum = S.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        return S / row_sum
    S_out = np.zeros_like(S, dtype=float)
    for i in range(C):
        row = S[i]
        idx = np.argpartition(-row, kth=min(k, C - 1))[:k]
        S_out[i, idx] = row[idx]
    S_out = 0.5 * (S_out + S_out.T)
    np.fill_diagonal(S_out, 0.0)
    S_out = np.maximum(S_out, 0.0)
    row_sum = S_out.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return S_out / row_sum

def safe_makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def load_feature_pickle(feature_path, split):
    p = os.path.join(feature_path, f"{split}_data.pkl")
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} not found")
    with open(p, 'rb') as f:
        data = pickle.load(f)
    return data


def prepare_rows_from_feature_data(feature_data, feature_type):
    rows = []
    if feature_type == 'both':
        sample_ids = list(feature_data['oxy'].keys())
    else:
        sample_ids = list(feature_data[feature_type].keys())
    for sid in sample_ids:
        label = feature_data['labels'].get(sid, None)
        if label is None:
            continue
        if feature_type == 'oxy':
            ts = feature_data['oxy'].get(sid)
        elif feature_type == 'dxy':
            ts = feature_data['dxy'].get(sid)
        else:
            oxy = feature_data['oxy'].get(sid)
            dxy = feature_data['dxy'].get(sid)
            if (oxy is None) or (dxy is None):
                continue
            ts = np.concatenate([oxy, dxy], axis=-1)

        if isinstance(ts, torch.Tensor):
            ts = ts.cpu().numpy()

        rows.append({'id': sid, 'data': ts, 'labels': int(label)})
    return rows
#这个就是用来构建RBF核的 输入的是一个二维张量
def rbf_gram_torch(x, sigma=None):
    device = x.device
    dists = torch.cdist(x, x, p=2.0)  # (N,N)
    d2 = dists ** 2
    if sigma is None:
        n = x.size(0)
        if n * n - n <= 0:
            sigma = 1.0
        else:
            triu_inds = torch.triu_indices(n, n, offset=1)
            vals = d2[triu_inds[0], triu_inds[1]]
            nz = vals[vals > 0]
            if nz.numel() == 0:
                sigma = 1.0
            else:
                med = torch.median(nz)
                sigma = torch.sqrt(med + 1e-8)
                if sigma <= 0:
                    sigma = 1.0
    denom = 2.0 * (sigma ** 2) + 1e-12
    K = torch.exp(-d2 / denom)
    return K

def linear_gram_torch(x):
    return x @ x.t()
#对核矩阵中心化 然后让它的均值为零
def center_gram_torch(K):
    n = K.size(0)
    device = K.device
    ones = torch.ones((n, n), device=device) / n
    H = torch.eye(n, device=device) - ones
    Kc = H @ K @ H
    return Kc
#计算两个矩阵之间的HSIC 值
def hsic_from_grams_torch(K, L, normalize=False):
    Kc = center_gram_torch(K)
    Lc = center_gram_torch(L)
    n = K.size(0)
    denom = (n - 1) ** 2 if n > 1 else 1.0
    val = torch.trace(Kc @ Lc) / denom
    if normalize:
        denom2 = torch.norm(Kc) * torch.norm(Lc) + 1e-12
        return val / denom2
    return val
#这个就是比较的接口
def hsic_torch(x, y, kernel='rbf', sigma=None, normalize=False):
    if kernel == 'rbf':
        K = rbf_gram_torch(x, sigma)
        L = rbf_gram_torch(y, sigma)
    elif kernel == 'linear':
        K = linear_gram_torch(x)
        L = linear_gram_torch(y)
    else:
        raise ValueError("Unknown kernel")
    return hsic_from_grams_torch(K, L, normalize=normalize)

def symmetrize_and_normalize(A):
    if A.dim() == 2:
        A = A.unsqueeze(0)
        is_single = True
    else:
        is_single = False
        
    B, N, _ = A.shape
    A = 0.5 * (A + A.transpose(1, 2))
    identity = torch.eye(N, device=A.device, dtype=torch.bool).unsqueeze(0).expand(B, N, N)
    A = A.masked_fill(identity, 0)
    A = torch.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    A = A.clamp(min=0.0)
    row_sum = A.sum(dim=2, keepdim=True)
    row_sum[row_sum == 0] = 1.0 
    A_normalized = A / (row_sum + 1e-12) # EOS
    return A_normalized.squeeze(0) if is_single else A_normalized
def graph_to_dense_if_needed(g, device):
    try:
        is_dgl = hasattr(g, 'num_nodes') and hasattr(g, 'edges') and hasattr(g, 'edata')
    except Exception:
        is_dgl = False
    if is_dgl:
        import dgl 
        rows, cols = g.edges()
        w = g.edata.get('w', torch.ones(rows.shape[0], device=rows.device))
        n = g.num_nodes()
        A = torch.zeros((n, n), device=device, dtype=torch.float32)
        A[rows.to(device), cols.to(device)] = w.to(device)
        return A
    if isinstance(g, np.ndarray):
        return torch.from_numpy(g).float().to(device)
    return g.to(device)