import copy
import math
import dgl
import torch
from graph_learner import *
from layers import GCNConv_dense, GCNConv_dgl
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_dense_adj,dense_to_sparse 
EPS = 1e-12
def _adj_to_dense(adj, n_nodes=None, device=None):
    """
    支持输入：
      - DGLGraph (dgl.DGLGraph / DGLHeteroGraph)
      - torch.sparse_coo_tensor
      - torch.Tensor (dense)
      - edge_index (Tensor 2 x E) 或 tuple (edge_index, num_nodes)
    返回 dense torch.FloatTensor (n_nodes x n_nodes) 在指定 device 上。
    """
    # DGL 图
    if isinstance(adj, (dgl.DGLGraph, dgl.DGLHeteroGraph)):
        g = adj
        n = n_nodes if n_nodes is not None else g.num_nodes()
        device = device if device is not None else (g.device if hasattr(g, 'device') else torch.device('cpu'))
        # 若有边权 'w' 则用它，否则用 1
        if 'w' in g.edata:
            vals = g.edata['w'].to(device)
        else:
            u, v = g.edges()
            vals = torch.ones(u.shape[0], device=device)
        u, v = g.edges()
        indices = torch.stack([u, v], dim=0).to(device)
        A_sparse = torch.sparse_coo_tensor(indices, vals, (n, n), device=device).coalesce()
        return A_sparse.to_dense().float()

    # torch.sparse_coo_tensor
    if isinstance(adj, torch.Tensor) and adj.is_sparse:
        device = device if device is not None else adj.device
        return adj.coalesce().to_dense().to(device).float()

    # dense torch tensor
    if isinstance(adj, torch.Tensor):
        device = device if device is not None else adj.device
        return adj.to(device).float()

    # edge_index 2xE
    if isinstance(adj, (list, tuple)) and len(adj) >= 1:
        edge_index = adj[0]
        if isinstance(edge_index, torch.Tensor) and edge_index.dim() == 2:
            if len(adj) == 2 and adj[1] is not None:
                n = adj[1]
            else:
                # infer n from max index
                n = int(edge_index.max().item()) + 1
            device = device if device is not None else edge_index.device
            rows = edge_index[0].to(device)
            cols = edge_index[1].to(device)
            vals = torch.ones(rows.size(0), device=device)
            idx = torch.stack([rows, cols], dim=0)
            A_sparse = torch.sparse_coo_tensor(idx, vals, (n, n), device=device).coalesce()
            return A_sparse.to_dense().float()

    raise TypeError(f"Unsupported adj type: {type(adj)}. Provide DGLGraph / sparse_coo / dense tensor / (edge_index, n_nodes).")
# GCN for evaluation.
class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, dropout_adj, Adj, sparse):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()

        if sparse:
            self.layers.append(GCNConv_dgl(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.layers.append(GCNConv_dgl(hidden_channels, hidden_channels))
            self.layers.append(GCNConv_dgl(hidden_channels, out_channels))
        else:
            self.layers.append(GCNConv_dense(in_channels, hidden_channels))
            for i in range(num_layers - 2):
                self.layers.append(GCNConv_dense(hidden_channels, hidden_channels))
            self.layers.append(GCNConv_dense(hidden_channels, out_channels))
        self.dropout = dropout
        self.Adj = Adj
        self.Adj.requires_grad = False
        self.dropout_adj = nn.Dropout(p=dropout_adj)
        self.sparse = sparse

    def forward(self, x):
        Adj = copy.deepcopy(self.Adj)
        if self.sparse:
            Adj.edata['w'] = self.dropout_adj(Adj.edata['w'])
        else:
            Adj = self.dropout_adj(Adj)

        for i, conv in enumerate(self.layers[:-1]):
            x = conv(x, Adj)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, Adj)
        return x


class GraphEncoder(nn.Module):
    def __init__(self, nlayers, in_dim, hidden_dim, emb_dim, dropout, sparse):

        super(GraphEncoder, self).__init__()
        self.dropout = dropout
        self.gnn_encoder_layers = nn.ModuleList()
        self.act = nn.ReLU()
        self.num_g = 2  
        if sparse:
            self.gnn_encoder_layers.append(GCNConv_dgl(in_dim, hidden_dim))
            for _ in range(nlayers - 2):
                self.gnn_encoder_layers.append(GCNConv_dgl(hidden_dim, hidden_dim))
            self.gnn_encoder_layers.append(GCNConv_dgl(hidden_dim, emb_dim))
        else:
            self.gnn_encoder_layers.append(GCNConv_dense(in_dim, hidden_dim))
            for _ in range(nlayers - 2):
                self.gnn_encoder_layers.append(GCNConv_dense(hidden_dim, hidden_dim))
            self.gnn_encoder_layers.append(GCNConv_dense(hidden_dim, emb_dim))
        self.sparse = sparse

    def forward(self, x, Adj):

        x = F.dropout(x, p=self.dropout, training=self.training)
        for conv in self.gnn_encoder_layers[:-1]:
            x = conv(x, Adj)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.gnn_encoder_layers[-1](x, Adj)
        return x
    def cal_custom_loss(self, z_specific_adjs, z_fused_adj, specific_adjs, 
                       temperature=1.0, h=1.0, alpha=1.0, beta=1.0, gamma=1.0):

        
        # 1. LFD损失: 融合视图与局部邻居聚合的对齐
        lfd_loss_total = 0
        for i in range(self.num_g):
            lfd_loss = compute_lfd_loss_optimized(
                z_teacher=z_specific_adjs[i],
                h_student=z_fused_adj,
                adj_teacher=specific_adjs[i],
                temperature=temperature
            )
            lfd_loss_total += lfd_loss
        lfd_loss_avg = lfd_loss_total / self.num_g
        
        s_high_values=[]
        for i in range(self.num_g):
            for j in range(self.num_g):
                if i==j:
                    continue
                s_high_value=compute_s_high(z_specific_adjs[i],specific_adjs[j])
                s_high_values.append(s_high_value)
        s_high_loss_avg=torch.abs(s_high_values[0]-s_high_values[1])
        # 3. SC损失: 不同视图间的分布相似性
        sc_loss_total = 0
        count = 0
        
        # 特定视图之间的SC损失
        for i in range(self.num_g):
            for j in range(i+1, self.num_g):
                sc_loss = compute_sc_loss(
                    Z_s_intra=z_specific_adjs[i],
                    Z_s_global=z_specific_adjs[j],
                    h=h
                )
                sc_loss_total += sc_loss
                count += 1
        
        sc_loss_avg = sc_loss_total / count if count > 0 else torch.tensor(0.0)
        

        total_loss = (alpha * lfd_loss_avg + 
                     beta * s_high_loss_avg + 
                     gamma * sc_loss_avg)
        
        loss_details = {
            'lfd_loss': lfd_loss_avg,
            's_high_loss': s_high_loss_avg,
            'sc_loss': sc_loss_avg,
            'total_loss': total_loss
        }
        return total_loss, loss_details
        
class GraphEncoderWithPooling(nn.Module):
    def __init__(self, nlayers, in_dim, hidden_dim, emb_dim, dropout, sparse):
        super().__init__()
        self.encoder = GraphEncoder(nlayers, in_dim, hidden_dim, emb_dim, dropout, sparse)
    def forward(self, x, adj,batch_vec):
        node_embedding=self.encoder(x,adj)
        node_embedding=F.normalize(node_embedding,dim=1,p=2)
        graph_embedding=global_mean_pool(node_embedding,batch_vec)
        return graph_embedding

class GraphClassifierHead(torch.nn.Module):
    def __init__(self, in_dim, nclasses):
        super().__init__()
        self.linear = torch.nn.Linear(in_dim, nclasses)
    def forward(self, graph_emb):
        return self.linear(graph_emb)
#用在最开始的图上  SGC
def AGG(h_list, adjs_o, nlayer, sparse=False):
    f_list = []
    for i in range(len(adjs_o)):
        z = h_list[i]
        adj = adjs_o[i]
        for i in range(nlayer):
            if sparse:
                z = torch.sparse.mm(adj, z)
            else:
                z = torch.matmul(adj, z)
        z = F.normalize(z, dim=1, p=2)
        f_list.append(z)

    return f_list
#这个是对于融合前后一个点将融合前的一个点聚合它的邻居的特征值然后取平均值用kl散度取拉近这两者之间的距离
def compute_lfd_loss_optimized(z_teacher, h_student, adj_teacher, temperature=1.0):
        device = z_teacher.device
        num_nodes = z_teacher.shape[0]
        adj_teacher = adj_teacher.float()
        if adj_teacher.is_sparse:
            indices = adj_teacher.coalesce().indices()
            values = adj_teacher.coalesce().values()
            self_loop_indices = torch.arange(0, num_nodes, device=device).repeat(2, 1)
            self_loop_values = torch.ones(num_nodes, device=device)

            all_indices = torch.cat([indices, self_loop_indices], dim=1)
            all_values = torch.cat([values, self_loop_values])
            adj_plus_self_loop = torch.sparse_coo_tensor(
                all_indices, all_values, (num_nodes, num_nodes)
            ).coalesce()
        else:
            adj_plus_self_loop = adj_teacher + torch.eye(num_nodes, device=device)
        if adj_plus_self_loop.is_sparse:
            row, col = adj_plus_self_loop.indices()
            deg = torch.zeros(num_nodes, device=device)
            deg = deg.scatter_add(0, row, adj_plus_self_loop.values())
            deg_inv_sqrt = torch.where(deg > 0, deg.pow(-0.5), torch.zeros_like(deg))
            norm_values = deg_inv_sqrt[row] * adj_plus_self_loop.values() * deg_inv_sqrt[col]
            adj_normalized = torch.sparse_coo_tensor(
                adj_plus_self_loop.indices(), norm_values, adj_plus_self_loop.size()
            )
            z_teacher_agg = torch.sparse.mm(adj_normalized, z_teacher)
        else:

            deg = torch.diag(adj_plus_self_loop.sum(dim=1))
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0
            adj_normalized = torch.mm(torch.mm(deg_inv_sqrt, adj_plus_self_loop), deg_inv_sqrt)
            z_teacher_agg = torch.mm(adj_normalized, z_teacher)

        p_teacher = F.softmax(z_teacher_agg / temperature, dim=1).detach()
        log_p_student = F.log_softmax(h_student / temperature, dim=1)

        loss = F.kl_div(log_p_student, p_teacher, reduction='batchmean', log_target=False)
        loss = loss * (temperature ** 2)      
        return loss
    #下面这个是对于S_high的计算，然后它的原理是让内部的图信号（经过卷积得到的节点嵌入）
    #去放到全局视图上去评估 同样的交叉操作一下
    #这里是到loss的时候要进行相减的操作
def compute_s_high(x, adj):
    """
    计算图信号的高频能量度量（S_high）。
    x: (N, F) 节点嵌入 torch.Tensor
    adj: DGLGraph / sparse coo / dense tensor / (edge_index, n_nodes)
    返回 scalar torch.Tensor
    """
    if not torch.is_tensor(x):
        x = torch.tensor(x).float()

    device = x.device
    N = x.size(0)

    A = _adj_to_dense(adj, n_nodes=N, device=device)  # (N,N) float
    # 如果 A 全零（孤立图），返回 0 防止数值异常
    if torch.allclose(A, torch.zeros_like(A)):
        return torch.tensor(0.0, device=device)

    # 归一化对称拉普拉斯 L = I - D^{-1/2} A D^{-1/2}
    deg = A.sum(dim=1)  # (N,)
    deg_inv_sqrt = torch.where(deg > 0, deg.pow(-0.5), torch.zeros_like(deg))
    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    I = torch.eye(N, device=device, dtype=A_norm.dtype)
    L = I - A_norm  # (N,N)

    # numerator per feature: x_f^T L x_f
    XLX = x.t() @ L @ x  # (F, F)
    numerator = torch.diag(XLX)  # (F,)
    denom_mat = x.t() @ x  # (F, F)
    denominator = torch.diag(denom_mat)  # (F,)

    s_high_vals = numerator / (denominator + EPS)
    s_high = s_high_vals.mean()
    return s_high
    #下面是第三个损失函数用在不同的视图的节点嵌入之间整体分布形状是否相似
'''
# Z_s_intra = gnn_encoder(X_s_intra, A_s_intra)
# Z_s_global = gnn_encoder(X_s_global, A_s_global)

# 计算 SC 损失
sc_loss = compute_sc_loss(Z_s_intra, Z_s_global, h=1.0)
'''
def gaussian_kernel(x, y, h):
    """
    返回 (Nx, Ny) 的高斯核相似度矩阵
    """
    x = x.float()
    y = y.float()
    x_norm = (x ** 2).sum(dim=1).unsqueeze(1)  # (Nx,1)
    y_norm = (y ** 2).sum(dim=1).unsqueeze(0)  # (1,Ny)
    dist_sq = x_norm + y_norm - 2.0 * (x @ y.t())
    dist_sq = torch.clamp(dist_sq, min=0.0)
    K = torch.exp(-dist_sq / (2.0 * (h ** 2 + EPS)))
    return K

def js_divergence(P, Q):
    """
    P, Q: (1, M) 或 (M,) 估计分布（未必归一化）
    返回：标量 JS 散度
    """
    P = P / (P.sum(dim=-1, keepdim=True) + EPS)
    Q = Q / (Q.sum(dim=-1, keepdim=True) + EPS)
    M = 0.5 * (P + Q)
    # 使用 KL-div，先保证无零
    kl_pm = F.kl_div((P + EPS).log(), M, reduction='batchmean')
    kl_qm = F.kl_div((Q + EPS).log(), M, reduction='batchmean')
    return 0.5 * (kl_pm + kl_qm)

def compute_sc_loss(Z_s_intra, Z_s_global, h=1.0):
    """
    计算两个视图嵌入分布的分布相似性（JS divergence of kernel estimates）
    """
    K_intra_global = gaussian_kernel(Z_s_intra, Z_s_global, h)  # (N,N)
    K_global_intra = gaussian_kernel(Z_s_global, Z_s_intra, h)  # (N,N)
    P_est = K_intra_global.mean(dim=0, keepdim=True)  # (1, N)
    Q_est = K_global_intra.mean(dim=0, keepdim=True)  # (1, N)
    sc_loss = js_divergence(P_est, Q_est)
    return sc_loss
