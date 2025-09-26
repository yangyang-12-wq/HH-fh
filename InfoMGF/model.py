import copy
import math
import dgl
import torch
from graph_learner import *
from layers import GCNConv_dense, GCNConv_dgl
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_laplacian,to_dense_adj


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
    def cal_custom_loss(self, z_specific_adjs, z_fused_adj, specific_adjs, fused_adj, 
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
        
class GraphEncoderWithPooling(nn.Moudle):
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
def compute_s_high(x,adj):
        num_nodes=x.size(0)
        edge_index,_=torch_geometric.utils.dense_to_sparse(adj)
        laplacian_edge_index, laplacian_edge_weight = to_laplacian(edge_index, num_nodes=num_nodes, normalization='sym')
        L = to_dense_adj(laplacian_edge_index, edge_attr=laplacian_edge_weight)
        if x.dim() == 1:
            x = x.unsqueeze(1)
        numerator = torch.einsum('nf, nm, mf -> f', x.t(), L.squeeze(0), x)
        denominator = torch.einsum('nf, nf -> f', x.t(), x)
        s_high_values = numerator / (denominator + 1e-8) # 加一个小的常数防止除以0
        s_high = s_high_values.mean() # 对所有特征维度求平均
        return s_high
    #下面是第三个损失函数用在不同的视图的节点嵌入之间整体分布形状是否相似
'''
# Z_s_intra = gnn_encoder(X_s_intra, A_s_intra)
# Z_s_global = gnn_encoder(X_s_global, A_s_global)

# 计算 SC 损失
sc_loss = compute_sc_loss(Z_s_intra, Z_s_global, h=1.0)
'''
def gaussian_kernel(x, y, h):
        x, y = x.to(y.device), y.to(x.device)
        diff = x.unsqueeze(1) - y.unsqueeze(0)
        dist_sq = torch.sum(diff ** 2, dim=-1)
        return torch.exp(-dist_sq / (2 * h ** 2))
def js_divergence(P,Q):
        P=P/P.sum(dim=-1,keepdim=True)
        Q=Q/Q.sum(dim=-1,keepdim=True)
        M=0.5*(P+Q)
        kl_pm=F.kl_div(P.log(),M,reduction='batchmean')
        kl_qm=F.kl_div(Q.log(),M,reduction='batchmean')
        return 0.5*(kl_pm+kl_qm)
def compute_sc_loss(Z_s_intra,Z_s_global,h=1.0):
        device=Z_s_intra
        Z_s_global=Z_s_intra.to(device)
        K_intra_global=gaussian_kernel(Z_s_intra,Z_s_global,h)
        k_global_intra=gaussian_kernel(Z_s_global,Z_s_intra,h)
        P_est=K_intra_global.mean(dim=0)
        Q_est=k_global_intra.mean(dim=0)
        sc_loss=js_divergence(P_est,Q_est)
        return sc_loss
