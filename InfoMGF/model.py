import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import dgl
from graph_learners import *
from layers import GCNConv_dense, GCNConv_dgl
from torch.nn import Sequential, Linear, ReLU
from utils import *
# 核函数/Gram矩阵
EPS = 1e-10


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

class GCL(nn.Module):
    def __init__(self, nlayers, in_dim, hidden_dim, emb_dim, proj_dim, dropout, sparse, num_g, 
                 num_classes:int,
                 pool:str='mean',
                 kernel_type='rbf', normalize_hsic=True):
        super(GCL, self).__init__()

        self.num_g = num_g
        self.kernel_type = kernel_type
        self.normalize_hsic = normalize_hsic
        self.encoder = GraphEncoder(nlayers, in_dim, hidden_dim, emb_dim, dropout, sparse)

        self.proj_s = nn.ModuleList([nn.Sequential(
            nn.Linear(emb_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim)
        ) for _ in range(self.num_g)])

        self.proj_u = nn.ModuleList([nn.Sequential(
            nn.Linear(emb_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim)
        ) for _ in range(self.num_g)])

        self.proj_f = nn.ModuleList([nn.Sequential(
            nn.Linear(emb_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim)
        ) for _ in range(self.num_g)])
        
        self.pool_name=pool
        if self.pool_name == 'mean':
            self.pool_fn = lambda x: torch.mean(x, dim=0)
        elif self.pool_name == 'sum':
            self.pool_fn = lambda x: torch.sum(x, dim=0)
        else:
            raise ValueError(f"Unsupported pooling method: {self.pool_name}")
        self.classifier = nn.Linear(emb_dim, num_classes)
    def forward(self, x, Adj_):
        embedding = self.encoder(x, Adj_)
        embedding = F.normalize(embedding, dim=1, p=2)
        return embedding
    def forward_classify(self,x,adj):
        node_embeddings=self.encoder(x,adj)
        graph_embedding=self.pool_fn(node_embeddings)
        logits=self.classifier(graph_embedding)
        return logits
    def cal_loss(self, z_specific_adjs, z_aug_adjs, z_fused_adj, sigma=None):
        """
        使用HSIC计算损失
        z_specific_adjs: list of tensors, each (n_nodes, emb_dim)
        z_aug_adjs: list of tensors, each (n_nodes, emb_dim)  (augmented)
        z_fused_adj: tensor (n_nodes, emb_dim)  (shared fused embedding)
        Returns: scalar loss (to minimize). We use negative HSIC (so minimization maximizes dependence).
        """
        device = z_fused_adj.device
        n = z_fused_adj.size(0)
        g = len(z_specific_adjs)
        
        # projections
        z_proj_s = [self.proj_s[i](z_specific_adjs[i]) for i in range(g)]
        z_proj_u = [self.proj_u[i](z_aug_adjs[i]) for i in range(g)]
        z_proj_f = [self.proj_f[i](z_fused_adj) for i in range(g)]

        # L2 normalize projections (helps kernel stability)
        z_proj_s = [F.normalize(z, dim=1, p=2) for z in z_proj_s]
        z_proj_u = [F.normalize(z, dim=1, p=2) for z in z_proj_u]
        z_proj_f = [F.normalize(z, dim=1, p=2) for z in z_proj_f]

        # compute HSIC terms
        # 1) specific-specific
        loss_smi = 0.0
        cnt = 0
        for i in range(g):
            for j in range(i + 1, g):
                hs = hsic_torch(z_proj_s[i], z_proj_s[j], kernel=self.kernel_type, sigma=sigma, normalize=self.normalize_hsic)
                loss_smi += (-hs)  # negative because we minimize
                cnt += 1
        if cnt > 0:
            loss_smi = loss_smi / cnt
        else:
            loss_smi = torch.tensor(0., device=device)

        # 2) fused vs specific
        loss_fused = 0.0
        for i in range(g):
            hs = hsic_torch(z_proj_f[i], z_proj_s[i], kernel=self.kernel_type, sigma=sigma, normalize=self.normalize_hsic)
            loss_fused += (-hs)
        loss_fused = loss_fused / g

        # 3) specific vs augmented
        loss_umi = 0.0
        for i in range(g):
            hs = hsic_torch(z_proj_s[i], z_proj_u[i], kernel=self.kernel_type, sigma=sigma, normalize=self.normalize_hsic)
            loss_umi += (-hs)
        loss_umi = loss_umi / g

        loss = loss_fused + loss_smi + loss_umi
        return loss

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

def sim_con(z1, z2, temperature):
    z1_norm = torch.norm(z1, dim=-1, keepdim=True)
    z2_norm = torch.norm(z2, dim=-1, keepdim=True)
    dot_numerator = torch.mm(z1, z2.t())
    dot_denominator = torch.mm(z1_norm, z2_norm.t()) + EPS
    sim_matrix = dot_numerator / dot_denominator / temperature
    return sim_matrix

def sce_loss(x, y, beta=1):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    loss = (1 - (x * y).sum(dim=-1)).pow_(beta)

    loss = loss.mean()
    return loss


