import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import dgl
from graph_learner import *
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
        return logits.squeeze(0)
    def forward_all(self,features,adj):
        embedding=self.forward(features,adj)
        logits=self.forward_classify()



