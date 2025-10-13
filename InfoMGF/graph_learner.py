import dgl
import torch
import torch.nn as nn
from layers import GCNConv_dense
from layers import Attentive
from utils import *
import math
class GraphLearnerGCN(nn.Module):
    def __init__(self, gcn_input_dim, gcn_hidden_dim, gcn_output_dim, k=5, dropedge_rate=0.2, sparse=False, act="relu"):
        super(GraphLearnerGCN, self).__init__()
        self.gcn_layers = nn.ModuleList([
            GCNConv_dense(input_size=gcn_input_dim, output_size=gcn_hidden_dim),
            GCNConv_dense(input_size=gcn_hidden_dim, output_size=gcn_output_dim)
        ])
        self.gcn_act = act
        self.k = k
        self.non_linearity = "relu"
        self.sparse = sparse
        self.dropedge_rate = dropedge_rate

        for layer in self.gcn_layers:
            if hasattr(layer, 'linear'):
                torch.nn.init.xavier_uniform_(layer.linear.weight)
                if layer.linear.bias is not None:
                    torch.nn.init.constant_(layer.linear.bias, 0)
    
    def forward(self, x, init_adj):

        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
        
        h = x
        for i, layer in enumerate(self.gcn_layers):
            if isinstance(init_adj, dgl.DGLGraph):
                h = layer(h, init_adj)
            else:
                h = layer(h, init_adj, sparse=self.sparse)

            h = torch.nan_to_num(h, nan=0.0, posinf=1.0, neginf=0.0)
            
            if i != len(self.gcn_layers) - 1:
                if self.gcn_act == "relu":
                    h = F.relu(h)
                elif self.gcn_act == "tanh":
                    h = torch.tanh(h)
                elif self.gcn_act == "sigmoid":
                    h = torch.sigmoid(h)
        
        return h
    
    def graph_process(self, embeddings):

        embeddings = torch.nan_to_num(embeddings, nan=0.0, posinf=1.0, neginf=0.0)
        emb = F.normalize(embeddings, p=2, dim=1)
        emb = torch.nan_to_num(emb, nan=0.0, posinf=1.0, neginf=0.0)
  
        d = max(1.0, emb.size(1))
        sim = torch.matmul(emb, emb.t()) / math.sqrt(d)
        sim = torch.clamp(sim, min=-50, max=50)
        mask = torch.eye(sim.size(0), device=sim.device, dtype=torch.bool)
        sim = sim.masked_fill(mask, float('-inf'))
        

        sim_max = torch.max(sim, dim=1, keepdim=True)[0]
        sim_exp = torch.exp(sim - sim_max)
        sim_sum = torch.sum(sim_exp, dim=1, keepdim=True)
        attn = sim_exp / (sim_sum + 1e-12)

        if (self.k is not None) and (0 < self.k < attn.size(1)):
            topk_vals, topk_idx = torch.topk(attn, self.k, dim=1)
            
            mask = torch.zeros_like(attn)
            mask = mask.scatter(1, topk_idx, 1.0)
            
            attn = attn * mask
            row_sum = attn.sum(dim=1, keepdim=True)
            attn = attn / (row_sum + 1e-12)

        attn = (attn + attn.t()) / 2.0

        if self.non_linearity == 'relu':
            attn = F.relu(attn)
        elif self.non_linearity == 'tanh':
            attn = torch.tanh(attn)
        elif self.non_linearity == 'sigmoid':
            attn = torch.sigmoid(attn)

        attn = torch.nan_to_num(attn, nan=0.0, posinf=1.0, neginf=0.0)

        if self.training:
            dropout_mask = torch.rand_like(attn) > self.dropedge_rate
            learned_adj = attn * dropout_mask.float()
        else:
            learned_adj = attn
        learned_adj = torch.nan_to_num(learned_adj, nan=0.0, posinf=1.0, neginf=0.0)
        
        return learned_adj

class AttentionFusion(nn.Module):
    def __init__(self, input_dim, num_views=2):
        super(AttentionFusion, self).__init__()
        self.num_views = num_views
        self.attention_mlp = nn.Sequential(
            nn.Linear(input_dim * num_views, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, num_views)
        )
        
    def forward(self, view_features):

        concatenated = torch.cat(view_features, dim=1)  
        scores = self.attention_mlp(concatenated)  
        attention_weights = F.softmax(scores, dim=1)  
        fused_feature = torch.zeros_like(view_features[0])
        for i in range(self.num_views):
            weight = attention_weights[:, i].unsqueeze(1)
            fused_feature += weight * view_features[i]
        
        return fused_feature, attention_weights