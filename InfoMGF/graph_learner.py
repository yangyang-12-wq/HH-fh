import dgl
import torch
import torch.nn as nn

from layers import Attentive
from utils import *
import math
class ATT_learner(nn.Module):
    def __init__(self, nlayers, isize, k, i, dropedge_rate, sparse, act):
        super(ATT_learner, self).__init__()

        self.layers = nn.ModuleList()
        for _ in range(nlayers):
            self.layers.append(Attentive(isize))

        self.k = k
        self.non_linearity = 'relu'
        self.i = i
        self.sparse = sparse
        self.act = act
        self.dropedge_rate = dropedge_rate

    def internal_forward(self, h):
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i != (len(self.layers) - 1):
                if self.act == "relu":
                    h = F.relu(h)
                elif self.act == "tanh":
                    h = F.tanh(h)

        return h

    def forward(self, features):
        embeddings = self.internal_forward(features)

        return embeddings

    def graph_process(self, embeddings):
        emb=F.normalize(embeddings,p=2,dim=1)
        emb=torch.nan_to_num(emb,nan=0.0,posinf=0.0,neginf=0.0)
        d=max(1.0,emb.size(1))
        sim=torch.matmul(emb,emb.t())/math.sqrt(d)
        sim=sim-torch.eye(sim.size(0),device=sim.device)*1e9
        attn=F.softmax(sim,dim=1)
  
        if (self.k is not None) and (0 < self.k < attn.size(1)):
            topk_vals, topk_idx = torch.topk(attn, self.k, dim=1) 
            mask = torch.zeros_like(attn)
            mask.scatter_(1, topk_idx, 1.0)
            attn = attn * mask

            row_sum = attn.sum(dim=1, keepdim=True)
            attn = attn / (row_sum + 1e-12)
        attn = (attn + attn.t()) / 2.0
        def _apply_nonlin_tensor(x, non_lin, param_i):
            if non_lin == 'relu':
                return F.relu(x)
            elif non_lin == 'tanh':
                return torch.tanh(x)
            elif non_lin == 'sigmoid':
                return torch.sigmoid(x)
            elif non_lin == 'softplus':
                return F.softplus(x)
            else:
                return x
        try:
            attn = apply_non_linearity(attn, self.non_linearity, self.i)
        except Exception:
            attn = _apply_nonlin_tensor(attn, self.non_linearity, self.i)
        learned_adj = F.dropout(attn, p=self.dropedge_rate, training=self.training)
        if self.sparse and (not self.training):
            try:
                rows, cols, values = knn_fast(embeddings, self.k, 1000)
                rows_ = torch.cat((rows, cols))
                cols_ = torch.cat((cols, rows))
                values_ = torch.cat((values, values))
                try:
                    values_ = apply_non_linearity(values_, self.non_linearity, self.i)
                except Exception:
                    values_ = _apply_nonlin_tensor(values_, self.non_linearity, self.i)
                values_ = F.dropout(values_, p=self.dropedge_rate, training=self.training)
                values_ = torch.nan_to_num(values_, nan=0.0, posinf=1.0, neginf=0.0)
                g = dgl.graph((rows_, cols_), num_nodes=embeddings.shape[0], device=embeddings.device)
                g.edata['w'] = values_
                return g
            except Exception as e:
                print("Warning: knn_fast->dgl graph failed in inference branch:", e)
                return learned_adj
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