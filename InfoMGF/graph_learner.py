import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GCNConv_dense
from layers import Attentive
from utils import *
from torch_geometric.nn import global_mean_pool
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
                    torch.nn.init.constant_(layer.linear.bias, 0.0)

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
    
    def graph_process(self, embeddings, batch=None):
        """
        Args:
            embeddings: [num_nodes, feature_dim]
            batch: [num_nodes] 指示每个节点属于哪个图 (可选)
            之前的一个大问题就出现在这把我们的视图拼成一个大图之后我就直接用这个
            论文里面的大图构造函数了，但是这个是有问题的 这个函数会不加限制的在整个大图上进行连接构建
            我们的视图应该是在不同视图内部构造
        """
        embeddings = torch.nan_to_num(embeddings, nan=0.0, posinf=1.0, neginf=0.0)
        emb = F.normalize(embeddings, p=2, dim=1)
        emb = torch.nan_to_num(emb, nan=0.0, posinf=1.0, neginf=0.0)

        N = emb.size(0)
        d = max(1.0, float(emb.size(1)))
        sim = torch.matmul(emb, emb.t()) / math.sqrt(d)
        sim = torch.clamp(sim, min=-50.0, max=50.0)

        mask = torch.eye(N, device=sim.device, dtype=torch.bool)
        sim = sim.masked_fill(mask, -1e9)
       
        if batch is not None:
           
            batch_i = batch.unsqueeze(1)  # [N, 1]
            batch_j = batch.unsqueeze(0)  # [1, N]
            cross_graph_mask = (batch_i != batch_j)  # [N, N]
            sim = sim.masked_fill(cross_graph_mask, -1e9)

        sim_max = torch.max(sim, dim=1, keepdim=True)[0]
        sim_exp = torch.exp(sim - sim_max)
        sim_exp = torch.nan_to_num(sim_exp, nan=0.0, posinf=0.0, neginf=0.0)
        sim_sum = torch.sum(sim_exp, dim=1, keepdim=True)
        attn = sim_exp / (sim_sum + 1e-12)

       
        if (self.k is not None) and (0 < self.k < attn.size(1)):
            # 获取top-k
            topk_vals, topk_idx = torch.topk(attn, self.k, dim=1)
            
            if self.training:
                threshold = topk_vals[:, -1].unsqueeze(1)  
                mask_k = torch.sigmoid((attn - threshold) * 50.0) 
                attn = attn * mask_k
            else:
                
                mask_k = torch.zeros_like(attn)
                mask_k = mask_k.scatter(1, topk_idx, 1.0)
                attn = attn * mask_k
            
            
            row_sum = attn.sum(dim=1, keepdim=True)
            attn = attn / (row_sum + 1e-12)

        attn = (attn + attn.t()) / 2.0
        
        if batch is not None:
            batch_i = batch.unsqueeze(1)  
            batch_j = batch.unsqueeze(0) 
            cross_graph_mask = (batch_i != batch_j)  
            attn = attn.masked_fill(cross_graph_mask, 0.0) 

        if self.non_linearity == 'relu':
            attn = F.relu(attn)
        elif self.non_linearity == 'tanh':
            attn = torch.tanh(attn)
        elif self.non_linearity == 'sigmoid':
            attn = torch.sigmoid(attn)

        attn = torch.nan_to_num(attn, nan=0.0, posinf=1.0, neginf=0.0)

        if self.training and (self.dropedge_rate > 0.0):
            dropout_mask = (torch.rand_like(attn) > self.dropedge_rate).float()
            learned_adj = attn * dropout_mask
        else:
            learned_adj = attn

        return learned_adj

class AttentionFusion(nn.Module):
    def __init__(self, input_dim, num_views=2, temperature=1.0):
        super(AttentionFusion, self).__init__()
        self.num_views = num_views
        self.input_dim = input_dim
        self.temperature = temperature  
        
        self.attention_mlp = nn.Sequential(
            nn.Linear(input_dim * num_views, input_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim * 2, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, num_views)
        )
        
       
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias) 
        
    def forward(self, view_features, batch=None):
        if batch is not None:
            return self._graph_level_attention(view_features, batch)
        else:
      
            return self._node_level_attention(view_features)
    
    def _graph_level_attention(self, view_features, batch):
        graph_embs = []
        for view_feat in view_features:
            graph_emb = global_mean_pool(view_feat, batch)  
            graph_embs.append(graph_emb)
        
        concatenated = torch.cat(graph_embs, dim=1)  
        
        scores = self.attention_mlp(concatenated) 
       
        attention_weights = F.softmax(scores / self.temperature, dim=1)  
        
        fused_feature = torch.zeros_like(view_features[0])
        batch_size = len(torch.unique(batch))
        
        for i in range(self.num_views):
            node_weights = torch.zeros_like(view_features[0][:, 0:1])  
            
            for graph_idx in range(batch_size):
                graph_mask = (batch == graph_idx)
                graph_weight = attention_weights[graph_idx, i]
                node_weights[graph_mask] = graph_weight
            
            fused_feature += node_weights * view_features[i]
        
        return fused_feature, attention_weights
    
    def _node_level_attention(self, view_features):
       
        concatenated = torch.cat(view_features, dim=1)  
        
        scores = self.attention_mlp(concatenated) 
      
        attention_weights = F.softmax(scores / self.temperature, dim=1)  
        
        fused_feature = torch.zeros_like(view_features[0])
        for i in range(self.num_views):
            weight = attention_weights[:, i].unsqueeze(1)  
            fused_feature += weight * view_features[i]  
        
        return fused_feature, attention_weights