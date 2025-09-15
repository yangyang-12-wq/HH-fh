import torch
import torch.nn.functional as F
import numpy as np
from utils import *
EPS = 1e-10

def graph_augment(adjs_original, dropedge_rate, training, sparse):

    adjs_aug = []
    adjs = copy.deepcopy(adjs_original)

    if not sparse:
        adjs = [adj.to_sparse().coalesce() for adj in adjs]
    for i in range(len(adjs)):
        adj_aug = adjs[i]
        diag_indices, diag_value = get_sparse_diag(adj_aug)
        adj_aug = remove_self_loop([adj_aug])[0]
        value = F.dropout(adj_aug.values(), p=dropedge_rate, training=training)

        adj_aug = torch.sparse.FloatTensor(torch.cat((adj_aug.indices(), diag_indices), dim=1), torch.cat((value, diag_value), dim=0), adj_aug.shape).coalesce().to(adjs[i].device)
        adjs_aug.append(adj_aug)
    if sparse:
        adjs_aug = [torch_sparse_to_dgl_graph(a) for a in adjs_aug]
    else:
        adjs_aug = [adj.to_dense() for adj in adjs_aug]

    return adjs_aug

def graph_generative_augment(adjs, features, discriminator, sparse):

    adjs_aug = []
    if not sparse:
        adjs = [adj.to_sparse().coalesce() for adj in adjs]
    adjs = remove_self_loop(adjs)
    for i in range(len(adjs)):
        edge_index = adjs[i].indices()
        adj_aug_value = discriminator(features, edge_index)
        adj_aug = torch.sparse.FloatTensor(edge_index, adj_aug_value, adjs[i].shape).to(adjs[i].device)

        adj_aug = sparse_tensor_add_self_loop(adj_aug)
        adj_aug = normalize(adj_aug, 'sym', sparse=True)

        adjs_aug.append(adj_aug)

    if sparse:
        adjs_aug = [torch_sparse_to_dgl_graph(a) for a in adjs_aug]
    else:
        adjs_aug = [adj.to_dense() for adj in adjs_aug]
    return adjs_aug



class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, rep_dim, aug_lambda, temperature=1.0, bias=0.0 + 0.0001):
        super(Discriminator, self).__init__()

        self.embedding_layers = nn.ModuleList()
        self.embedding_layers.append(nn.Linear(input_dim, hidden_dim))
        self.edge_mlp = nn.Sequential(nn.Linear(hidden_dim * 2, 1))

        self.temperature = temperature
        self.bias = bias
        self.aug_lambda = aug_lambda

        self.decoder = nn.Sequential(nn.Linear(rep_dim, input_dim))

    def get_node_embedding(self, h):
        for layer in self.embedding_layers:
            h = layer(h)
            h = F.relu(h)
        return h

    def get_edge_weight(self, embeddings, edges):
        s1 = self.edge_mlp(torch.cat((embeddings[edges[0]], embeddings[edges[1]]), dim=1)).flatten()
        s2 = self.edge_mlp(torch.cat((embeddings[edges[1]], embeddings[edges[0]]), dim=1)).flatten()
        return (s1 + s2) / 2

    def gumbel_sampling(self, edges_weights_raw):
        eps = (self.bias - (1 - self.bias)) * torch.rand(edges_weights_raw.size()) + (1 - self.bias)
        gate_inputs = torch.log(eps) - torch.log(1 - eps)
        gate_inputs = gate_inputs.to(edges_weights_raw.device)
        gate_inputs = (gate_inputs + edges_weights_raw) / self.temperature
        output = torch.sigmoid(gate_inputs).squeeze()

        return output

    def forward(self, embedding, edges):
        embedding_ = self.get_node_embedding(embedding)
        edges_weights_raw = self.get_edge_weight(embedding_, edges)
        weights = self.gumbel_sampling(edges_weights_raw)
        return weights
    

    def cal_loss_dis(self, adjs_aug, adjs_original, view_features,
                     node_reps_aug=None, node_reps_orig=None,
                     kernel='rbf', normalize_hsic=True):
        """
        Torch-HSIC 版本的 Discriminator loss（GPU友好、可微）：
        - node_reps_orig, node_reps_aug: lists of node embeddings tensors (N, d) for each sample in batch.
          如果 node_reps_orig 为 None，会使用 view_features -> get_node_embedding(view_features[i]) 作为替代。
        - adjs_aug, adjs_original: lists of adjacency (dense torch tensors)
        - 返回: loss_dis
        """
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if isinstance(view_features, (list, tuple)):
            view_features = [
                vf.to(device) if isinstance(vf, torch.Tensor) else torch.tensor(vf, device=device, dtype=torch.float32)
                for
                vf in view_features]
        else:
            raise TypeError("view_features must be a list or tuple of tensors or arrays")
        batch_size = len(view_features)     

        # reconstruction loss (torch)
        loss_rec = torch.tensor(0.0, device=device)
        if node_reps_aug is not None:
            for i in range(batch_size):
                feat_agg_rec = self.decoder(node_reps_aug[i])  # -> reconstructed node features (N, feat_dim)
                vf = view_features[i]
                if not isinstance(vf, torch.Tensor):
                    vf = torch.tensor(vf, device=device, dtype=torch.float32)
                else:
                    vf = vf.to(device)
                loss_rec = loss_rec + sce_loss(feat_agg_rec, vf)
            loss_rec = loss_rec / batch_size

        #  HSIC loss between original and augmented node embeddings (torch)
        if node_reps_orig is None:
            node_reps_orig = []
            for i in range(batch_size):
                vf = view_features[i]
                if not isinstance(vf, torch.Tensor):
                    vf = torch.tensor(vf, device=device, dtype=torch.float32)
                else:
                    vf = vf.to(device)
                emb = self.get_node_embedding(vf)  # (N, hidden_dim)
                node_reps_orig.append(emb)

        if node_reps_aug is None:
            loss_kernel = torch.tensor(0.0, device=device)
        else:
            loss_kernel = torch.tensor(0.0, device=device)
            for i in range(batch_size):
                x = node_reps_orig[i].to(device)
                y = node_reps_aug[i].to(device)
                x = F.normalize(x, p=2, dim=1)
                y = F.normalize(y, p=2, dim=1)
                hs = hsic_torch(x, y, kernel=kernel, sigma=None, normalize=normalize_hsic)
                loss_kernel = loss_kernel + (-hs)  # negative because we want to maximize HSIC
            loss_kernel = loss_kernel / batch_size

        loss_dis = self.aug_lambda * loss_kernel + loss_rec

        return loss_dis