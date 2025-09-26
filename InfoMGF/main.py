import argparse
import copy
from datetime import datetime

import numpy as np
from sklearn.base import accuracy_score
import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from data_loader import BrainGraphDataset
from model import *
from graph_learner import *
from utils import *
from params import *
from sklearn.cluster import KMeans
from kmeans_pytorch import kmeans as KMeans_py
from sklearn.metrics import f1_score
from torch_geometric.loader import DataLoader

import random

EOS = 1e-10
args = set_params()

class Experiment:
    def __init__(self):
        super(Experiment, self).__init__()
        self.training = False

    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        np.random.seed(seed)
        random.seed(seed)


    def test_cls_graphlevel(self, encoder, classifier, loader,specific_graph_learner,fused_graph_learner,args):
        encoder.eval(); classifier.eval()
        [m.eval() for m in specific_graph_learner]
        fused_graph_learner.eval()
        all_preds, all_labels = [], []
        total_loss = 0.0; nb = 0
        device= self.device if hasattr(self, 'device') else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                adjs_list, node2graph, N = build_adjs_from_batch(batch, device=self.device)
                feat=batch.x
                try:
                    view_features = AGG([feat for _ in range(len(adjs_list))], adjs_list, args.r, sparse=getattr(args, 'sparse', False))
                    if not isinstance(view_features, list):
                        view_features = [view_features for _ in range(len(adjs_list))]
                except Exception:
                    view_features = [feat for _ in range(len(adjs_list))]
                learned_specific_adjs = []
                for i in range(len(adjs_list)):
                    emb = specific_graph_learner[i].internal_forward(view_features[i])
                    learned_adj = specific_graph_learner[i].graph_process(emb)  # 可能是 DGL graph 或稀疏/dense tensor
                    if getattr(args, 'sparse', False):
                        if hasattr(learned_adj, 'edata') and 'w' in learned_adj.edata:
                            learned_adj.edata['w'] = learned_adj.edata['w'].detach()
                    else:
                        try:
                            learned_adj = learned_adj.detach()
                        except Exception:
                            pass    
                    learned_specific_adjs.append(learned_adj)
                fused_input = torch.cat(view_features + [feat], dim=1)
                fused_emb = fused_graph_learner.internal_forward(fused_input)
                learned_fused_adj = fused_graph_learner.graph_process(fused_emb)
                if not getattr(args, 'sparse', False):
                    try:
                        learned_fused_adj = learned_fused_adj.detach()
                    except Exception:
                        pass
                else:
                    if hasattr(learned_fused_adj, 'edata') and 'w' in learned_fused_adj.edata:
                        learned_fused_adj.edata['w'] = learned_fused_adj.edata['w'].detach()
                try:
                    z_nodes=encoder(feat, learned_fused_adj)
                except Exception:
                    spar=adj_to_sparse_coo(learned_fused_adj,N,self.device)
                    z_nodes=encoder(feat,spar)
                graph_emb = global_mean_pool(z_nodes, node2graph)
                logits = classifier(graph_emb)
                loss = F.cross_entropy(logits, batch.y.view(-1))
                total_loss += float(loss.item()); nb += 1
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds.tolist()); all_labels.extend(batch.y.cpu().numpy().reshape(-1).tolist())
        if nb == 0:
            return 0.0, 0.0, 0.0, 0.0
        avg_loss = total_loss / nb
        from sklearn.metrics import accuracy_score, f1_score
        acc = accuracy_score(all_labels, all_preds)
        f1m = f1_score(all_labels, all_preds, average='macro')
        f1mi = f1_score(all_labels, all_preds, average='micro')
        return avg_loss, acc, f1m, f1mi

  

    def train(self, args):
        print(args)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        torch.cuda.set_device(args.gpu)
        #原始的邻接矩阵adjs_original
        data_root='././proceed_data'
        train_dataset=BrainGraphDataset(root=data_root, split='train')
        val_dataset=BrainGraphDataset(root=data_root, split='val')
        test_dataset=BrainGraphDataset(root=data_root, split='test')

        train_loader=DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader=DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader=DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        if len(train_dataset)>0:
            nfeats=train_dataset.num_node_features
            nclasses=train_dataset.num_classes
        else:
            nfeats=0
            nclasses=0
        print(f"Number of features: {nfeats}")
        print(f"Number of classes: {nclasses}")
        num_views=2
     
        test_accuracies = []
        test_maf1 = []
        test_mif1 = []
        validation_accuracies = []

        #     fh = open("result_" + args.dataset + "_Class.txt", "a")
        #     print(args, file=fh)
        #     fh.write('\r\n')
        #     fh.flush()
        #     fh.close()

        for trial in range(args.ntrials):

            self.setup_seed(trial)
            # adjs = copy.deepcopy(adjs_original)
            # features = copy.deepcopy(features_original)
            # view_features = AGG([features for _ in range(len(adjs))], adjs, args.r, sparse=args.sparse)
            # view_features.append(features)

            specific_graph_learner = [ATT_learner(2, nfeats, args.k, 6, args.dropedge_rate, args.sparse, args.activation_learner) for _ in range(num_views)]
            fused_graph_learner = ATT_learner(2, nfeats*(3), args.k, 6, args.dropedge_rate, args.sparse, args.activation_learner)

            encoder=GraphEncoder(nlayers=args.nlayer_gnn, in_dim=nfeats, hidden_dim=args.hidden_dim, emb_dim=args.emb_dim, dropout=args.dropout, sparse=args.sparse)
            classifier=GraphClassifierHead(in_dim=args.emb_dim,nclasses=nclasses)
            encoder = encoder.to(device)
            classifier = classifier.to(device)
            specific_graph_learner = [m.to(device) for m in specific_graph_learner]
            fused_graph_learner = fused_graph_learner.to(device)
            params = []
            for m in specific_graph_learner:
                params.append({'params': m.parameters()})
            params.append({'params': fused_graph_learner.parameters()})
            params.append({'params': encoder.parameters()})
            params.append({'params': classifier.parameters()})
            optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.w_decay)
            # if torch.cuda.is_available():
            #     model = model.cuda()
            #     specific_graph_learner = [m.cuda() for m in specific_graph_learner]
            #     fused_graph_learner = fused_graph_learner.cuda()
            best_val = -1.0
            best_state = None
            for epoch in range(1, args.epochs + 1):
                encoder.train()
                classifier.train()
                [learner.train() for learner in specific_graph_learner]
                fused_graph_learner.train()
                total_loss = 0.0
                n_batches = 0
                for batch in train_loader:
                    batch = batch.to(device)
                    adj_list,node2graph,N=build_adjs_from_batch(batch, device)
                    feat=batch.x
                    view_features = AGG([feat for _ in range(len(adj_list))], adj_list, args.r, sparse=args.sparse)
                    fused_input = torch.cat(view_features + [feat], dim=1)
                    learned_specific_adjs = []
                    for i in range(len(adj_list)):
                        emb = specific_graph_learner[i].internal_forward(view_features[i])
                        learned_adj = specific_graph_learner[i].graph_process(emb)  # 可能是 DGL graph 或稀疏/dense tensor
                        learned_specific_adjs.append(learned_adj)
                    fused_emb = fused_graph_learner.internal_forward(fused_input)
                    learned_fused_adj = fused_graph_learner.graph_process(fused_emb)
                    z_nodes = encoder(feat, learned_fused_adj)
                    graph_emb = global_mean_pool(z_nodes, node2graph)
                    logits = classifier(graph_emb)
                    loss_sup = F.cross_entropy(logits, batch.y.view(-1))
                    z_specifics=[]
                    sparse_specific_adjs=[]
                    for adj_sp in learned_specific_adjs:
                        z_sp=encoder(feat,adj_sp)
                        z_specifics.append(z_sp)
                        sp=adj_to_sparse_coo(adj_sp,N,device)
                        sparse_specific_adjs.append(sp)
                    sparse_fused=adj_to_sparse_coo(learned_fused_adj,N,device)
                    loss_self,loss_details=encoder.cal_custom_loss(z_specifics,z_nodes,sparse_specific_adjs,sparse_fused
                                                                ,args.tau,args.h,args.alpha,args.beta,args.gamma)
                    loss=loss_sup+args.lambda1*loss_self
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += float(loss.item())
                    n_batches += 1
                
                avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
                print(f"Trial {trial} Epoch {epoch}  Loss {avg_loss:.4f}")
            


                if epoch % args.eval_freq == 0:
                    val_loss, val_acc, val_f1_macro, val_f1_micro = self.test_cls_graphlevel(self, encoder, classifier, val_loader,specific_graph_learner,fused_graph_learner,args)
                    print(f"Val Acc: {val_acc:.4f}  Val Loss: {val_loss:.4f}  Val F1_macro: {val_f1_macro:.4f}")
                    if val_acc > best_val:
                        best_val = val_acc
                        best_state = {
                            'encoder': encoder.state_dict(),
                            'classifier': classifier.state_dict(),
                            'specific': [m.state_dict() for m in specific_graph_learner],
                            'fused': fused_graph_learner.state_dict()
                        }
            if best_state is not None:
                encoder.load_state_dict(best_state['encoder'])
                classifier.load_state_dict(best_state['classifier'])
                for i, m in enumerate(specific_graph_learner):
                    m.load_state_dict(best_state['specific'][i])
                fused_graph_learner.load_state_dict(best_state['fused'])

            test_loss, test_acc, test_f1_macro, test_f1_micro = self.test_cls_graphlevel(self, encoder, classifier, test_loader,specific_graph_learner,fused_graph_learner,args)
            print(f"Trial {trial} TEST acc: {test_acc:.4f}  f1_macro: {test_f1_macro:.4f}")
            if args.downstream_task == 'classification':
                validation_accuracies.append(best_val)
                test_accuracies.append(test_acc)
                test_maf1.append(test_f1_macro)
                test_mif1.append(test_f1_micro)
        if args.downstream_task == 'classification' and len(validation_accuracies) > 0:
            self.print_results(validation_accuracies, test_accuracies, test_maf1, test_mif1)
    def print_results(self, validation_accu, test_accu, test_maf1, test_mif1):
        # 计算均值（mean）和标准差（std），体现结果稳定性
        s_val = "Val accuracy: {:.4f} +/- {:.4f}".format(np.mean(validation_accu), np.std(validation_accu))
        s_test = "Test accuracy: {:.4f} +/- {:.4f}".format(np.mean(test_accu), np.std(test_accu))
        maf1_test = "Test maf1: {:.4f} +/- {:.4f}".format(np.mean(test_maf1), np.std(test_maf1))
        mif1_test = "Test mif1: {:.4f} +/- {:.4f}".format(np.mean(test_mif1), np.std(test_mif1))
        
        # 控制台打印统计结果
        print(s_val)
        print(s_test)
        print(maf1_test)
        print(mif1_test)
        
        # 追加写入统计结果到分类任务文件
        fh = open("result_" + args.dataset + "_Class.txt", "a")
        fh.write("Test maf1: {:.4f} +/- {:.4f}".format(np.mean(test_maf1), np.std(test_maf1)))
        fh.write('\r\n')
        fh.write("Test mif1: {:.4f} +/- {:.4f}".format(np.mean(test_mif1), np.std(test_mif1)))
        fh.write('\r\n')
        fh.flush()
        fh.close()


if __name__ == '__main__':

        experiment = Experiment()
        experiment.train(args)
