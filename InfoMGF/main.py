import argparse
import copy
from datetime import datetime

import numpy as np
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score,confusion_matrix
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

    def callculate_detailed(self,all_labels,all_preds,all_probs,n_classes,trial,split='test'):
        accuracy = accuracy_score(all_labels, all_preds)
        precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        precision_micro = precision_score(all_labels, all_preds, average='micro', zero_division=0)
        recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        recall_micro = recall_score(all_labels, all_preds, average='micro', zero_division=0)
        f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)
        precision_per_class = precision_score(all_labels, all_preds, average=None, zero_division=0)
        recall_per_class = recall_score(all_labels, all_preds, average=None, zero_division=0)
        f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
        try:
            if n_classes == 2:
                # 二分类情况
                auc_roc_macro = roc_auc_score(all_labels, all_probs[:, 1])
                auc_roc_micro = auc_roc_macro
            else:
                # 多分类情况 - 使用one-vs-rest策略
                auc_roc_macro = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
                auc_roc_micro = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='micro')
        except Exception as e:
            print(f"AUC calculation warning: {e}")
            auc_roc_macro = 0.0
            auc_roc_micro = 0.0
        
        cm = confusion_matrix(all_labels, all_preds)
        cm=confusion_matrix(all_labels, all_preds)
        print(f"\n=== Detailed Metrics - {np.split} - Trial {trial} ===")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision - Macro: {precision_macro:.4f}, Micro: {precision_micro:.4f}")
        print(f"Recall    - Macro: {recall_macro:.4f}, Micro: {recall_micro:.4f}")
        print(f"F1-Score  - Macro: {f1_macro:.4f}, Micro: {f1_micro:.4f}")
        print(f"AUC-ROC   - Macro: {auc_roc_macro:.4f}, Micro: {auc_roc_micro:.4f}")

        print("\nPer-class metrics:")
        for i in range(n_classes):
            print(f"Class {i}: Precision={precision_per_class[i]:.4f}, Recall={recall_per_class[i]:.4f}, F1={f1_per_class[i]:.4f}")
        
        # 打印混淆矩阵
        print(f"\nConfusion Matrix (shape: {cm.shape}):")
        print(cm)
        return {
            'acc': accuracy,
            'precision_macro': precision_macro,
            'precision_micro': precision_micro,
            'recall_macro': recall_macro,
            'recall_micro': recall_micro,
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'auc_macro': auc_roc_macro,
            'auc_micro': auc_roc_micro,
            'confusion_matrix': cm,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class
        }

    def test_cls_graphlevel(self, encoder, classifier, loader,specific_graph_learner,attention_fusion,args,trial=0,split='test'):
        encoder.eval(); classifier.eval()
        specific_graph_learner.eval()
        attention_fusion.eval()
        all_preds, all_labels = [], []
        all_probs = []
        total_loss = 0.0; nb = 0
        n_classes=None
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
                z_specifics=[]
                for i in range(len(adjs_list)):
                    emb = specific_graph_learner.internal_forward(view_features[i])
                    learned_adj = specific_graph_learner.graph_process(emb)  # 可能是 DGL graph 或稀疏/dense tensor
                    if getattr(args, 'sparse', False):
                        if hasattr(learned_adj, 'edata') and 'w' in learned_adj.edata:
                            learned_adj.edata['w'] = learned_adj.edata['w'].detach()
                    else:
                        try:
                            learned_adj = learned_adj.detach()
                        except Exception:
                            pass    
                    learned_specific_adjs.append(learned_adj)
                    z_sp=encoder(feat,learned_adj)
                    z_specifics.append(z_sp)
                fused_z,attention_weight=attention_fusion(z_specifics)

             
                graph_emb = global_mean_pool(fused_z, node2graph)
                logits = classifier(graph_emb)
                probs = F.softmax(logits, dim=1)
                loss = F.cross_entropy(logits, batch.y.view(-1))
                total_loss += float(loss.item()); nb += 1
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds.tolist()); all_labels.extend(batch.y.cpu().numpy().reshape(-1).tolist())
                all_probs.extend(probs.cpu().numpy())
                if n_classes is None:
                    n_classes = logits.shape[1]
        if nb == 0:
            return 0.0, 0.0, 0.0, 0.0
        avg_loss = total_loss / nb
        all_probs=np.array(all_probs)
        detailed_metrics = self.callculate_detailed(all_labels, all_preds, all_probs, n_classes, trial, split)
        return avg_loss, detailed_metrics

  

    def train(self, args):
        print(args)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        torch.cuda.set_device(args.gpu)
        #原始的邻接矩阵adjs_original
        data_root='../../processed_data'
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
     
        test_results = []
        val_results = []

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

            specific_graph_learner = ATT_learner(2, nfeats, args.k, 6, args.dropedge_rate, args.sparse, args.activation_learner) 
            attention_fusion=AttentionFusion(input_dim=args.emb_dim, num_views=num_views)
            encoder=GraphEncoder(nlayers=args.nlayer_gnn, in_dim=nfeats, hidden_dim=args.hidden_dim, emb_dim=args.emb_dim, dropout=args.dropout, sparse=args.sparse)
            classifier=GraphClassifierHead(in_dim=args.emb_dim,nclasses=nclasses)
            encoder = encoder.to(device)
            classifier = classifier.to(device)
            specific_graph_learner = specific_graph_learner.to(device)
            attention_fusion = attention_fusion.to(device)
            params = []
            params.append({'params': specific_graph_learner.parameters()})
            params.append({'params': encoder.parameters()})
            params.append({'params': classifier.parameters()})
            params.append({'params': attention_fusion.parameters()})
            optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.w_decay)
            # if torch.cuda.is_available():
            #     model = model.cuda()
            #     specific_graph_learner = [m.cuda() for m in specific_graph_learner]
            #     fused_graph_learner = fused_graph_learner.cuda()
            best_val = -1.0
            best_state = None
            best_val_metrics=None
            for epoch in range(1, args.epochs + 1):
                encoder.train()
                classifier.train()
                specific_graph_learner.train()
                attention_fusion.train()
                total_loss = 0.0
                n_batches = 0
                for batch in train_loader:
                    batch = batch.to(device)
                    adj_list,node2graph,N=build_adjs_from_batch(batch, device)
                    feat=batch.x
                    view_features = AGG([feat for _ in range(len(adj_list))], adj_list, args.r, sparse=args.sparse)
                    learned_specific_adjs = []
                    z_specifics=[]
                    for i in range(len(adj_list)):
                        emb = specific_graph_learner.internal_forward(view_features[i])
                        learned_adj = specific_graph_learner.graph_process(emb)  # 可能是 DGL graph 或稀疏/dense tensor
                        learned_specific_adjs.append(learned_adj)
                    sparse_specific_adjs=[]
                    for adj_sp in learned_specific_adjs:
                        z_sp=encoder(feat,adj_sp)
                        z_specifics.append(z_sp)
                        sp=adj_to_sparse_coo(adj_sp,N,device)
                        sparse_specific_adjs.append(sp)  
                    fused_z,attention_weights = attention_fusion(z_specifics)
                    graph_emb = global_mean_pool(fused_z, node2graph)
                    logits = classifier(graph_emb)
                    loss_sup = F.cross_entropy(logits, batch.y.view(-1))
                    loss_self,loss_details=encoder.cal_custom_loss(z_specifics,fused_z,sparse_specific_adjs
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
                    val_loss,val_metrics = self.test_cls_graphlevel(encoder, classifier, val_loader,specific_graph_learner,attention_fusion,args,trial,'val')
                    print(f"Val Acc: {val_metrics['acc']:.4f}  Val Loss: {val_loss:.4f}  Val F1_macro: {val_metrics['f1_macro']:.4f}")
                    if val_metrics['acc'] > best_val:
                        best_val = val_metrics['acc']
                        best_val_metrics=val_metrics
                        best_state = {
                            'encoder': encoder.state_dict(),
                            'classifier': classifier.state_dict(),
                            'specific': specific_graph_learner.state_dict(),
                            'attention_fusion': attention_fusion.state_dict()
                        }
                    if best_val_metrics is not None:
                        val_results.append({
                            'trial': trial,
                            'acc': best_val,
                            'metrics': best_val_metrics
                        })
                    if best_state is not None:
                        encoder.load_state_dict(best_state['encoder'])
                        classifier.load_state_dict(best_state['classifier'])
                        specific_graph_learner.load_state_dict(best_state['specific'])
                        attention_fusion.load_state_dict(best_state['attention_fusion'])

            test_loss, test_metrics = self.test_cls_graphlevel(encoder, classifier, test_loader,specific_graph_learner,attention_fusion,args,trial,'test')
            print(f"Trial {trial} TEST acc: {test_metrics['acc']:.4f}  f1_macro: {test_metrics['f1_macro']:.4f}")
            test_results.append({
                'trial': trial,
                'acc': test_metrics['acc'],
                'metrics': test_metrics
            })
        if args.downstream_task == 'classification' and len(test_results) > 0:
            self.print_results(val_results, test_results)
    def print_results(self, val_results, test_results):

        val_accs = [r['acc'] for r in val_results]
        test_accs = [r['acc'] for r in test_results]
        test_f1_macro = [r['metrics']['f1_macro'] for r in test_results]
        test_f1_micro = [r['metrics']['f1_micro'] for r in test_results]
        test_precision_macro = [r['metrics']['precision_macro'] for r in test_results]
        test_recall_macro = [r['metrics']['recall_macro'] for r in test_results]
        test_auc_macro = [r['metrics']['auc_macro'] for r in test_results]
        stats = {
            'val_acc': (np.mean(val_accs), np.std(val_accs)),
            'test_acc': (np.mean(test_accs), np.std(test_accs)),
            'test_f1_macro': (np.mean(test_f1_macro), np.std(test_f1_macro)),
            'test_f1_micro': (np.mean(test_f1_micro), np.std(test_f1_micro)),
            'test_precision_macro': (np.mean(test_precision_macro), np.std(test_precision_macro)),
            'test_recall_macro': (np.mean(test_recall_macro), np.std(test_recall_macro)),
            'test_auc_macro': (np.mean(test_auc_macro), np.std(test_auc_macro))
        }
        print("FINAL RESULTS ACROSS ALL TRIALS")
        print("="*60)
        for metric, (mean, std) in stats.items():
            print(f"{metric.replace('_', ' ').title():<20}: {mean:.4f} ± {std:.4f}")
        
        # 保存到文件
        result_file = f"results/result_{args.dataset}_Class.txt"
        with open(result_file, "w") as fh:
            fh.write("FINAL RESULTS ACROSS ALL TRIALS\n")
            fh.write("="*60 + "\n")
            for metric, (mean, std) in stats.items():
                line = f"{metric.replace('_', ' ').title():<20}: {mean:.4f} ± {std:.4f}\n"
                fh.write(line)
            
            # 添加每个trial的详细结果
            fh.write("\n\nDETAILED TRIAL RESULTS:\n")
            fh.write("="*60 + "\n")
            for trial_result in test_results:
                trial = trial_result['trial']
                metrics = trial_result['metrics']
                fh.write(f"\nTrial {trial}:\n")
                fh.write(f"  Accuracy: {metrics['acc']:.4f}\n")
                fh.write(f"  F1-Macro: {metrics['f1_macro']:.4f}\n")
                fh.write(f"  Precision-Macro: {metrics['precision_macro']:.4f}\n")
                fh.write(f"  Recall-Macro: {metrics['recall_macro']:.4f}\n")
                fh.write(f"  AUC-Macro: {metrics['auc_macro']:.4f}\n")
  


if __name__ == '__main__':

        experiment = Experiment()
        experiment.train(args)