import argparse
import copy
from datetime import datetime
import matplotlib.pyplot as plt
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
from torch.utils.tensorboard import SummaryWriter

EOS = 1e-10
args = set_params()
def quick_diagnose(logits, batch, model_modules, optimizer, criterion, attention_weights=None, graph_emb=None, 
                   attention_balance_loss=None, view_diversity_loss=None):
    # Support both multi-class and binary (single-logit) cases
    if logits.dim() == 2 and logits.size(1) > 1:
        preds = logits.argmax(dim=1)
        print("\n" + "="*50)
        print("Pred distribution:", torch.bincount(preds).cpu().numpy())
        print("Label distribution:", torch.bincount(batch.y.view(-1)).cpu().numpy())
        probs = F.softmax(logits, dim=1)
        class_conf = probs.mean(dim=0).detach().cpu().numpy()
        print(f"Avg pred confidence per class: {[f'{c:.3f}' for c in class_conf]}")
        print("Logits stats mean/std/min/max:", float(logits.mean().item()), float(logits.std().item()),
              float(logits.min().item()), float(logits.max().item()))
        print("Logits per class:")
        for i in range(logits.shape[1]):
            class_logits = logits[:, i]
            print(f"  Class {i}: mean={class_logits.mean().item():.4f}, std={class_logits.std().item():.4f}")
    else:
        probs_pos = torch.sigmoid(logits.view(-1))
        preds = (probs_pos > 0.5).long()
        print("\n" + "="*50)
        print("Pred distribution:", torch.bincount(preds).cpu().numpy())
        print("Label distribution:", torch.bincount(batch.y.view(-1)).cpu().numpy())
        print(f"Avg positive probability: {probs_pos.mean().item():.3f}")
        print("Logits stats mean/std/min/max:", float(logits.mean().item()), float(logits.std().item()),
              float(logits.min().item()), float(logits.max().item()))
    
    per_sample_std = logits.std(dim=1)
    print("Per-sample logits std mean:", float(per_sample_std.mean().item()),
          "frac zero-std:", float((per_sample_std==0).float().mean().item()))
    # Compute criterion loss preview with correct shapes/types
    try:
        if logits.dim() == 2 and logits.size(1) > 1:
            loss_val = criterion(logits, batch.y.view(-1))
        else:
            loss_val = F.binary_cross_entropy_with_logits(logits.view(-1), batch.y.float())
    except Exception as e:
        loss_val = torch.tensor(float('nan'))
    print("Loss value:", float(loss_val.item()))
    
    # 分析图嵌入的区分度
    if graph_emb is not None:
        print("\nGraph embeddings stats:")
        print(f"  Mean: {graph_emb.mean().item():.4f}, Std: {graph_emb.std().item():.4f}")
        
        graph_emb_norm = F.normalize(graph_emb, p=2, dim=1)
        sim_matrix = torch.mm(graph_emb_norm, graph_emb_norm.t())
       
        mask = ~torch.eye(sim_matrix.size(0), dtype=torch.bool, device=sim_matrix.device)
        off_diag_sim = sim_matrix[mask]
        print(f"  Inter-graph similarity: mean={off_diag_sim.mean().item():.4f}, std={off_diag_sim.std().item():.4f}")
        if off_diag_sim.mean().item() > 0.9:
            print("  ⚠️ WARNING: Graph embeddings are too similar! Model may not distinguish different graphs.")
    
    if attention_weights is not None:
        print("\nAttention weights stats:")
        print("  Mean per view:", attention_weights.mean(dim=0).detach().cpu().numpy())
        print("  Std per view:", attention_weights.std(dim=0).detach().cpu().numpy())
  
        attn_entropy = -(attention_weights * torch.log(attention_weights + 1e-8)).sum(dim=1).mean()
        max_entropy = torch.log(torch.tensor(float(attention_weights.size(1))))
        print(f"  Attention entropy: {attn_entropy.item():.4f} / {max_entropy.item():.4f} (higher=more balanced)")
        if attention_weights.mean(dim=0).max().item() > 0.75:
            print("  ⚠️ WARNING: Attention is heavily biased towards one view!")
  
    if attention_balance_loss is not None:
        print(f"\nRegularization losses:")
        print(f"  Attention balance loss: {attention_balance_loss.item():.4f} (lower=more balanced)")
    if view_diversity_loss is not None:
        print(f"  View diversity loss: {view_diversity_loss.item():.4f} (lower=more diverse)")
    
    opt_param_ids = {id(p) for g in optimizer.param_groups for p in g['params']}
    for name, m in model_modules.items():
        for n,p in m.named_parameters():
            if id(p) not in opt_param_ids:
                print("WARNING: param not in optimizer:", name, n)

    if 'fusion' in model_modules:
        print("\nFusion layer gradients:")
        for n,p in model_modules['fusion'].named_parameters():
            grad_norm = 0.0 if p.grad is None else float(p.grad.norm().item())
            print(f"  {n}: grad_norm={grad_norm:.6f}")
    print("="*50 + "\n")
    return preds
class Experiment:
    def __init__(self):
        super(Experiment, self).__init__()
        self.training = False
        self.writer = None

    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        np.random.seed(seed)
        random.seed(seed)

    def callculate_detailed(self,all_labels,all_preds,all_probs,n_classes,trial,split='test',log_file=None,epoch=None):
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
                auc_roc_macro = roc_auc_score(all_labels, all_probs[:, 1])
                auc_roc_micro = auc_roc_macro
            else:
                
                auc_roc_macro = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
                auc_roc_micro = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='micro')
        except Exception as e:
            if log_file:
                log_file.write(f"AUC calculation warning: {e}\n")
            else:
                print(f"AUC calculation warning: {e}")
            auc_roc_macro = 0.0
            auc_roc_micro = 0.0
        
        cm = confusion_matrix(all_labels, all_preds)
        
        # 准备输出内容
        epoch_info = f" - Epoch {epoch}" if epoch is not None else ""
        output = f"\n{'='*80}\n"
        output += f"=== {split.upper()} Metrics - Trial {trial}{epoch_info} ===\n"
        output += f"{'='*80}\n"
        output += f"Accuracy: {accuracy:.4f}\n"
        output += f"Precision - Macro: {precision_macro:.4f}, Micro: {precision_micro:.4f}\n"
        output += f"Recall    - Macro: {recall_macro:.4f}, Micro: {recall_micro:.4f}\n"
        output += f"F1-Score  - Macro: {f1_macro:.4f}, Micro: {f1_micro:.4f}\n"
        output += f"AUC-ROC   - Macro: {auc_roc_macro:.4f}, Micro: {auc_roc_micro:.4f}\n"
        output += f"\nPer-class metrics:\n"
        for i in range(n_classes):
            output += f"  Class {i}: Precision={precision_per_class[i]:.4f}, Recall={recall_per_class[i]:.4f}, F1={f1_per_class[i]:.4f}\n"
        output += f"\nConfusion Matrix (shape: {cm.shape}):\n"
        output += f"{cm}\n"
        output += f"{'='*80}\n"
        
        if log_file:
            log_file.write(output)
            log_file.flush()  # 立即写入磁盘
        
        # 控制台只显示简要信息
        print(f"[{split.upper()}] Trial {trial}{epoch_info}: Acc={accuracy:.4f}, F1_macro={f1_macro:.4f}, AUC_macro={auc_roc_macro:.4f}")
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

    def test_cls_graphlevel(self, encoder, classifier, loader, specific_graph_learner, attention_fusion, args, trial=0, split='test', log_file=None, epoch=None):
        encoder.eval()
        classifier.eval()
        specific_graph_learner.eval()
        attention_fusion.eval()
        
        all_preds, all_labels = [], []
        all_probs = []
        total_loss = 0.0
        nb = 0
        n_classes = None
        
        device = self.device if hasattr(self, 'device') else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                adjs_list, node2graph, N = build_adjs_from_batch(batch, device=self.device)
                feat = batch.x
                
                try:
                    view_features = AGG([feat for _ in range(len(adjs_list))], adjs_list, args.r, sparse=getattr(args, 'sparse', False))
                    if not isinstance(view_features, list):
                        view_features = [view_features for _ in range(len(adjs_list))]
                except Exception:
                    view_features = [feat for _ in range(len(adjs_list))]
                
                learned_specific_adjs = []
                z_specifics = []
                
               
                for i in range(len(adjs_list)):
                    learned_adj = specific_graph_learner.graph_process(view_features[i], batch=node2graph)
                    learned_specific_adjs.append(learned_adj)
                    
                    emb = specific_graph_learner(view_features[i], learned_adj)
                    emb = F.normalize(emb, p=2, dim=1)
                    z_specifics.append(emb)
                
                fused_z, attention_weight = attention_fusion(z_specifics, batch=node2graph)
             
                graph_emb = global_mean_pool(fused_z, node2graph)
                logits = classifier(graph_emb)
                if n_classes is None:
                    if logits.dim() == 2 and logits.size(1) > 1:
                        n_classes = logits.size(1)
                    else:
                        n_classes = 2
                if n_classes == 2 and (logits.dim() == 1 or logits.size(1) == 1):
                    # Binary: single-logit
                    loss = F.binary_cross_entropy_with_logits(logits.view(-1), batch.y.float())
                    prob_pos = torch.sigmoid(logits.view(-1))
                    preds = (prob_pos > 0.5).long().cpu().numpy()
                    probs_np = prob_pos.detach().cpu().numpy()
                    probs_2col = np.stack([1.0 - probs_np, probs_np], axis=1)
                    all_probs.extend(probs_2col)
                else:
                    probs = F.softmax(logits, dim=1)
                    loss = F.cross_entropy(logits, batch.y.view(-1))
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    all_probs.extend(probs.cpu().numpy())
                
                total_loss += float(loss.item())
                nb += 1
                all_preds.extend(preds.tolist())
                all_labels.extend(batch.y.cpu().numpy().reshape(-1).tolist())
                
                
                all_preds.extend(preds.tolist())
                all_labels.extend(batch.y.cpu().numpy().reshape(-1).tolist())
                
        
        if nb == 0:
            return 0.0, 0.0, 0.0, 0.0
        
        avg_loss = total_loss / nb
        all_probs = np.array(all_probs)
        detailed_metrics = self.callculate_detailed(all_labels, all_preds, all_probs, n_classes, trial, split, log_file=log_file, epoch=epoch)
        
        return avg_loss, detailed_metrics


    def train(self, args):
        print(args)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        torch.cuda.set_device(args.gpu)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"runs/{args.dataset}_{timestamp}"
        self.writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard logs will be saved to: {log_dir}")
        
        # 创建结果日志文件
        results_log_path = f"results_{args.dataset}_{timestamp}.txt"
        results_log = open(results_log_path, 'w', encoding='utf-8')
        results_log.write(f"Training Results Log\n")
        results_log.write(f"Dataset: {args.dataset}\n")
        results_log.write(f"Timestamp: {timestamp}\n")
        results_log.write(f"{'='*80}\n\n")
        results_log.flush()
        print(f"Results will be saved to: {results_log_path}")
    
        data_root='../../../processed_fnirs/processed_data1'
        train_dataset=BrainGraphDataset(root=data_root, split='train',label_mode=args.label_mode)
        val_dataset=BrainGraphDataset(root=data_root, split='val',label_mode=args.label_mode)
        test_dataset=BrainGraphDataset(root=data_root, split='test',label_mode=args.label_mode)
    
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
        if nclasses == 2:
            criterion = torch.nn.CrossEntropyLoss()
        else:
            criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
        num_views=2
     
        test_results = []
        val_results = []

  
        trial = 0
        self.setup_seed(trial)
        trial_log_dir = f"{log_dir}/trial_{trial}"
        trial_writer = SummaryWriter(log_dir=trial_log_dir)
        specific_graph_learner = GraphLearnerGCN(gcn_input_dim=nfeats,
            gcn_hidden_dim=args.hidden_dim,
            gcn_output_dim=args.emb_dim,
            k=args.k,
            dropedge_rate=args.dropedge_rate,
            sparse=args.sparse,
            act=args.activation_learner) 
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
        if args.lr_schedule == 'cosine':
            total_epochs = max(1, args.epochs)
            warmup_epochs = max(0, min(args.warmup_epochs, total_epochs))
            if warmup_epochs > 0:
                warmup = LambdaLR(optimizer, lr_lambda=lambda e: (e + 1) / float(warmup_epochs))
            else:
                warmup = LambdaLR(optimizer, lr_lambda=lambda e: 1.0)
            cosine = CosineAnnealingLR(optimizer, T_max=max(1, total_epochs - warmup_epochs), eta_min=args.min_lr)
            scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
            lr_mode = 'cosine'
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.7, patience=args.scheduler_patience, verbose=True, min_lr=1e-5
            )
            lr_mode = 'plateau'
        best_val = -1.0
        best_state = None
        best_val_metrics=None

        epoch_losses = {
            'total': [],
            'supervised': [],
            'self_supervised': [],
            'lfd': [],
            's_high': [],
            'sc': [],
            'train_acc': []  
        }
        model_modules = {'learner': specific_graph_learner, 'encoder': encoder, 'fusion': attention_fusion, 'classifier': classifier}
        torch.autograd.set_detect_anomaly(True) 
        for epoch in range(1, args.epochs + 1):
            encoder.train()
            classifier.train()
            specific_graph_learner.train()
            attention_fusion.train()
            total_loss = 0.0
            train_correct = 0  
            train_total = 0    
            total_sup_loss = 0.0
            total_self_loss = 0.0
            total_lfd_loss = 0.0
            total_s_high_loss = 0.0
            total_sc_loss = 0.0
            n_batches = 0
            for batch_idx, batch in enumerate(train_loader):
                batch = batch.to(device)
                adj_list,node2graph,N=build_adjs_from_batch(batch, device)
                feat=batch.x
                view_features = AGG([feat for _ in range(len(adj_list))], adj_list, args.r, sparse=args.sparse)
    
                learned_specific_adjs = []
                z_specifics=[]
                for i in range(len(adj_list)):
                    
                    learned_adj = specific_graph_learner.graph_process(view_features[i], batch=node2graph)
                    learned_specific_adjs.append(learned_adj)
    
                    emb = specific_graph_learner(view_features[i], learned_adj)
                    
                    emb = F.normalize(emb, p=2, dim=1)
                    z_specifics.append(emb)
                
                
                if batch_idx % 50 == 0:

                    print(f"\n[Feature Stats at batch {batch_idx}]")
                    batch_size = len(torch.unique(node2graph))
                    print(f"Batch info: {batch_size} graphs, {N} total nodes")
                    for g in range(batch_size):
                        n_nodes_in_g = (node2graph == g).sum().item()
                        print(f"  Graph {g}: {n_nodes_in_g} nodes")
                    print(f"Original feat: mean={feat.mean().item():.4f}, std={feat.std().item():.4f}")
                    for i, vf in enumerate(view_features):
                        print(f"View {i} features: mean={vf.mean().item():.4f}, std={vf.std().item():.4f}")
                    for i, zs in enumerate(z_specifics):
                        print(f"Z_specific {i}: mean={zs.mean().item():.4f}, std={zs.std().item():.4f}")
                    
                    
                    la = learned_specific_adjs[0]
                    print(f"Learned_adj: nnz={la.sum().item():.0f}, density={la.sum().item()/(N*N)*100:.2f}%")
                    
                    
                    batch_size = len(torch.unique(node2graph))
                    if batch_size > 1:
                        within_sum = 0.0
                        cross_sum = 0.0
                        for g in range(batch_size):
                            mask_g = (node2graph == g)
                            within_block = la[mask_g][:, mask_g]
                            within_sum += within_block.sum().item()
                            for g2 in range(batch_size):
                                if g != g2:
                                    mask_g2 = (node2graph == g2)
                                    cross_block = la[mask_g][:, mask_g2]
                                    cross_sum += cross_block.sum().item()
                        print(f"  Within-graph connections: {within_sum:.2f}")
                        print(f"  Cross-graph connections: {cross_sum:.6f} (should be ~0)")
                        if cross_sum > 1e-6:
                            print(f"   WARNING: Cross-graph connections detected!")

                
                sparse_specific_adjs = []
                for adj_sp in learned_specific_adjs:
                    
                    sp = adj_to_sparse_coo(adj_sp, N, device) 
                    sparse_specific_adjs.append(sp)  
                fused_z,attention_weights = attention_fusion(z_specifics, batch=node2graph)
                
                graph_emb = global_mean_pool(fused_z, node2graph)
                logits = classifier(graph_emb)
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    print(f"Epoch {epoch} Batch {batch_idx}: Invalid logits detected")
                    continue
                
                if nclasses == 2 and (logits.size(1) == 1 or logits.dim()==1):
                    loss_sup = F.binary_cross_entropy_with_logits(logits.view(-1), batch.y.float())
                else:
                    loss_sup = criterion(logits, batch.y.view(-1))
                
                
                # attention_entropy = -(attention_weights * torch.log(attention_weights + 1e-8)).sum(dim=1).mean()
                # max_entropy = torch.log(torch.tensor(float(attention_weights.size(1)), device=attention_weights.device))
                # attention_balance_loss = (max_entropy - attention_entropy) 
                
                
                # view_graph_embs = []
                # for z in z_specifics:
                #     view_graph_emb = global_mean_pool(z, node2graph)  # [batch_size, dim]
                #     view_graph_emb = F.normalize(view_graph_emb, p=2, dim=1)
                #     view_graph_embs.append(view_graph_emb)
                
            
                # view_similarity = (view_graph_embs[0] * view_graph_embs[1]).sum(dim=1).abs().mean()
                # view_diversity_loss = view_similarity  
                
                loss_self,loss_details=encoder.cal_custom_loss(z_specifics,fused_z,learned_specific_adjs
                                                                ,args.tau,args.h,args.alpha,args.beta,args.gamma)
                
                if args.loss_mode == 'ce_only':
                    loss=loss_sup
                else:
                    loss=loss_sup + args.lambda1 * loss_self
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=5.0)
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=5.0)
                torch.nn.utils.clip_grad_norm_(specific_graph_learner.parameters(), max_norm=5.0)
                torch.nn.utils.clip_grad_norm_(attention_fusion.parameters(), max_norm=5.0)
                if batch_idx % 50 == 0:
                    preds = quick_diagnose(logits, batch, model_modules, optimizer, criterion, attention_weights, graph_emb
                                            )

                optimizer.step()
                
                
                total_loss += float(loss.item())
                total_sup_loss += float(loss_sup.item())
                total_self_loss += float(loss_self.item())
                total_lfd_loss += float(loss_details['lfd_loss'].item())
                total_s_high_loss += float(loss_details['s_high_loss'].item())
                total_sc_loss += float(loss_details['sc_loss'].item())
                n_batches += 1
                
                with torch.no_grad():
                    preds = logits.argmax(dim=1)
                    train_correct += (preds == batch.y.view(-1)).sum().item()
                    train_total += batch.y.size(0)
                if batch_idx % 50 == 0:
                    grad_norms = {}
                    current_batch_acc = train_correct / train_total if train_total > 0 else 0.0
                    print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}, Train Acc so far: {current_batch_acc:.4f}')
                    
                    for name, param in encoder.named_parameters():
                        if param.grad is not None:
                            grad_norms[f"encoder_{name}"] = param.grad.norm().item()
                    
                    for name, param in classifier.named_parameters():
                        if param.grad is not None:
                            grad_norms[f"classifier_{name}"] = param.grad.norm().item()
                    for name, param in specific_graph_learner.named_parameters():
                        if param.grad is not None:
                            grad_norms[f"specific_graph_learner_{name}"] = param.grad.norm().item()
                    for name, param in attention_fusion.named_parameters():
                        if param.grad is not None:
                            grad_norms[f"attention_fusion_{name}"] = param.grad.norm().item()
                    # 打印主要梯度
                    print("  Main gradients:")
                    for key, value in grad_norms.items():
                        print(f"  {key}: {value:.6f}")


            train_acc = train_correct / train_total if train_total > 0 else 0.0
            
            avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
            avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
            avg_sup_loss = total_sup_loss / n_batches if n_batches > 0 else 0.0
            avg_self_loss = total_self_loss / n_batches if n_batches > 0 else 0.0
            avg_lfd_loss = total_lfd_loss / n_batches if n_batches > 0 else 0.0
            avg_s_high_loss = total_s_high_loss / n_batches if n_batches > 0 else 0.0
            avg_sc_loss = total_sc_loss / n_batches if n_batches > 0 else 0.0
            
            trial_writer.add_scalar('Loss/Total', avg_loss, epoch)
            trial_writer.add_scalar('Loss/Supervised', avg_sup_loss, epoch)
            trial_writer.add_scalar('Loss/Self_Supervised', avg_self_loss, epoch)
            trial_writer.add_scalar('Loss/LFD', avg_lfd_loss, epoch)
            trial_writer.add_scalar('Loss/S_High', avg_s_high_loss, epoch)
            trial_writer.add_scalar('Loss/SC', avg_sc_loss, epoch)
            
            trial_writer.add_scalar('Training/Accuracy', train_acc, epoch)
            
            epoch_losses['total'].append(avg_loss)
            epoch_losses['supervised'].append(avg_sup_loss)
            epoch_losses['self_supervised'].append(avg_self_loss)
            epoch_losses['lfd'].append(avg_lfd_loss)
            epoch_losses['s_high'].append(avg_s_high_loss)
            epoch_losses['sc'].append(avg_sc_loss)
            epoch_losses['train_acc'].append(train_acc)  
            print(f"Trial {trial} Epoch {epoch} - "
                    f"Total: {avg_loss:.4f}, Sup: {avg_sup_loss:.4f}, Self: {avg_self_loss:.4f}, "
                    f"LFD: {avg_lfd_loss:.4f}, S_high: {avg_s_high_loss:.4f}, SC: {avg_sc_loss:.4f}")
            print(f"   Train Accuracy: {train_acc:.4f} ({train_correct}/{train_total})")
            print(f"  Loss Contribution: CE={avg_sup_loss:.3f}, Self={avg_self_loss*args.lambda1:.3f} "
                    f"({avg_self_loss:.4f}×{args.lambda1}), Ratio={avg_self_loss*args.lambda1/avg_sup_loss:.2%}")
            if epoch % args.eval_freq == 0:
                val_loss,val_metrics = self.test_cls_graphlevel(encoder, classifier, val_loader,specific_graph_learner,attention_fusion,args,trial,'val',log_file=results_log,epoch=epoch)
                current_f1 = val_metrics['f1_macro']
                
                if lr_mode == 'plateau':
                   scheduler.step(current_f1)
                
                trial_writer.add_scalar('Validation/Loss', val_loss, epoch)
                trial_writer.add_scalar('Validation/Accuracy', val_metrics['acc'], epoch)
                trial_writer.add_scalar('Validation/F1_Macro', val_metrics['f1_macro'], epoch)
                trial_writer.add_scalar('Validation/F1_Micro', val_metrics['f1_micro'], epoch)
                trial_writer.add_scalar('Validation/Precision_Macro', val_metrics['precision_macro'], epoch)
                trial_writer.add_scalar('Validation/Recall_Macro', val_metrics['recall_macro'], epoch)
                trial_writer.add_scalar('Validation/AUC_Macro', val_metrics['auc_macro'], epoch)
                trial_writer.add_scalar('Training/LearningRate', optimizer.param_groups[0]['lr'], epoch)
                
                
                trial_writer.add_scalars('Accuracy_Comparison', {
                    'Train': train_acc,
                    'Validation': val_metrics['acc']
                }, epoch)
                
                
                per_class_f1 = val_metrics.get('per_class_f1', [0]*5)
                print(f"  Val Accuracy: {val_metrics['acc']:.4f} | Train-Val Gap: {train_acc - val_metrics['acc']:.4f}")
                print(f"  Per-class F1: {[f'{f:.2f}' for f in per_class_f1]}")
                
                if max(per_class_f1) - min(per_class_f1) > 0.5:
                    print(f"   WARNING: Severe class imbalance detected! Max F1={max(per_class_f1):.2f}, Min F1={min(per_class_f1):.2f}")
                
                if current_f1> best_val:
                    best_val = current_f1
                    best_val_metrics=val_metrics
                    best_state = {
                        'encoder': copy.deepcopy(encoder.state_dict()),
                        'classifier': copy.deepcopy(classifier.state_dict()),
                        'specific': copy.deepcopy(specific_graph_learner.state_dict()),
                        'attention_fusion': copy.deepcopy(attention_fusion.state_dict()),
                        'optimizer': copy.deepcopy(optimizer.state_dict())
                    }
                    patience_counter = 0 
                    print(f"  New best F1: {best_val:.4f} (saved model)")
                else:
                    patience_counter += 1
                    print(f"  No improvement (current F1: {current_f1:.4f}, best F1: {best_val:.4f})") 

        if best_val_metrics is not None:
            val_results.append({
                'trial': trial,
                'acc': best_val_metrics['acc'],
                'metrics': best_val_metrics
            })
            if lr_mode == 'cosine':
                scheduler.step()
        self.plot_loss_curves(epoch_losses, trial, trial_writer)
        trial_writer.close()
        if best_state is not None:
            encoder.load_state_dict(best_state['encoder'])
            classifier.load_state_dict(best_state['classifier'])
            specific_graph_learner.load_state_dict(best_state['specific'])
            attention_fusion.load_state_dict(best_state['attention_fusion'])
            print(f"Restored best model with F1_macro: {best_val:.4f} for testing")
        test_loss, test_metrics = self.test_cls_graphlevel(encoder, classifier, test_loader,specific_graph_learner,attention_fusion,args,trial,'test',log_file=results_log,epoch=None)
        test_results.append({
            'trial': trial,
            'acc': test_metrics['acc'],
            'metrics': test_metrics
        })
        if args.downstream_task == 'classification' and len(test_results) > 0:
           
            results_log.write("\n\n")
            results_log.write(f"{'='*80}\n")
            results_log.write("FINAL SUMMARY ACROSS ALL TRIALS\n")
            results_log.write(f"{'='*80}\n")
            results_log.flush()
            self.print_results(val_results, test_results, log_file=results_log)
       
        results_log.close()
        print(f"\nAll results have been saved to: {results_log_path}")

    def plot_loss_curves(self, epoch_losses, trial, writer):

        
        epochs = list(range(1, len(epoch_losses['total']) + 1))
      
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(epochs, epoch_losses['total'], 'b-', label='Total Loss', linewidth=2)
        plt.plot(epochs, epoch_losses['supervised'], 'r-', label='Supervised Loss', linewidth=2)
        plt.plot(epochs, epoch_losses['self_supervised'], 'g-', label='Self-Supervised Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Main Loss Components')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        plt.plot(epochs, epoch_losses['lfd'], 'c-', label='LFD Loss', linewidth=2)
        plt.plot(epochs, epoch_losses['s_high'], 'm-', label='S_High Loss', linewidth=2)
        plt.plot(epochs, epoch_losses['sc'], 'y-', label='SC Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Self-Supervised Loss Components')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 3)
        plt.semilogy(epochs, epoch_losses['total'], 'b-', label='Total Loss', linewidth=2)
        plt.semilogy(epochs, epoch_losses['supervised'], 'r-', label='Supervised Loss', linewidth=2)
        plt.semilogy(epochs, epoch_losses['self_supervised'], 'g-', label='Self-Supervised Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss (log scale)')
        plt.title('Main Loss Components (Log Scale)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 4)
        plt.semilogy(epochs, epoch_losses['lfd'], 'c-', label='LFD Loss', linewidth=2)
        plt.semilogy(epochs, epoch_losses['s_high'], 'm-', label='S_High Loss', linewidth=2)
        plt.semilogy(epochs, epoch_losses['sc'], 'y-', label='SC Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss (log scale)')
        plt.title('Self-Supervised Loss Components (Log Scale)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
 
        plt.savefig(f'loss_curves_trial_{trial}.png', dpi=300, bbox_inches='tight')
        writer.add_figure('Loss_Curves/Summary', plt.gcf())
        plt.close()
  
        if 'train_acc' in epoch_losses and len(epoch_losses['train_acc']) > 0:
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, epoch_losses['train_acc'], 'b-', label='Training Accuracy', linewidth=2, marker='o')
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Accuracy', fontsize=12)
            plt.title('Training Accuracy over Epochs', fontsize=14, fontweight='bold')
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.ylim([0, 1.0])  
            
            
            plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='50% baseline')
            plt.axhline(y=0.8, color='g', linestyle='--', alpha=0.5, label='80% target')
            plt.legend(fontsize=10)
            
            plt.tight_layout()
            plt.savefig(f'train_acc_curve_trial_{trial}.png', dpi=300, bbox_inches='tight')
            writer.add_figure('Training/Accuracy_Curve', plt.gcf())
            plt.close()
            print(f"Training accuracy curve saved for trial {trial}")
        
        print(f"Loss curves saved for trial {trial}")
    def print_results(self, val_results, test_results, log_file=None):

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
       
        output = "FINAL RESULTS ACROSS ALL TRIALS\n"
        output += "="*60 + "\n"
        for metric, (mean, std) in stats.items():
            output += f"{metric.replace('_', ' ').title():<20}: {mean:.4f} ± {std:.4f}\n"
     
        print(output.rstrip())
        if log_file:
            log_file.write(output)
            log_file.flush()

        result_file = f"results/result_{args.dataset}_Class.txt"
        with open(result_file, "w") as fh:
            fh.write("FINAL RESULTS ACROSS ALL TRIALS\n")
            fh.write("="*60 + "\n")
            for metric, (mean, std) in stats.items():
                line = f"{metric.replace('_', ' ').title():<20}: {mean:.4f} ± {std:.4f}\n"
                fh.write(line)

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