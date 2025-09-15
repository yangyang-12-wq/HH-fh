import argparse
import copy
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

from data_loader import RestingDataModule
from model import GCN, GCL, AGG
from graph_learner import *
from utils import *
from params import *
from augment import *
from sklearn.cluster import KMeans
from kmeans_pytorch import kmeans as KMeans_py
from sklearn.metrics import f1_score
from tqdm import tqdm
import random

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



    def evaluate_finetuned(self, model, fused_graph_learner, loader, args, final_test=False):
        model.eval()
        fused_graph_learner.eval()

        all_preds = []
        all_labels = []

        for batch in loader:
            X_batch = batch['X'].to(self.device)
            A_intra_batch = batch['A_intra'].to(self.device)
            A_global_batch = batch['A_global'].to(self.device)
            labels = batch['labels']
        
            batch_size = X_batch.shape[0]

        # 用于收集当前批次中每个样本的logits
            logits_list = []
        
        # 循环处理批次中的每个样本
            for i in range(batch_size):

                features = X_batch[i]
                adjs = [A_intra_batch[i], A_global_batch[i]]

                adjs_norm = [symmetrize_and_normalize(adj.unsqueeze(0)).squeeze(0) for adj in adjs]
                view_features = AGG([features for _ in adjs_norm], adjs_norm, args.r)
                view_features.append(features)
            
                fused_embedding_nodes = fused_graph_learner(torch.cat(view_features, dim=1))
                learned_fused_adj = fused_graph_learner.graph_process(fused_embedding_nodes)
                learned_fused_adj = graph_to_dense_if_needed(learned_fused_adj, self.device)
                learned_fused_adj = symmetrize_and_normalize(learned_fused_adj.unsqueeze(0)).squeeze(0)
            
        
                logits_sample = model.forward_classify(features, learned_fused_adj)
                logits_list.append(logits_sample)
            logits_batch = torch.stack(logits_list)
            preds = torch.argmax(logits_batch, dim=1).cpu()

            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
        
    # 使用sklearn的函数计算准确率和F1分数
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        if final_test:
            precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
            recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
            return acc, f1, precision, recall
        else:
            return acc, f1
    def loss_discriminator(self, discriminator, model, specific_graph_learner, features, view_features, adjs, optimizer_discriminator):

        optimizer_discriminator.zero_grad()

        learned_specific_adjs = []
        for i in range(len(adjs)):
            specific_adjs_embedding = specific_graph_learner[i](view_features[i])
            learned_specific_adj = specific_graph_learner[i].graph_process(specific_adjs_embedding)
            learned_specific_adjs.append(learned_specific_adj)

        z_specific_adjs = [model(features, learned_specific_adjs[i]) for i in range(len(adjs))]

        adjs_aug = graph_generative_augment(adjs, features, discriminator, sparse=args.sparse)
        z_aug_adjs = [model(features, adjs_aug[i]) for i in range(len(adjs))]
        loss_dis = discriminator.cal_loss_dis(z_aug_adjs, z_specific_adjs, view_features)

        loss_dis.backward()
        optimizer_discriminator.step()
        return loss_dis

    def loss_gcl(self, model, specific_graph_learner, fused_graph_learner, features, view_features, adjs,
                 optimizer, discriminator=None):
        optimizer.zero_grad()

        learned_specific_adjs = []
        for i in range(len(adjs)):
            specific_adjs_embedding = specific_graph_learner[i](view_features[i])
            learned_specific_adj = specific_graph_learner[i].graph_process(specific_adjs_embedding)
            learned_specific_adjs.append(learned_specific_adj)

        fused_embedding = fused_graph_learner(torch.cat(view_features, dim=1))
        learned_fused_adj = fused_graph_learner.graph_process(fused_embedding)
        z_specific_adjs = [model(features, learned_specific_adjs[i]) for i in range(len(adjs))]
        z_fused_adj = model(features, learned_fused_adj)

        if args.augment_type == 'random':
            adjs_aug = graph_augment(adjs, args.dropedge_rate, training=self.training, sparse=args.sparse)
        elif args.augment_type == 'generative':
            adjs_aug = graph_generative_augment(adjs, features, discriminator, sparse=args.sparse)
        if args.sparse:
            for i in range(len(adjs)):
                adjs_aug[i].edata['w'] = adjs_aug[i].edata['w'].detach()
        else:
            adjs_aug = [a.detach() for a in adjs_aug]
        z_aug_adjs = [model(features, adjs_aug[i]) for i in range(len(adjs))]


        if args.contrast_batch_size:
            node_idxs = list(range(features.shape[0]))
            random.shuffle(node_idxs)
            batches = split_batch(node_idxs, args.contrast_batch_size)
            loss = 0
            for batch in batches:
                weight = len(batch) / features.shape[0]
                loss += model.cal_loss([z[batch] for z in z_specific_adjs], [z[batch] for z in z_aug_adjs], z_fused_adj[batch]) * weight
        else:
            loss = model.cal_loss(z_specific_adjs, z_aug_adjs, z_fused_adj)

        loss.backward()
        optimizer.step()

        return loss

    def train(self, args):
        print(args)
        device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
        self.setup_seed(args.seed)

        datamodule = RestingDataModule(
          processed_dir=args.out_dir,
          feature_path=args.feature_path,
          batch_size=args.batch_size,
          use_npy=True,
          preload=args.preload,
          num_workers=args.num_workers
        )
        datamodule.setup()
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        test_loader = datamodule.test_dataloader()

        sample0 = datamodule.trainset[0]
        nfeats = sample0['X'].shape[1]
        nclasses = len(set([d['label'].item() for d in datamodule.trainset]))
        print(f"Data loaded via DataLoader. Node feature dim: {nfeats}, Num classes: {nclasses}")
        
        num_views = 2 # intra, global
        specific_graph_learner = [ATT_learner(2, nfeats, args.k, 6, args.dropedge_rate, args.sparse, args.activation_learner) for _ in range(num_views)]
        fused_graph_learner = ATT_learner(2, nfeats * (num_views + 1), args.k, 6, args.dropedge_rate, args.sparse, args.activation_learner)
        model = GCL(
            nlayers=args.nlayers, in_dim=nfeats, hidden_dim=args.hidden_dim,
            emb_dim=args.rep_dim, proj_dim=args.proj_dim,
            dropout=args.dropout, sparse=args.sparse, num_g=num_views,
            num_classes=nclasses
        )
        model.to(device)
        specific_graph_learner = [m.to(device) for m in specific_graph_learner]
        fused_graph_learner.to(device)
        params_pretrain = [{'params': l.parameters()} for l in specific_graph_learner] + \
        [{'params': fused_graph_learner.parameters()}, {'params': model.parameters()}]
        optimizer_pretrain = torch.optim.Adam(params_pretrain, lr=args.lr, weight_decay=args.w_decay)
        optimizer_finetune = torch.optim.Adam(model.parameters(), lr=args.lr_finetune, weight_decay=args.w_decay) 
        if args.augment_type == 'generative':
            print("Using Generative Augmentation.")
            discriminator = Discriminator(
                input_dim=nfeats, 
                hidden_dim=args.hidden_dim, # 可以复用或设置新参数
                rep_dim=args.rep_dim, 
                aug_lambda=args.aug_lambda # 需要新增此参数
            ).to(device)
            optimizer_discriminator = torch.optim.Adam(
                discriminator.parameters(), 
                lr=args.lr_dis, # 需要新增此参数
                weight_decay=args.w_decay
            )
        else:
            print("Using Random Augmentation.")
            discriminator = None
            optimizer_discriminator = None
        optimizer_pretrain = torch.optim.Adam(params_pretrain, lr=args.lr, weight_decay=args.w_decay)

        optimizer_finetune = torch.optim.Adam(model.parameters(), lr=args.lr_finetune, weight_decay=args.w_decay)
        print("\n--- STAGE 1: Self-supervised Pre-training ---")
        for epoch in range(1, args.pretrain_epochs + 1):
            model.train()
            [learner.train() for learner in specific_graph_learner]
            fused_graph_learner.train()
            if discriminator:
                discriminator.eval()
            epoch_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Pre-train Epoch {epoch}[GCL]")

            for batch in pbar:
                X_batch, A_intra_batch, A_global_batch = batch['X'].to(device), batch['A_intra'].to(device), batch['A_global'].to(device)
                batch_size = X_batch.shape[0]
            
                # 对Batch中的每个样本独立计算GCL损失
                total_batch_loss = 0
                for i in range(batch_size):
                    features = X_batch[i]
                    adjs = [A_intra_batch[i], A_global_batch[i]]
                    adjs_norm = [symmetrize_and_normalize(adj.unsqueeze(0)).squeeze(0) for adj in adjs] # 需要symmetrize_and_normalize函数
                
                    view_features = AGG([features for _ in adjs_norm], adjs_norm, args.r)
                    view_features.append(features)
                
                # loss_gcl的逻辑现在逐样本执行
                    learned_specific_adjs = [specific_graph_learner[j].graph_process(specific_graph_learner[j](view_features[j])) for j in range(num_views)]
                    fused_embedding = fused_graph_learner(torch.cat(view_features, dim=1))
                    learned_fused_adj = fused_graph_learner.graph_process(fused_embedding)

                    z_specific = [model(features, learned_specific_adjs[j]) for j in range(num_views)]
                    z_fused = model(features, learned_fused_adj)
                    if args.augment_type == 'generative':
                        adjs_aug = graph_generative_augment(adjs, features, discriminator, sparse=args.sparse)
                        adjs_aug = [aug.detach() for aug in adjs_aug]
                    else:
                       adjs_aug = [graph_augment(adj, args.dropedge_rate) for adj in adjs]
                    z_aug = [model(features, adjs_aug[j]) for j in range(num_views)]
                
                    loss_sample = model.cal_loss(z_specific, z_aug, z_fused)
                    total_batch_loss += loss_sample

                if batch_size > 0:
                    optimizer_pretrain.zero_grad()
                    avg_batch_loss = total_batch_loss / batch_size
                    avg_batch_loss.backward()
                    optimizer_pretrain.step()
                    epoch_loss += avg_batch_loss.item()
                    pbar.set_postfix(cl_loss=avg_batch_loss.item())
            if args.augment_type=='generative':
                model.eval()
                [learner.eval() for learner in specific_graph_learner]
                fused_graph_learner.eval()
                discriminator.train()
                epoch_dis_loss = 0.0
                pbar_dis = tqdm(train_loader, desc=f"Pre-train Epoch {epoch}[Discriminator]")
                for batch in pbar_dis:
                    X_batch, A_intra_batch, A_global_batch = batch['X'].to(device), batch['A_intra'].to(device), batch['A_global'].to(device)
                    batch_size = X_batch.shape[0]
                
                    total_dis_loss = 0
                    for i in range(batch_size):
                        features = X_batch[i]
                        adjs = [A_intra_batch[i], A_global_batch[i]]
                        while torch.no_grad():
                            adjs_norm = [symmetrize_and_normalize(adj.unsqueeze(0)).squeeze(0) for adj in adjs]
                            view_features = AGG([features for _ in adjs_norm], adjs_norm, args.r)
                            view_features.append(features)
                            learned_specific_adjs = [specific_graph_learner[j].graph_process(specific_graph_learner[j](view_features[j])) for j in range(num_views)]
                            z_specific = [model(features, learned_specific_adjs[j]) for j in range(num_views)]
                        adjs_aug_edges=adjs[0].to_sparse().coalesce().indices()
                        adjs_aug_weights=discriminator(features, adjs_aug_edges)
                        adjs_aug = [torch.sparse.FloatTensor(adjs_aug_edges, adjs_aug_weights.detach(), adjs[j].shape) for j in range(num_views)]
                        z_aug = [model(features, adjs_aug[j]) for j in range(num_views)]
                        loss_dis_sample=discriminator.cal_loss_dis(adjs_aug=a.to_dense() for a in adjs_aug, adjs_original=[graph_to_dense_if_needed(a,device) for a in learned_specific_adjs],
                         view_features=[features],
                         node_reps_aug=z_aug,
                         node_reps_orig=z_specific
                        )
                        total_dis_loss+=loss_dis_sample
                    if batch_size>0:
                        optimizer_discriminator.zero_grad()
                        avg_batch_dis_loss = total_dis_loss / batch_size
                        avg_batch_dis_loss.backward()
                        optimizer_discriminator.step()
                        epoch_dis_loss += avg_batch_dis_loss.item()
                        pbar_dis.set_postfix(dis_loss=avg_batch_dis_loss.item())
        avg_epoch_cl_loss = epoch_loss / len(train_loader)
        print_str = f"Pre-train Epoch {epoch} | Avg CL Loss: {avg_epoch_cl_loss:.4f}"
        if args.augment_type == 'generative':
            avg_epoch_dis_loss = epoch_dis_loss / len(train_loader)
            print_str += f" | Avg DIS Loss: {avg_epoch_dis_loss:.4f}"
        print(print_str)
        
        print("Pre-training Finished")
        print("\n--- STAGE 2: Supervised Fine-tuning ---")
        best_val_f1 = 0
        patience_counter = 0

        for epoch in range(1, args.finetune_epochs + 1):
            model.train()
            fused_graph_learner.train() # 图学习器也参与微调

            pbar = tqdm(train_loader, desc=f"Finetune Epoch {epoch}")
            for batch in pbar:
                X_batch, A_intra_batch, A_global_batch, labels_batch = batch['X'].to(device), batch['A_intra'].to(device), batch['A_global'].to(device), batch['labels'].to(device)
                batch_size = X_batch.shape[0]
            
                logits_list = []
                for i in range(batch_size):
                    features, adjs = X_batch[i], [A_intra_batch[i], A_global_batch[i]]
                    adjs_norm = [symmetrize_and_normalize(adj.unsqueeze(0)).squeeze(0) for adj in adjs]
                    view_features = AGG([features for _ in adjs_norm], adjs_norm, args.r)
                    view_features.append(features)
                
                    fused_embedding_nodes = fused_graph_learner(torch.cat(view_features, dim=1))
                    learned_fused_adj = fused_graph_learner.graph_process(fused_embedding_nodes)
                    learned_fused_adj = graph_to_dense_if_needed(learned_fused_adj, device) # 需要graph_to_dense_if_needed函数
                    learned_fused_adj = symmetrize_and_normalize(learned_fused_adj.unsqueeze(0)).squeeze(0)

                    logits_sample = model.forward_classify(features, learned_fused_adj)
                    logits_list.append(logits_sample)

                logits_batch = torch.stack(logits_list)
                loss_ce_batch = F.cross_entropy(logits_batch, labels_batch)

                optimizer_finetune.zero_grad()
                loss_ce_batch.backward()
                optimizer_finetune.step()
                pbar.set_postfix(ce_loss=loss_ce_batch.item())

        # [修改] 6. 评估微调后的模型
            val_acc, val_f1 = self.evaluate_finetuned(model, fused_graph_learner, val_loader, args)
            print(f"Finetune Epoch {epoch} | Val F1: {val_f1:.4f} | Val Acc: {val_acc:.4f}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                torch.save(model.state_dict(), args.model_save_path)
                print(f"Saved best model to {args.model_save_path}")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print("Early stopping triggered.")
                    break

    # [修改] 7. 最终测试
        print(f"\nLoading best model from {args.model_save_path} for final testing...")
        if os.path.exists(args.model_save_path):
            model.load_state_dict(torch.load(args.model_save_path))
            test_acc, test_f1, _, _ = self.evaluate_finetuned(model, fused_graph_learner, test_loader, args, final_test=True)
            print("-" * 50)
            print(f"Final Test Accuracy: {test_acc:.4f}")
            print(f"Final Test F1-Macro: {test_f1:.4f}")
            print("-" * 50)

if __name__ == '__main__':

        experiment = Experiment()
        experiment.train(args)