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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import random
import os

EOS = 1e-10
args = set_params()


class Experiment:
    def __init__(self):
        super(Experiment, self).__init__()

    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        np.random.seed(seed)
        random.seed(seed)

    def _forward_pass(self, X_batch,
                      actual_specific_learners, actual_fused_learner, actual_model,
                      return_for_loss=False):

        intra_emb = actual_specific_learners[0](X_batch)
        A_intra_refined = actual_specific_learners[0].graph_process(intra_emb, perform_normalization=True)

        global_emb = actual_specific_learners[1](X_batch)
        A_global_refined = actual_specific_learners[1].graph_process(global_emb, perform_normalization=True)

        z_intra_refined = actual_model.encoder_batch(X_batch, A_intra_refined)
        z_global_refined = actual_model.encoder_batch(X_batch, A_global_refined)

        fusion_features = torch.cat([z_intra_refined, z_global_refined, X_batch], dim=-1)

        fused_embedding = actual_fused_learner(fusion_features)
        learned_fused_adj_raw = actual_fused_learner.graph_process(fused_embedding, perform_normalization=False)
        learned_fused_adj_norm = symmetrize_and_normalize(learned_fused_adj_raw)

        z_teacher_nodes, logits_batch = actual_model.forward_all_batch(X_batch, learned_fused_adj_norm)

        if not return_for_loss:
            return logits_batch
        else:
            # For consistency loss, we compare the final fused embedding with the refined view embeddings
            z_specific_nodes_list = [z_intra_refined, z_global_refined]
            # The student MLP is still used for a form of regularization
            z_student_nodes = actual_model.forward_student_batch(X_batch)

            return logits_batch, learned_fused_adj_raw, z_teacher_nodes, z_student_nodes, z_specific_nodes_list

    def evaluate_finetuned(self, model, specific_learners, fused_graph_learner, loader, args, final_test=False):
        model.eval()
        [learner.eval() for learner in specific_learners]
        fused_graph_learner.eval()

        all_preds, all_labels = [], []

        actual_model = model.module if isinstance(model, torch.nn.DataParallel) else model
        actual_specific_learners = [learner.module if isinstance(learner, torch.nn.DataParallel) else learner for learner in specific_learners]
        actual_fused_learner = fused_graph_learner.module if isinstance(fused_graph_learner, torch.nn.DataParallel) else fused_graph_learner

        device = next(model.parameters()).device

        with torch.no_grad():
            for batch in loader:
                X_batch = batch['X'].to(device)
                labels = batch['labels']

                logits_batch = self._forward_pass(
                    X_batch,
                    actual_specific_learners, actual_fused_learner, actual_model,
                    return_for_loss=False
                )

                preds = torch.argmax(logits_batch, dim=1).cpu()
                all_preds.extend(preds.numpy())
                all_labels.extend(labels.numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        if final_test:
            precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
            recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
            return acc, f1, precision, recall
        else:
            return acc, f1


    def _calculate_dirichlet_energy_loss_robust(self, z_batch, adj_raw_batch):
        B, N, feat_dim = z_batch.shape
        eye = torch.eye(N, device=adj_raw_batch.device).unsqueeze(0)
        adj_clean = adj_raw_batch * (1 - eye)
        degrees = torch.sum(adj_clean, dim=-1)
        D_inv_sqrt = torch.diag_embed(1.0 / (torch.sqrt(degrees) + EOS))
        L_sym = eye - torch.bmm(torch.bmm(D_inv_sqrt, adj_clean), D_inv_sqrt)

        loss_matrix = torch.bmm(z_batch.transpose(1, 2), torch.bmm(L_sym, z_batch)) 
        dirichlet_energy = torch.einsum('bii->b', loss_matrix) 

        normalized_energy = dirichlet_energy / (N * feat_dim)

        return normalized_energy.mean()

    def _calculate_smoothness_loss_batch_robust(self, adj_raw_batch, features_batch):

        B, N, feat_dim = features_batch.shape

        eye = torch.eye(N, device=adj_raw_batch.device).unsqueeze(0)
        adj_clean = adj_raw_batch * (1 - eye)

        degrees = torch.sum(adj_clean, dim=-1)

        D_inv_sqrt = torch.diag_embed(1.0 / (torch.sqrt(degrees) + EOS))

        L_sym = eye - torch.bmm(torch.bmm(D_inv_sqrt, adj_clean), D_inv_sqrt)

        loss_matrix = torch.bmm(features_batch.transpose(1, 2), torch.bmm(L_sym, features_batch))
        smoothness = torch.einsum('bii->b', loss_matrix) # shape: (B,)

        normalized_smoothness = smoothness / (N * feat_dim)

        return normalized_smoothness.mean()

    def _calculate_consistency_loss_batch_robust(self, z_fused_batch, z_specific_list_batch, alpha: float = 0.1):
        mse_loss = 0.0
        for z_specific in z_specific_list_batch:
            mse_loss += F.mse_loss(z_fused_batch, z_specific)

        cohesion_loss = mse_loss / len(z_specific_list_batch)

        B, N, feat_dim = z_fused_batch.shape
        z_norm = z_fused_batch - z_fused_batch.mean(dim=1, keepdim=True)
        cov_matrix = torch.bmm(z_norm.transpose(1, 2), z_norm) / (N - 1 + 1e-6) # (B, feat_dim, feat_dim)
        off_diagonal_elements = cov_matrix * (1 - torch.eye(feat_dim, device=z_fused_batch.device).unsqueeze(0))
        separation_loss = torch.mean(off_diagonal_elements**2) # encourage off-diagonals to 0

        return cohesion_loss + alpha * separation_loss

    def train(self, args):
        # Minimal console output by default; enable verbose with args.debug
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu is not None else "cpu")
        self.device = device
        self.setup_seed(args.seed)

        datamodule = RestingDataModule(
            processed_dir=args.out_dir, feature_path=args.feature_path,
            batch_size=args.batch_size, use_npy=False, preload=args.preload,
            num_workers=args.num_workers
        )
        datamodule.setup()
        train_loader, val_loader, test_loader = datamodule.train_dataloader(), datamodule.val_dataloader(), datamodule.test_dataloader()

        sample0 = datamodule.trainset[0]
        nfeats = sample0['X'].shape[1]
        nclasses = len(set([d['label'].item() for d in datamodule.trainset]))

        num_views = 2
        gnn_emb_dim = args.rep_dim 
        specific_learner_isize = nfeats
        specific_graph_learner = [ATT_learner(2, specific_learner_isize, args.k, 6, args.dropedge_rate, args.activation_learner, emb_dim=32, temperature=args.temperature) for _ in range(num_views)]


        fused_learner_isize = gnn_emb_dim * num_views + nfeats
        fused_graph_learner = ATT_learner(2, fused_learner_isize, args.k, 6, args.dropedge_rate, args.activation_learner, emb_dim=32, temperature=args.temperature)
        
        model = GCL(
            nlayers=args.nlayers, in_dim=nfeats, hidden_dim=args.hidden_dim,
            emb_dim=gnn_emb_dim, dropout=args.dropout, num_classes=nclasses
        )
        
        model.to(device)
        specific_graph_learner = [m.to(device) for m in specific_graph_learner]
        fused_graph_learner.to(device)
        
        # Multi-GPU support (simple DataParallel)
        if args.multi_gpu and torch.cuda.is_available() and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
            specific_graph_learner = [torch.nn.DataParallel(m) for m in specific_graph_learner]
            fused_graph_learner = torch.nn.DataParallel(fused_graph_learner)

        # optimizer param groups
        all_params = []
        for m in specific_graph_learner:
            all_params.append({'params': m.parameters()})
        all_params.append({'params': fused_graph_learner.parameters()})
        all_params.append({'params': model.parameters()})

        optimizer = torch.optim.Adam(all_params, lr=args.lr, weight_decay=args.w_decay)

        # Optional mixed precision
        use_amp = getattr(args, 'use_amp', False)
        scaler = torch.cuda.amp.GradScaler() if use_amp else None

        # LR scheduler and early stopping
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=False)
        patience = getattr(args, 'early_stop_patience', 30)
        min_epochs = args.min_training_epochs

        writer = SummaryWriter(log_dir=f'runs/{args.dataset_name}_Improved_{datetime.now().strftime("%Y-%m-%d_%H-%M")}')
        best_val_f1 = -1.0
        epochs_no_improve = 0

        actual_model = model.module if isinstance(model, torch.nn.DataParallel) else model
        actual_specific_learners = [learner.module if isinstance(learner, torch.nn.DataParallel) else learner for learner in specific_graph_learner]
        actual_fused_learner = fused_graph_learner.module if isinstance(fused_graph_learner, torch.nn.DataParallel) else fused_graph_learner

        for epoch in range(1, args.epochs + 1):
            model.train()
            [learner.train() for learner in specific_graph_learner]
            fused_graph_learner.train()

            epoch_losses = {'ce': 0.0, 'smooth': 0.0, 'consistency': 0.0, 'lfd': 0.0, 'sparsity': 0.0}
            total_batches = len(train_loader)

            pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)

            for batch in pbar:
                X_batch = batch['X'].to(device)
                labels_batch = batch['labels'].to(device)

                if use_amp:
                    with torch.cuda.amp.autocast():
                        logits_batch, learned_fused_adj_raw, z_teacher_nodes, z_student_nodes, z_specific_nodes_list = self._forward_pass(
                            X_batch, actual_specific_learners, actual_fused_learner, actual_model, return_for_loss=True)

                        ce_loss = F.cross_entropy(logits_batch, labels_batch)
                        smooth_loss = self._calculate_smoothness_loss_batch_robust(learned_fused_adj_raw, X_batch)
                        consistency_loss = self._calculate_consistency_loss_batch_robust(z_teacher_nodes, z_specific_nodes_list)
                        lfd_loss = self._calculate_dirichlet_energy_loss_robust(z_teacher_nodes, learned_fused_adj_raw)
                        sparsity_loss = torch.mean(torch.abs(learned_fused_adj_raw))

                        total_loss = args.classification_weight * ce_loss + \
                                     args.lambda_smooth * smooth_loss + \
                                     args.lambda_consistency * consistency_loss + \
                                     args.lambda_lfd * lfd_loss + \
                                     getattr(args, 'lambda_sparsity', 0.0) * sparsity_loss

                    optimizer.zero_grad()
                    scaler.scale(total_loss).backward()
                    if getattr(args, 'grad_clip', 0.0) > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()

                else:
                    logits_batch, learned_fused_adj_raw, z_teacher_nodes, z_student_nodes, z_specific_nodes_list = self._forward_pass(
                        X_batch, actual_specific_learners, actual_fused_learner, actual_model, return_for_loss=True)

                    ce_loss = F.cross_entropy(logits_batch, labels_batch)
                    smooth_loss = self._calculate_smoothness_loss_batch_robust(learned_fused_adj_raw, X_batch)
                    consistency_loss = self._calculate_consistency_loss_batch_robust(z_teacher_nodes, z_specific_nodes_list)
                    lfd_loss = self._calculate_dirichlet_energy_loss_robust(z_teacher_nodes, learned_fused_adj_raw)
                    sparsity_loss = torch.mean(torch.abs(learned_fused_adj_raw))

                    total_loss = args.classification_weight * ce_loss + \
                                 args.lambda_smooth * smooth_loss + \
                                 args.lambda_consistency * consistency_loss + \
                                 args.lambda_lfd * lfd_loss + \
                                 getattr(args, 'lambda_sparsity', 0.0) * sparsity_loss

                    optimizer.zero_grad()
                    total_loss.backward()
                    if getattr(args, 'grad_clip', 0.0) > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()

                # accumulate
                epoch_losses['ce'] += ce_loss.item()
                epoch_losses['smooth'] += smooth_loss.item()
                epoch_losses['consistency'] += consistency_loss.item()
                epoch_losses['lfd'] += lfd_loss.item()
                epoch_losses['sparsity'] += sparsity_loss.item()

                # update progress bar
                pbar.set_postfix({
                    'tot_loss': f"{total_loss.item():.4f}",
                    'ce': f"{ce_loss.item():.4f}",
                    'f1_best': f"{best_val_f1:.4f}"
                })

            # average epoch losses
            for k in epoch_losses:
                epoch_losses[k] /= max(1, total_batches)

            # Validation
            val_acc, val_f1 = self.evaluate_finetuned(model, specific_graph_learner, fused_graph_learner, val_loader, args)

            # Scheduler step (monitor val_f1)
            scheduler.step(val_f1)

            # Logging
            writer.add_scalar('Loss/ce', epoch_losses['ce'], epoch)
            writer.add_scalar('Loss/smooth', epoch_losses['smooth'], epoch)
            writer.add_scalar('Loss/consistency', epoch_losses['consistency'], epoch)
            writer.add_scalar('Loss/lfd', epoch_losses['lfd'], epoch)
            writer.add_scalar('Loss/sparsity', epoch_losses['sparsity'], epoch)
            writer.add_scalar('Metrics/val_f1', val_f1, epoch)
            writer.add_scalar('Metrics/val_acc', val_acc, epoch)

            # concise console output
            if getattr(args, 'debug', False):
                print(f"Epoch {epoch} | ce: {epoch_losses['ce']:.4f} | smooth: {epoch_losses['smooth']:.6f} | consist: {epoch_losses['consistency']:.5f} | lfd: {epoch_losses['lfd']:.6f} | val_f1: {val_f1:.4f}")
            else:
                print(f"Epoch {epoch} | val_f1: {val_f1:.4f} | val_acc: {val_acc:.4f}")

            # checkpointing and early stopping
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                epochs_no_improve = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'specific_learners_state_dict': [learner.state_dict() for learner in specific_graph_learner],
                    'fused_learner_state_dict': fused_graph_learner.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_f1': best_val_f1,
                }, args.model_save_path)
            else:
                epochs_no_improve += 1

            if epoch >= min_epochs and epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch}. No improvement in val_f1 for {patience} epochs.")
                break

        # training end
        # load best model for final test
        if os.path.exists(args.model_save_path):
            ckpt = torch.load(args.model_save_path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            for sl, sd in zip(specific_graph_learner, ckpt['specific_learners_state_dict']):
                sl.load_state_dict(sd)
            fused_graph_learner.load_state_dict(ckpt['fused_learner_state_dict'])

        # final evaluation
        final_metrics = self.evaluate_finetuned(model, specific_graph_learner, fused_graph_learner, test_loader, args, final_test=True)
        if getattr(args, 'verbose_final', True):
            acc, f1, precision, recall = final_metrics
            print("Final Test Results | ", f"Acc: {acc:.4f} | F1: {f1:.4f} | Prec: {precision:.4f} | Rec: {recall:.4f}")

        writer.close()
        return final_metrics


if __name__ == '__main__':
    experiment = Experiment()
    experiment.train(args)
