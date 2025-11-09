import argparse
import torch
import sys

def set_params():
    """设置实验参数"""
    parser = argparse.ArgumentParser(description='InfoMGF: Single-Stage End-to-End fNIRS Classification Framework')
    
    # 基础实验设置
    parser.add_argument('--dataset', type=str, default='fnirs', 
                        help='Dataset name')
    parser.add_argument('--dataset_name', type=str, default='fnirs_exp',
                        help='Dataset name for experiment tracking')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device number (used when multi_gpu=False)')
    parser.add_argument('--multi_gpu', type=bool, default=True,
                        help='Whether to use multiple GPUs with DataParallel')
    parser.add_argument('--gpu_ids', type=str, default='0,1,2,3,4,5,6,7',
                        help='GPU IDs to use (comma-separated, e.g., "0,1,2,3")')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # 数据相关参数
    parser.add_argument('--out_dir', type=str, default='/data1/cuichenyang/processed_data',
                        help='Directory containing processed data files')
    parser.add_argument('--feature_path', type=str, default='features',
                        help='Path to feature files')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size (increased to 4 for stable gradient estimation)')
    parser.add_argument('--preload', type=bool, default=True,
                        help='Whether to preload data')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # 训练相关参数
    parser.add_argument('--epochs', type=int, default=500,
                        help='Maximum training epochs (very large number - essentially unlimited until target achieved)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (increased to 0.001 to match larger batch size)')
    parser.add_argument('--w_decay', type=float, default=1e-4,
                        help='Weight decay (reduced to allow model to fit training data)')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--classification_weight', type=float, default=5.0,
                        help='Weight multiplier for classification loss')
    parser.add_argument('--min_ce_loss_threshold', type=float, default=0.3,
                        help='Minimum CE loss threshold before allowing early stopping (relaxed from 0.05 to be more achievable)')
    parser.add_argument('--min_training_epochs', type=int, default=30,
                        help='Minimum number of training epochs before considering early stopping')
    
    # 模型架构参数
    parser.add_argument('--nlayers', type=int, default=1,
                        help='Number of GNN layers (restored to 2 after fixing over-normalization)')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension')
    parser.add_argument('--rep_dim', type=int, default=64,
                        help='Representation dimension')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate (reduced to 0.3 to allow better training set fitting)')
    
    # 图学习参数
    parser.add_argument('--k', type=int, default=20,
                        help='Number of nearest neighbors (reduced to 10 to avoid over-smoothing in 53-node graphs)')
    parser.add_argument('--dropedge_rate', type=float, default=0.0,
                        help='Edge dropout rate (set to 0 to avoid over-sparsification)')
    parser.add_argument('--activation_learner', type=str, default='tanh',
                        choices=['relu', 'tanh'], help='Activation function for graph learner')
    parser.add_argument('--r', type=int, default=2,
                        help='Number of AGG layers')
    
    # 重新平衡的损失权重参数 (基于实际损失值的数量级计算)
    parser.add_argument('--lambda_smooth', type=float, default=1000.0,
                        help='Weight for graph smoothness loss (λ ≈ 0.5/0.0003 ≈ 1666, rounded to 1000)')
    parser.add_argument('--lambda_consistency', type=float, default=25.0,
                        help='Weight for multi-view consistency loss (increased from 15 to 25 for better balance)')
    parser.add_argument('--lambda_distill', type=float, default=0.01,
                        help='Weight for distillation loss (may be replaced by L_LFD)')
    parser.add_argument('--lambda_lfd', type=float, default=1000.0,
                        help='Weight for LFD/Dirichlet energy loss (λ ≈ 0.5/0.0006 ≈ 833, rounded to 1000)')
    parser.add_argument('--temperature', type=float, default=0.5,
                        help='Temperature parameter for graph learning sparsity control (lower = sparser graphs)')
    parser.add_argument('--loss_topk', type=int, default=10,
                        help='Top-k neighbors to keep when computing graph-based losses')
    
    # 模型保存参数
    parser.add_argument('--model_save_path', type=str, default='best_model.pth',
                        help='Path to save the best model')
    # 在 params.py 中添加缺失的参数
    parser.add_argument('--sparse', type=bool, default=False, 
                        help='Whether to use sparse graph representation')
    parser.add_argument('--ntrials', type=int, default=5,
                        help='Number of experimental trials')
    parser.add_argument('--nlayer_gnn', type=int, default=2,
                        help='Number of GNN encoder layers')
    parser.add_argument('--emb_dim', type=int, default=64,
                        help='Embedding dimension')
    parser.add_argument('--tau', type=float, default=1.0,
                        help='Temperature parameter for custom loss')
    parser.add_argument('--h', type=float, default=1.0,
                        help='H parameter for SC loss')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Weight for LFD loss')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='Weight for S_high loss (reduced to 0.5 after fixing calculation)')
    parser.add_argument('--gamma', type=float, default=1.0,
                        help='Weight for SC loss')
    parser.add_argument('--lambda1', type=float, default=1.0,
                        help='Overall weight for self-supervised loss (reduced to 1.0 to prioritize CE loss)')
    parser.add_argument('--eval_freq', type=int, default=10,
                        help='Validation evaluation frequency (every N epochs)')
    parser.add_argument('--downstream_task', type=str, default='classification',
                        help='Downstream task type')
    
    args = parser.parse_args()
    return args


