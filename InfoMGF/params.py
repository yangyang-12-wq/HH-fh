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
    parser.add_argument('--gpu', type=int, default=6,
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
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Maximum training epochs (reduced to 500 to prevent overfitting)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (increased for better convergence)')
    parser.add_argument('--w_decay', type=float, default=1e-4,
                        help='Weight decay (increased to 5e-4 for stronger regularization)')
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping patience (increased to allow more exploration)')
    parser.add_argument('--classification_weight', type=float, default=5.0,
                        help='Weight multiplier for classification loss')
    parser.add_argument('--min_ce_loss_threshold', type=float, default=0.3,
                        help='Minimum CE loss threshold before allowing early stopping (relaxed from 0.05 to be more achievable)')
    parser.add_argument('--min_training_epochs', type=int, default=30,
                        help='Minimum number of training epochs before considering early stopping')
 
    
    # 模型架构参数
    parser.add_argument('--nlayers', type=int, default=2,
                        help='Number of GNN layers (restored to 2 after fixing over-normalization)')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension (reduced to 32 to combat overfitting)')
    parser.add_argument('--rep_dim', type=int, default=64,
                        help='Representation dimension')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate (balanced regularization)')
    
    # 图学习参数
    parser.add_argument('--k', type=int, default=20,
                        help='Number of nearest neighbors (reduced to 10 to avoid over-smoothing in 53-node graphs)')
    parser.add_argument('--dropedge_rate', type=float, default=0.0,
                        help='Edge dropout rate (set to 0 to avoid over-sparsification)')
    parser.add_argument('--activation_learner', type=str, default='tanh',
                        choices=['relu', 'tanh'], help='Activation function for graph learner')
    parser.add_argument('--r', type=int, default=2,
                        help='Number of AGG layers')
  
    
    # 在 params.py 中添加缺失的参数
    parser.add_argument('--sparse', type=bool, default=False, 
                        help='Whether to use sparse graph representation')
    parser.add_argument('--ntrials', type=int, default=5,
                        help='Number of experimental trials')
    parser.add_argument('--nlayer_gnn', type=int, default=2,
                        help='Number of GNN encoder layers')
    parser.add_argument('--emb_dim', type=int, default=64,
                        help='Embedding dimension (reduced to 32 to combat overfitting)')
    parser.add_argument('--tau', type=float, default=1.0,
                        help='Temperature parameter for custom loss')
    parser.add_argument('--h', type=float, default=1.0,
                        help='H parameter for SC loss')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Weight for LFD loss (reduced to focus more on classification)')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='Weight for S_high loss (reduced to avoid over-regularization)')
    parser.add_argument('--gamma', type=float, default=1.0,
                        help='Weight for SC loss (reduced to focus on classification)')
    parser.add_argument('--lambda1', type=float, default=1.0,
                        help='Overall weight for self-supervised loss (drastically reduced from 0.8 to 0.1 to prioritize classification)')
    # parser.add_argument('--lambda_att', type=float, default=0.05,
    #                     help='Weight for attention balance loss in full loss mode (reduced)')
    # parser.add_argument('--lambda_view', type=float, default=0.05,
    #                     help='Weight for view diversity loss in full loss mode (reduced)')
    parser.add_argument('--eval_freq', type=int, default=10,
                        help='Validation evaluation frequency (every N epochs)')
    parser.add_argument('--downstream_task', type=str, default='classification',
                        help='Downstream task type')

    parser.add_argument('--scheduler_patience', type=int, default=30,
                        help='ReduceLROnPlateau patience (eval steps) - reduced for faster adaptation')
    parser.add_argument('--lr_schedule', type=str, default='plateau',
                        choices=['plateau', 'cosine'],
                        help='Learning rate schedule: plateau (ReduceLROnPlateau) or cosine (warmup+cosine)')
    parser.add_argument('--warmup_epochs', type=int, default=70,
                        help='Warmup epochs before cosine annealing (only when lr_schedule=cosine)')
    parser.add_argument('--min_lr', type=float, default=1e-5,
                        help='Minimum learning rate for cosine annealing')
   
  
    parser.add_argument('--label_mode', type=str, default='binary',
                        choices=['multi', 'binary'],
                        help='How to load labels: multi-class or binary (0 vs others)')
    
    # 损失函数模式
    parser.add_argument('--loss_mode', type=str, default='ce_only',
                        choices=['full', 'ce_only'],
                        help='Loss mode: full (all losses) or ce_only (classification loss only) - DEFAULT ce_only for stability')
    parser.add_argument('--smart_resample', type=bool, default=True,
                        help='Whether to apply smart resampling for binary classification (train set only) - DEFAULT False to check raw data')
    parser.add_argument('--use_original_only', action='store_true',
                        help='Use only original samples, filtering out all _aug augmented samples (train only). Default: False (includes augmented samples).')
    parser.add_argument('--balance_strategy', type=str, default='downsample_class1',
                        choices=['upsample_class0', 'downsample_class1'],
                        help='Strategy for balancing classes: upsample_class0 (increase Class 0) or downsample_class1 (proportionally sample Class 1)')
    parser.add_argument('--attention_use_layer_norm', type=bool, default=True,
                    help='Whether to apply LayerNorm inside the attention fusion module')
    
   
    args = parser.parse_args()
    return args



