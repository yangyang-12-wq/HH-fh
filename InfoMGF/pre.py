import os
import argparse
import pickle
from tqdm import tqdm
from collections import Counter
import shutil
from utils import *
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import random
from utils_graph_build import (
    make_region_map,
    compute_global_edges,
    build_intra_region_view_mi,
    build_global_view,
    brain_regions
)
from node_extractor import *

class DataAugmentor:
    @staticmethod
    def time_shifting(data):
        # 时间平移
        shift = random.randint(-200, 200)
        if isinstance(data, torch.Tensor):
            return torch.roll(data, shifts=shift, dims=0)
        else:  # numpy array
            return np.roll(data, shift=shift, axis=0)
    
    @staticmethod
    def time_reversal(data):
        # 时间反转
        if isinstance(data, torch.Tensor):
            return torch.flip(data, dims=[0])
        else:  # numpy array
            return np.flip(data, axis=0)

    @staticmethod
    def noise_injection(data):
        # 噪声注入
        if isinstance(data, torch.Tensor):
            noise = torch.randn_like(data) * 0.02
            return data + noise
        else:  # numpy array
            noise = np.random.normal(0, 0.02, data.shape)
            return data + noise
        
    @staticmethod
    def time_masking(data):
        data_copy = data.copy() if isinstance(data, np.ndarray) else data.clone()
        mask_len = random.randint(50, 300)
        if isinstance(data_copy, torch.Tensor):
            mask_start = random.randint(0, data_copy.size(0) - mask_len)
            data_copy[mask_start:mask_start + mask_len] = 0
        else:  # numpy array
            mask_start = random.randint(0, data_copy.shape[0] - mask_len)
            data_copy[mask_start:mask_start + mask_len] = 0
            
        return data_copy

    @staticmethod
    def augment_data(data):
        """随机选择一种增强方法"""
        augmentation_methods = [
            lambda x: x,  # 不增强
            DataAugmentor.time_shifting,
            DataAugmentor.noise_injection,
            DataAugmentor.time_masking,
            DataAugmentor.time_reversal
        ]
        method = random.choice(augmentation_methods)
        return method(data)

def oversample_training_data(rows, augment_minority=True):
    label_counts = Counter([r['labels'] for r in rows])
    max_count = max(label_counts.values())
    
    print(f"原始类别分布: {dict(label_counts)}")
    
    augmented_data = rows.copy()

    for label, count in label_counts.items():
        if count < max_count:
            minority_samples = [r for r in rows if r['labels'] == label]
            needed = max_count - count
            
            for i in range(needed):

                sample = random.choice(minority_samples).copy()

                if augment_minority:
                    sample['data'] = DataAugmentor.augment_data(sample['data'])
                    sample['id'] = f"{sample['id']}_aug_{i}"
                
                augmented_data.append(sample)
    
    new_counts = Counter([r['labels'] for r in augmented_data])
    print(f"过采样后类别分布: {dict(new_counts)}")
    print(f"总样本数: {len(augmented_data)} (原始: {len(rows)})")
    
    return augmented_data

def prepare_and_balance_data(feature_path, split, feature_type, augment_train=True):
    feature_data = load_feature_pickle(feature_path, split)
    rows = prepare_rows_from_feature_data(feature_data, feature_type)
    
    # 打印关键信息
    print(f"\n=== {split}集数据信息 ===")
    print(f"样本数量: {len(rows)}")
    
    # 标签分布统计
    if rows:
        original_labels = [r['labels'] for r in rows]
        label_counter = Counter(original_labels)
        print(f"多分类标签分布: {dict(label_counter)}")
        
        # 只对训练集进行过采样和增强
        if split == 'train' and augment_train:
            rows = oversample_training_data(rows, augment_minority=True)
            new_label_counter = Counter([r['labels'] for r in rows])
            print(f"过采样后多分类分布: {dict(new_label_counter)}")
        
        # 数据形状信息
        sample_shape = rows[0]['data'].shape
        print(f"数据形状: (时间点, 通道数) = {sample_shape}")
    
    return rows

def process_split(split_name, rows, out_dir, feature_path, region_map, global_edges, node_extractor,
                  k_intra, k_global, w1, w2, fs, device, batch_size, save_npy):
    out = {}
    adjs_dir = os.path.join(feature_path, 'adjs_precomputed1')
    safe_makedirs(adjs_dir)
    safe_makedirs(out_dir)

    print(f"Processing {split_name} data on device {device}...")

    progress_bar = tqdm(total=len(rows), desc=f"{split_name} split on {device}")

    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]

        max_T = max(entry['data'].shape[0] for entry in batch)
        C = batch[0]['data'].shape[1]

        # padding 对齐时间长度
        padded_ts = torch.zeros(len(batch), max_T, C)
        for j, entry in enumerate(batch):
            T = entry['data'].shape[0]
            padded_ts[j, :T, :] = torch.from_numpy(entry['data'])

        padded_ts = padded_ts.to(device)

        with torch.no_grad():
            feats_batch = node_extractor(padded_ts)  # [B, C, d]

        feats_batch_np = feats_batch.cpu().numpy()

        for j, entry in enumerate(batch):
            sid = entry['id']
            ts_np = entry['data']
            label = entry['labels']

            node_feats_np = feats_batch_np[j]

            G_intra, _, _ = build_intra_region_view_mi(
                ts_np, region_map, global_edges,
                n_bins=16, strategy="uniform",
                window_size=400, stride=200
            )

            S_global, _, _ = build_global_view(ts_np, w1=w1, w2=w2, fs=fs)

            A_intra_sp = topk_sparsify_sym_row_normalize(G_intra, k_intra)
            A_global_sp = topk_sparsify_sym_row_normalize(S_global, k_global)

            
            if save_npy:
                np.save(os.path.join(adjs_dir, f"{sid}_intra.npy"), A_intra_sp.astype(np.float32))
                np.save(os.path.join(adjs_dir, f"{sid}_global.npy"), A_global_sp.astype(np.float32))
                np.save(os.path.join(adjs_dir, f"{sid}_nodefeat.npy"), node_feats_np.astype(np.float32))

            out[sid] = {
                'id': sid,
                'label': label,
                'node_feats': node_feats_np.astype(np.float32),
                'A_intra': A_intra_sp.astype(np.float32),
                'A_global': A_global_sp.astype(np.float32),
            }
        progress_bar.update(len(batch))

    progress_bar.close()
    return out


def run_on_gpu(rank, args, all_rows, splits, region_map, global_edges):
    device = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'
    if device != 'cpu':
        torch.cuda.set_device(rank)
    print(f"Process {rank} is using {device}")

    if args.model_type == "cnn":
        conv = ConvOnlyFeatureExtractor(
            out_size=args.node_out_size,
            conv_channels=[32, 32, 16],
            kernel_sizes=[8, 8, 8]
        )
    elif args.model_type == "cnn_lstm":
        conv = ConvKRegionCNNLSTM(
            k=1,
            out_size=args.node_out_size,
            time_series=3400,
            conv_channels=[32, 32, 16],
            kernel_sizes=[8, 8, 8],
            pool_size=16,
            lstm_hidden=32,
            lstm_layers=1,
            bidirectional=True
        )
    else:
        raise ValueError(f"Unknown model_type {args.model_type}")
    node_extractor = NodeFeatureExtractor(conv, out_size=args.node_out_size).to(device).eval()

    results = {}
    for split in splits:
        rows = all_rows[split]
        start = len(rows) // args.nprocs * rank
        end = len(rows) // args.nprocs * (rank + 1)
        if rank == args.nprocs - 1:
            end = len(rows)
        part = rows[start:end]
        if not part:
            continue

        out_dict = process_split(split, part, args.out_dir, args.feature_path, region_map, global_edges,
                                 node_extractor, k_intra=args.k_intra, k_global=args.k_global,
                                 w1=args.w1, w2=args.w2, fs=args.fs, device=device,
                                 batch_size=args.batch_size, save_npy=args.save_npy)
        results[split] = out_dict

    out_file = os.path.join(args.out_dir, f"processed_rank{rank}.pkl")
    with open(out_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"Rank {rank} saved {out_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_path', type=str, required=True)
    parser.add_argument('--feature_type',type=str,default='oxy',choices=['oxy','dxy','both'])
    parser.add_argument('--out_dir', type=str, default='processed_fnirs')
    parser.add_argument('--k_intra', type=int, default=8)
    parser.add_argument('--k_global', type=int, default=8)
    parser.add_argument('--w1', type=float, default=0.5)
    parser.add_argument('--w2', type=float, default=0.5)
    parser.add_argument('--fs', type=float, default=1.0)
    parser.add_argument('--model_type', type=str, default='cnn', choices=['cnn', 'cnn_lstm'])
    parser.add_argument('--node_out_size', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--nprocs', type=int, default=torch.cuda.device_count())
    parser.add_argument('--save_npy', type=bool, default=True, help='Save individual .npy files for each sample')
    parser.add_argument('--global_edges_sample_size', type=int, default=200,
                        help='Number of samples to use for global edges computation')
    parser.add_argument('--augment_train', type=bool, default=True)
    args = parser.parse_args()

    splits = ['train', 'val', 'test']
    all_rows = {}
    for s in splits:
        augment=(s=='train')and args.augment_train
        all_rows[s] = prepare_and_balance_data(args.feature_path, s, args.feature_type, augment_train=augment)
        print(f"{s}: {len(all_rows[s])} samples | label dist: {Counter([r['labels'] for r in all_rows[s]])}")

    region_map_path = os.path.join(args.feature_path, 'region_map1.npy')
    if os.path.exists(region_map_path):
        region_map = np.load(region_map_path)
        print("Loaded region_map from", region_map_path)
    else:
        C = all_rows['train'][0]['data'].shape[1]
        region_map = make_region_map(brain_regions, C)
        np.save(region_map_path, region_map)
        print("Saved region_map to", region_map_path)

    global_edges_path = os.path.join(args.feature_path, 'global_edges1.npy')
    if os.path.exists(global_edges_path):
        global_edges = np.load(global_edges_path, allow_pickle=True)
        print("Loaded global_edges from", global_edges_path)
    else:
        # 使用抽样来计算 global_edges
        train_list_full = [r['data'] for r in all_rows['train']]
        sample_size = min(len(train_list_full), args.global_edges_sample_size)

        indices = np.random.choice(len(train_list_full), size=sample_size, replace=False)
        train_list_sample = [train_list_full[i] for i in indices]

        global_edges = compute_global_edges(train_list_sample, n_bins=16, strategy="uniform")
        np.save(global_edges_path, global_edges, allow_pickle=True)
        print(f"Saved global_edges to {global_edges_path} using {sample_size} samples")

    safe_makedirs(args.out_dir)

    if args.nprocs > 1 and torch.cuda.is_available():
        mp.spawn(run_on_gpu, nprocs=args.nprocs, args=(args, all_rows, splits, region_map, global_edges), join=True)
    else:
        run_on_gpu(0, args, all_rows, splits, region_map, global_edges)

    merged_results = {s: {} for s in splits}
    merged_files = []
    for fname in os.listdir(args.out_dir):
        if fname.startswith("processed_rank"):
            fpath = os.path.join(args.out_dir, fname)
            try:
                with open(fpath, 'rb') as f:
                    part_dict = pickle.load(f)
                for s in splits:
                    if s in part_dict:
                        merged_results[s].update(part_dict[s])
                merged_files.append(fpath)
            except Exception as e:
                print(f"Warning: Could not read {fpath}. Error: {e}")

    for s in splits:
        if merged_results[s]:
            final_file = os.path.join(args.out_dir, f"processed_{s}.pkl")
            with open(final_file, 'wb') as f:
                pickle.dump(merged_results[s], f)
            print(f"Successfully merged {len(merged_results[s])} samples for {s} split into {final_file}")
        else:
            print(f"Warning: No data to merge for {s} split.")

    for fpath in merged_files:
        os.remove(fpath)
    print("Cleaned up temporary files.")

    if args.save_npy:
        print("Finished. per-sample .npy saved under:", os.path.join(args.feature_path, 'adjs_precomputed'))
    else:
        print("Finished. Per-sample .npy files were not saved.")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()