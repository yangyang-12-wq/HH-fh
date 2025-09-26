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
from utils_graph_build import (
    make_region_map,
    compute_global_edges,
    build_intra_region_view_mi,
    build_global_view,
    brain_regions
)
from node_extractor import *

def process_split(split_name, rows, out_dir, feature_path, region_map, global_edges, node_extractor,
                  k_intra, k_global, w1, w2, fs, device, batch_size, save_npy):
    out = {}
    adjs_dir = os.path.join(feature_path, 'adjs_precomputed')
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

            # build intra/global (CPU-bound)
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
    args = parser.parse_args()

    splits = ['train', 'val', 'test']
    all_rows = {}
    for s in splits:
        feature_data = load_feature_pickle(args.feature_path, s)
        rows = prepare_rows_from_feature_data(feature_data, args.feature_type)
        all_rows[s] = rows
        print(f"{s}: {len(rows)} samples | label dist: {Counter([r['labels'] for r in rows])}")

    region_map_path = os.path.join(args.feature_path, 'region_map.npy')
    if os.path.exists(region_map_path):
        region_map = np.load(region_map_path)
        print("Loaded region_map from", region_map_path)
    else:
        C = all_rows['train'][0]['data'].shape[1]
        region_map = make_region_map(brain_regions, C)
        np.save(region_map_path, region_map)
        print("Saved region_map to", region_map_path)

    global_edges_path = os.path.join(args.feature_path, 'global_edges.npy')
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