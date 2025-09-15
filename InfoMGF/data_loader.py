import os
import pickle
from typing import Optional, List, Dict, Any
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


class ProcessedRestingDataset(Dataset):
    def __init__(self,
                 processed_dir: str,
                 split: str = 'train',
                 feature_path: Optional[str] = None,
                 use_npy: bool = False,
                 preload: bool = False,
                 to_binary: bool = False,
                 transform=None):
        assert split in ('train', 'val', 'test')
        self.processed_dir = processed_dir
        self.split = split
        self.feature_path = feature_path
        self.use_npy = use_npy
        self.preload = preload
        self.to_binary = to_binary
        self.transform = transform

        pkl_path = os.path.join(processed_dir, f'processed_{split}.pkl')
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"processed file not found: {pkl_path}")

        with open(pkl_path, 'rb') as f:
            data_map = pickle.load(f)

        # data_map is expected to be dict: id -> sample_dict
        # convert to list for indexing
        self.ids = list(data_map.keys())
        self._orig_map = data_map

        # optional preload
        self._items: List[Dict[str, Any]] = []
        if preload:
            for sid in self.ids:
                item = self._load_item_by_id(sid)
                self._items.append(item)

    def __len__(self):
        return len(self.ids)

    def _load_item_by_id(self, sid: str) -> Dict[str, Any]:
        # prefer pickle entry
        entry = self._orig_map[sid]

        if self.use_npy and self.feature_path is not None:
            # try load per-sample npy from feature_path/adjs_precomputed
            npy_dir = os.path.join(self.feature_path, 'adjs_precomputed')
            node_path = os.path.join(npy_dir, f'{sid}_nodefeat.npy')
            intra_path = os.path.join(npy_dir, f'{sid}_intra.npy')
            global_path = os.path.join(npy_dir, f'{sid}_global.npy')
            if os.path.exists(node_path) and os.path.exists(intra_path) and os.path.exists(global_path):
                node_feats = np.load(node_path)
                A_intra = np.load(intra_path)
                A_global = np.load(global_path)
            else:
                # fallback to pickle
                node_feats = entry.get('node_feats')
                A_intra = entry.get('A_intra')
                A_global = entry.get('A_global')
        else:
            node_feats = entry.get('node_feats')
            A_intra = entry.get('A_intra')
            A_global = entry.get('A_global')

        if node_feats is None or A_intra is None or A_global is None:
            raise ValueError(f"Missing required fields for sample {sid}. Ensure pre.py saved node_feats and adjs.")

        label = int(entry.get('label'))
        if self.to_binary:
            label = 1 if label > 0 else 0

        sample = {
            'id': sid,
            'X': torch.from_numpy(np.asarray(node_feats)).float(),    # shape (C, d)
            'A_intra': torch.from_numpy(np.asarray(A_intra)).float(), # shape (C, C)
            'A_global': torch.from_numpy(np.asarray(A_global)).float(),
            'label': torch.tensor(label, dtype=torch.long)
        }
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.preload:
            return self._items[idx]
        sid = self.ids[idx]
        return self._load_item_by_id(sid)


def collate_processed_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    # batch is list of samples returned by __getitem__
    ids = [b['id'] for b in batch]
    Xs = [b['X'] for b in batch]
    A_intras = [b['A_intra'] for b in batch]
    A_globals = [b['A_global'] for b in batch]
    labels = torch.stack([b['label'] for b in batch], dim=0)

    # ensure same shapes
    # X: list of (C,d) -> stack to (B,C,d)
    X = torch.stack(Xs, dim=0)
    A_intra = torch.stack(A_intras, dim=0)
    A_global = torch.stack(A_globals, dim=0)

    return {'id': ids, 'X': X, 'A_intra': A_intra, 'A_global': A_global, 'labels': labels}


# Helper to compute class weights for WeightedRandomSampler
def compute_class_weights(dataset: ProcessedRestingDataset):
    labels = []
    for i in range(len(dataset)):
        l = dataset[i]['label'].item() if dataset.preload else int(dataset._orig_map[dataset.ids[i]]['label'])
        labels.append(l)
    classes, counts = np.unique(labels, return_counts=True)
    class_weights = {c: float(len(labels)) / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    sample_weights = [class_weights[l] for l in labels]
    return sample_weights, class_weights


class RestingDataModule:
    def __init__(self, processed_dir: str, feature_path: Optional[str] = None,
                 batch_size: int = 16, num_workers: int = 4, to_binary: bool = False,
                 use_npy: bool = False, preload: bool = False):
        self.processed_dir = processed_dir
        self.feature_path = feature_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.to_binary = to_binary
        self.use_npy = use_npy
        self.preload = preload

        self.trainset = None
        self.valset = None
        self.testset = None

    def setup(self):
        self.trainset = ProcessedRestingDataset(self.processed_dir, 'train', feature_path=self.feature_path,
                                                use_npy=self.use_npy, preload=self.preload, to_binary=self.to_binary)
        self.valset = ProcessedRestingDataset(self.processed_dir, 'val', feature_path=self.feature_path,
                                              use_npy=self.use_npy, preload=self.preload, to_binary=self.to_binary)
        self.testset = ProcessedRestingDataset(self.processed_dir, 'test', feature_path=self.feature_path,
                                               use_npy=self.use_npy, preload=self.preload, to_binary=self.to_binary)

    def train_dataloader(self, weighted_sampler: bool = False):
        if weighted_sampler:
            sample_weights, _ = compute_class_weights(self.trainset)
            sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
            return DataLoader(self.trainset, batch_size=self.batch_size, num_workers=self.num_workers,
                              collate_fn=collate_processed_batch, sampler=sampler)
        return DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                          collate_fn=collate_processed_batch)

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          collate_fn=collate_processed_batch)

    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          collate_fn=collate_processed_batch)



