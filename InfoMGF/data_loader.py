import os
import torch
from torch_geometric.data import Data, InMemoryDataset
import pickle
import numpy as np
from torch_geometric.utils import dense_to_sparse
'''
train_dataset = BrainGraphDataset(root='path/to/your/data', split='train')
val_dataset = BrainGraphDataset(root='path/to/your/data', split='val')
test_dataset = BrainGraphDataset(root='path/to/your/data', split='test')

print(f"Train dataset size: {len(train_dataset)}")
print(f"Val dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

# 创建DataLoader
from torch_geometric.loader import DataLoader

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
'''
class BrainGraphDataset(InMemoryDataset):
    def __init__(self, root, split='train', transform=None, pre_transform=None):
        self.split = split
        self.raw_file_path = os.path.join(root, f'processed_{split}.pkl')
        self.processed_file_path = os.path.join(root, f'processed_brain_graph_{split}.pt')
        
        super().__init__(root, transform, pre_transform)

        if os.path.exists(self.processed_file_path):
            self.data, self.slices = torch.load(self.processed_file_path)
        else:
            self.process()
            self.data, self.slices = torch.load(self.processed_file_path)
    

    @property
    def raw_file_names(self):
        return []  
    
    @property
    def processed_file_names(self):
        return []  
    
    @property
    def raw_paths(self):
        return [self.raw_file_path]
    
    @property
    def processed_paths(self):
        return [self.processed_file_path]
    
    def process(self):
        """处理数据并保存到自定义路径"""
        print(f"Loading data from: {self.raw_file_path}")
        
        with open(self.raw_file_path, 'rb') as f:
            data_dict = pickle.load(f)

        data_list = []
        for sid, graph_data in data_dict.items():
            node_features = torch.from_numpy(graph_data['node_feats']).float()
            label = torch.tensor(graph_data['label'], dtype=torch.long)
            A_intra = graph_data['A_intra']
            A_global = graph_data['A_global']

            edge_index_intra = dense_to_sparse(torch.from_numpy(A_intra).float())[0]
            edge_index_global = dense_to_sparse(torch.from_numpy(A_global).float())[0]

            data = Data(
                x=node_features,
                y=label,
                edge_index_intra=edge_index_intra,
                edge_index_global=edge_index_global,
                sid=sid
            )
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_file_path)
        print(f"Saved processed data to: {self.processed_file_path}")


