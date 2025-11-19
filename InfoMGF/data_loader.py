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
    def __init__(self, root, split='train', label_mode='binary', smart_resample=True, 
                 balance_strategy='downsample_class1', use_original_only=False, 
                 transform=None, pre_transform=None):
        self.split = split
        self.label_mode = label_mode  
        self.smart_resample = smart_resample
        self.balance_strategy = balance_strategy
        self.use_original_only = use_original_only
        self.raw_file_path = os.path.join(root, f'processed_{split}.pkl')
        cache_suffix = f'{label_mode}'
        if smart_resample:
            cache_suffix += '_smartresample'
        if use_original_only:
            cache_suffix += '_original'
        self.processed_file_path = os.path.join(root, f'processed_brain_graph_{split}_{cache_suffix}.pt')
        
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

        # 首先构建原始数据列表，保留原始标签信息
        data_list_with_original_labels = []
        for sid, graph_data in data_dict.items():
            node_features = torch.from_numpy(graph_data['node_feats']).float()
            raw_label = int(graph_data['label'])
            
            A_intra = graph_data['A_intra']
            A_global = graph_data['A_global']
            edge_index_intra = dense_to_sparse(torch.from_numpy(A_intra).float())[0]
            edge_index_global = dense_to_sparse(torch.from_numpy(A_global).float())[0]

            data = Data(
                x=node_features,
                y=torch.tensor(raw_label, dtype=torch.long),  # 先保留原始标签
                edge_index_intra=edge_index_intra,
                edge_index_global=edge_index_global,
                sid=sid,
                original_label=raw_label  # 额外保存原始标签
            )
            data_list_with_original_labels.append(data)
        
        # 如果需要，过滤掉增强样本（只使用原始样本）
        if self.use_original_only and self.split == 'train':
            original_count = len(data_list_with_original_labels)
            data_list_with_original_labels = [
                data for data in data_list_with_original_labels 
                if '_aug' not in str(data.sid)
            ]
            filtered_count = len(data_list_with_original_labels)
            print(f"Filtered augmented samples: {original_count} → {filtered_count} (removed {original_count - filtered_count} samples)")
        
        # 如果是binary模式且需要智能重采样
        if self.label_mode == 'binary' and self.smart_resample and self.split == 'train':
            print(f"\nApplying smart binary resampling strategy: {self.balance_strategy}")
            data_list = self._smart_binary_resample(data_list_with_original_labels)
        else:
            # 常规处理：直接转换标签
            data_list = []
            for data in data_list_with_original_labels:
                if self.label_mode == 'binary':
                    mapped = 0 if data.original_label == 0 else 1
                    data.y = torch.tensor(mapped, dtype=torch.long)
                data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_file_path)
        print(f"Saved processed data to: {self.processed_file_path}")
    
    def _smart_binary_resample(self, data_list_with_original_labels):
        """
        智能二分类重采样策略
        
        核心思想:
        1. 分离 Class 0 和非0类
        2. 对每个非0类先过采样到 Class 0 的数量
        3. 按原始比例从过采样的非0类中抽取样本
        4. 最终 Class 0 和 Class 1 的数量平衡
        """
        # Step 1: 按原始标签分组
        class_groups = {}
        for data in data_list_with_original_labels:
            label = data.original_label
            if label not in class_groups:
                class_groups[label] = []
            class_groups[label].append(data)
        
        print(f"Original class distribution:")
        for label, samples in sorted(class_groups.items()):
            print(f"   Class {label}: {len(samples)} samples")
        
        # Step 2: 分离 Class 0 和其他类
        class_0_samples = class_groups.get(0, [])
        n_class_0 = len(class_0_samples)
        
        other_classes = {k: v for k, v in class_groups.items() if k != 0}
        
        if not other_classes:
            print(" No other classes found, returning original data")
            # 直接转换为二分类标签
            for data in data_list_with_original_labels:
                data.y = torch.tensor(0 if data.original_label == 0 else 1, dtype=torch.long)
            return data_list_with_original_labels
        
        # Step 3: 计算其他类的原始比例
        original_counts = {k: len(v) for k, v in other_classes.items()}
        total_other = sum(original_counts.values())
        proportions = {k: count / total_other for k, count in original_counts.items()}
        
        print(f"\nOther classes proportions:")
        for k, prop in sorted(proportions.items()):
            print(f"   Class {k}: {prop:.2%} ({original_counts[k]} samples)")
        
         # Step 4: 对每个非0类，分离原始数据和增强数据
        separated_others = {}
        for class_label, samples in other_classes.items():
            original_samples = [s for s in samples if '_aug' not in str(s.sid)]
            augmented_samples = [s for s in samples if '_aug' in str(s.sid)]
            separated_others[class_label] = {
                'original': original_samples,
                'augmented': augmented_samples,
                'all': samples
            }
            print(f"   Class {class_label}: {len(original_samples)} original + {len(augmented_samples)} augmented")
            
        # Step 5: 根据策略决定最终的平衡方式
       
        target_count = n_class_0
        balanced_class_0 = class_0_samples
        
        # 首先收集所有原始样本
        balanced_class_1 = []
        original_samples_by_class = {}
        
        for class_label, class_info in separated_others.items():
            original_samples = class_info['original']
            original_samples_by_class[class_label] = original_samples
            balanced_class_1.extend(original_samples)
            print(f"   Class {class_label}: included all {len(original_samples)} original samples")
        
        # 计算还需要补充的增强样本数量
        current_class1_count = len(balanced_class_1)
        remaining_needed = target_count - current_class1_count
        
        if remaining_needed > 0:
            print(f"   Need {remaining_needed} additional augmented samples")
            
            # 按比例分配需要补充的增强样本数量
            augmented_samples_to_add = {}
            for class_label, proportion in sorted(proportions.items()):
                n_needed_from_class = int(remaining_needed * proportion)
                available_augmented = separated_others[class_label]['augmented']
                
                # 不能超过可用的增强样本数量
                n_to_take = min(n_needed_from_class, len(available_augmented))
                augmented_samples_to_add[class_label] = n_to_take
                
                print(f"   Class {class_label}: need {n_needed_from_class}, available {len(available_augmented)}, will take {n_to_take}")
            
            # 如果因为四舍五入导致总数不够，按比例补充
            total_allocated = sum(augmented_samples_to_add.values())
            if total_allocated < remaining_needed:
                deficit = remaining_needed - total_allocated
                print(f"   Allocation deficit: {deficit}, redistributing...")
                
                # 按比例重新分配差额
                for class_label, proportion in sorted(proportions.items()):
                    if deficit <= 0:
                        break
                    additional = min(
                        int(deficit * proportion),
                        len(separated_others[class_label]['augmented']) - augmented_samples_to_add[class_label]
                    )
                    augmented_samples_to_add[class_label] += additional
                    deficit -= additional
            
            # 实际抽取增强样本
            for class_label, n_to_take in augmented_samples_to_add.items():
                if n_to_take > 0:
                    available_augmented = separated_others[class_label]['augmented']
                    if len(available_augmented) >= n_to_take:
                        indices = np.random.choice(len(available_augmented), size=n_to_take, replace=False)
                        selected = [available_augmented[i] for i in indices]
                    else:
                        # 如果增强样本不够，重复使用
                        indices = np.random.choice(len(available_augmented), size=n_to_take, replace=True)
                        selected = [available_augmented[i] for i in indices]
                    
                    balanced_class_1.extend(selected)
                    print(f"   Added {len(selected)} augmented samples from Class {class_label}")
        
        elif remaining_needed < 0:
            # 如果原始样本已经超过目标数量，需要下采样
            print(f"Original samples exceed target, need to downsample {abs(remaining_needed)} samples")
            # 这里可以按类别比例下采样，但为了简单起见，我们随机下采样
            indices = np.random.choice(len(balanced_class_1), size=target_count, replace=False)
            balanced_class_1 = [balanced_class_1[i] for i in indices]
        
        print(f"\nStrategy: Downsample Class 1 (proportionally, preserve originals)")
        print(f"   Class 0: {len(balanced_class_0)} (unchanged)")
        print(f"   Class 1: {total_other} → {len(balanced_class_1)}")
        print(f"   - Original samples in Class 1: {current_class1_count}")
        print(f"   - Augmented samples in Class 1: {len(balanced_class_1) - current_class1_count}")
        final_data_list = []
        for data in balanced_class_0:
            data.y = torch.tensor(0, dtype=torch.long)
            final_data_list.append(data)
        
        for data in balanced_class_1:
            data.y = torch.tensor(1, dtype=torch.long)
            final_data_list.append(data)
        
        # 打乱顺序
        np.random.shuffle(final_data_list)
        
        print(f"\nFinal balanced dataset:")
        print(f"   Total samples: {len(final_data_list)}")
        print(f"   Class 0: {len(balanced_class_0)} ({len(balanced_class_0)/len(final_data_list)*100:.1f}%)")
        print(f"   Class 1: {len(balanced_class_1)} ({len(balanced_class_1)/len(final_data_list)*100:.1f}%)")
        
        return final_data_list
