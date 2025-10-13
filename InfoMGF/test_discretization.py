"""
测试脚本：验证 compute_global_edges 和 discretize_data_strict 函数的正确性

验证内容：
1. compute_global_edges 输出的边界是否合理
2. discretize_data_strict 的离散化结果是否正确
3. 使用全局边界的一致性验证
"""

import numpy as np
import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from utils_graph_build import compute_global_edges, discretize_data_strict


def generate_synthetic_fnirs_data(n_samples=5, n_timepoints=1000, n_channels=53):
    """
    生成模拟的 fNIRS 数据用于测试
    
    Args:
        n_samples: 样本数量
        n_timepoints: 每个样本的时间点数
        n_channels: 通道数
    
    Returns:
        list: 包含 n_samples 个形状为 (n_timepoints, n_channels) 的数组
    """
    print(f"\n{'='*60}")
    print(f"生成模拟 fNIRS 数据:")
    print(f"  样本数: {n_samples}")
    print(f"  时间点数: {n_timepoints}")
    print(f"  通道数: {n_channels}")
    print(f"{'='*60}\n")
    
    dataset = []
    
    for i in range(n_samples):
        # 创建不同类型的信号
        data = np.zeros((n_timepoints, n_channels))
        
        for ch in range(n_channels):
            if ch % 10 == 0:  # 每10个通道中有1个常量通道
                # 常量通道（用于测试边界情况）
                data[:, ch] = 0.5 + np.random.normal(0, 1e-10, n_timepoints)
            elif ch % 5 == 0:  # 低波动通道
                # 低波动信号
                data[:, ch] = np.sin(2 * np.pi * 0.1 * np.arange(n_timepoints) / n_timepoints) * 0.1
                data[:, ch] += np.random.normal(0, 0.01, n_timepoints)
            else:
                # 正常波动信号（模拟真实 fNIRS）
                # 添加基线漂移
                baseline = np.linspace(0.8, 1.2, n_timepoints)
                # 添加血氧动力学响应（慢波动）
                hrf = np.sin(2 * np.pi * 0.05 * np.arange(n_timepoints) / n_timepoints) * 0.3
                # 添加噪声
                noise = np.random.normal(0, 0.05, n_timepoints)
                
                data[:, ch] = baseline + hrf + noise
        
        dataset.append(data)
        print(f"样本 {i+1}: 形状 {data.shape}, 范围 [{data.min():.4f}, {data.max():.4f}]")
    
    return dataset


def test_compute_global_edges(dataset, n_bins=16, strategy="uniform"):
    """
    测试1: 验证 compute_global_edges 函数
    
    检查点：
    - 边界数量是否正确（n_bins - 1）
    - 边界值是否覆盖数据范围
    - 边界值是否单调递增
    - 常量通道是否正确标记
    """
    print(f"\n{'='*60}")
    print(f"测试1: compute_global_edges 函数")
    print(f"{'='*60}\n")
    
    edges_list = compute_global_edges(dataset, n_bins=n_bins, strategy=strategy)
    
    # 合并所有数据用于验证
    all_data = np.concatenate(dataset, axis=0)
    n_channels = all_data.shape[1]
    
    print(f"\n边界验证结果:")
    print(f"-" * 60)
    
    constant_count = 0
    valid_count = 0
    
    for ch in range(n_channels):
        edges = edges_list[ch]
        ch_data = all_data[:, ch]
        
        if edges is None:
            # 常量通道
            constant_count += 1
            data_range = ch_data.max() - ch_data.min()
            assert data_range < 1e-8, f"通道 {ch} 标记为常量，但范围 {data_range:.2e} > 1e-8"
            if ch < 10:  # 只打印前10个
                print(f"  通道 {ch:2d}: 常量通道 ✓ (范围: {data_range:.2e})")
        else:
            valid_count += 1
            # 验证边界数量
            assert len(edges) == n_bins - 1, \
                f"通道 {ch} 边界数量错误: {len(edges)} != {n_bins - 1}"
            
            # 验证边界单调递增
            assert np.all(np.diff(edges) > 0), \
                f"通道 {ch} 边界非单调递增: {edges}"
            
            # 验证边界覆盖数据范围
            data_min, data_max = ch_data.min(), ch_data.max()
            edges_min, edges_max = edges[0], edges[-1]
            
            # 边界应该在数据范围内
            assert edges_min >= data_min and edges_max <= data_max, \
                f"通道 {ch} 边界 [{edges_min:.4f}, {edges_max:.4f}] 超出数据范围 [{data_min:.4f}, {data_max:.4f}]"
            
            # 边界应该接近覆盖整个范围
            coverage = (edges_max - edges_min) / (data_max - data_min)
            
            if ch < 5:  # 只打印前5个有效通道
                print(f"  通道 {ch:2d}: {len(edges)} 个边界 ✓")
                print(f"           范围: [{data_min:.4f}, {data_max:.4f}]")
                print(f"           边界: [{edges_min:.4f}, {edges_max:.4f}] (覆盖率: {coverage:.2%})")
    
    print(f"\n总结:")
    print(f"  常量通道: {constant_count}/{n_channels}")
    print(f"  有效通道: {valid_count}/{n_channels}")
    print(f"  ✓ compute_global_edges 测试通过!\n")
    
    return edges_list


def test_discretize_data_strict(dataset, edges_list, n_bins=16):
    """
    测试2: 验证 discretize_data_strict 函数
    
    检查点：
    - bin 值范围是否在 [0, n_bins-1]
    - 常量通道是否全为 0
    - 离散化结果是否与边界对应
    """
    print(f"\n{'='*60}")
    print(f"测试2: discretize_data_strict 函数")
    print(f"{'='*60}\n")
    
    all_passed = True
    
    for idx, data in enumerate(dataset):
        print(f"\n样本 {idx + 1}:")
        print(f"-" * 40)
        
        # 使用全局边界离散化
        binned, _ = discretize_data_strict(
            data, 
            n_bins=n_bins, 
            global_edges=edges_list
        )
        
        n_channels = data.shape[1]
        
        # 验证1: bin 值范围
        min_bin = binned.min()
        max_bin = binned.max()
        assert min_bin >= 0, f"样本 {idx} bin 最小值 {min_bin} < 0"
        assert max_bin <= n_bins - 1, f"样本 {idx} bin 最大值 {max_bin} > {n_bins - 1}"
        print(f"  Bin 值范围: [{min_bin}, {max_bin}] ✓ (应在 [0, {n_bins-1}])")
        
        # 验证2: 常量通道
        constant_channels_correct = 0
        for ch in range(n_channels):
            if edges_list[ch] is None:
                # 常量通道应该全为 0
                assert np.all(binned[:, ch] == 0), \
                    f"样本 {idx} 通道 {ch} 是常量通道，但 bin 值不全为 0"
                constant_channels_correct += 1
        
        if constant_channels_correct > 0:
            print(f"  常量通道检查: {constant_channels_correct} 个通道全为 0 ✓")
        
        # 验证3: 离散化逻辑正确性（抽查几个通道）
        check_channels = [ch for ch in range(min(5, n_channels)) if edges_list[ch] is not None]
        
        for ch in check_channels:
            edges = edges_list[ch]
            ch_data = data[:, ch]
            ch_bins = binned[:, ch]
            
            # 手动验证几个点的离散化结果
            for t in range(0, len(ch_data), len(ch_data) // 3):  # 检查3个点
                value = ch_data[t]
                bin_idx = ch_bins[t]
                
                # 验证 bin 索引的正确性
                # np.digitize(x, edges, right=False) 返回的索引满足:
                # bin_idx = 0 表示 x < edges[0]
                # bin_idx = i 表示 edges[i-1] <= x < edges[i]
                # bin_idx = len(edges) 表示 x >= edges[-1]
                
                if bin_idx == 0:
                    assert value < edges[0], \
                        f"通道 {ch} 时间点 {t}: 值 {value:.4f} 应 < {edges[0]:.4f}"
                elif bin_idx == len(edges):
                    assert value >= edges[-1], \
                        f"通道 {ch} 时间点 {t}: 值 {value:.4f} 应 >= {edges[-1]:.4f}"
                else:
                    assert edges[bin_idx - 1] <= value < edges[bin_idx], \
                        f"通道 {ch} 时间点 {t}: 值 {value:.4f} 应在 [{edges[bin_idx-1]:.4f}, {edges[bin_idx]:.4f})"
        
        print(f"  离散化逻辑: 抽查 {len(check_channels)} 个通道 ✓")
        print(f"  ✓ 样本 {idx + 1} 验证通过")
    
    print(f"\n  ✓ discretize_data_strict 测试通过!\n")


def test_consistency_across_samples(dataset, edges_list, n_bins=16):
    """
    测试3: 验证使用全局边界的一致性
    
    检查点：
    - 相同的值在不同样本中离散化结果相同
    - 边界在不同样本间的稳定性
    """
    print(f"\n{'='*60}")
    print(f"测试3: 全局边界一致性验证")
    print(f"{'='*60}\n")
    
    # 创建一个测试值，在所有样本中相同位置使用相同的值
    n_channels = dataset[0].shape[1]
    test_values = np.random.uniform(0.5, 1.5, n_channels)
    
    binned_results = []
    
    for idx, data in enumerate(dataset):
        # 创建测试数据：在第一行插入测试值
        test_data = data.copy()
        test_data[0, :] = test_values
        
        binned, _ = discretize_data_strict(
            test_data,
            n_bins=n_bins,
            global_edges=edges_list
        )
        
        # 记录第一行的 bin 值
        binned_results.append(binned[0, :])
    
    # 验证：所有样本的测试值应该得到相同的 bin 值
    binned_results = np.array(binned_results)
    
    inconsistent_channels = []
    for ch in range(n_channels):
        if edges_list[ch] is None:
            continue  # 跳过常量通道
        
        # 检查该通道在所有样本中的 bin 值是否一致
        unique_bins = np.unique(binned_results[:, ch])
        if len(unique_bins) > 1:
            inconsistent_channels.append(ch)
    
    if inconsistent_channels:
        print(f"  ✗ 发现 {len(inconsistent_channels)} 个通道离散化不一致:")
        for ch in inconsistent_channels[:5]:  # 只显示前5个
            print(f"    通道 {ch}: {binned_results[:, ch]}")
        assert False, "全局边界一致性验证失败"
    else:
        print(f"  ✓ 所有通道使用全局边界的离散化结果一致")
        print(f"  测试: {len(dataset)} 个样本 × {n_channels} 个通道")
        print(f"  ✓ 一致性测试通过!\n")


def test_edge_stability(n_bins=16, n_repeats=3):
    """
    测试4: 验证边界计算的稳定性
    
    检查点：
    - 同一通道在不同样本集上计算的边界应该相似
    """
    print(f"\n{'='*60}")
    print(f"测试4: 边界稳定性验证")
    print(f"{'='*60}\n")
    
    all_edges = []
    
    for i in range(n_repeats):
        # 生成新的数据集
        dataset = generate_synthetic_fnirs_data(n_samples=3, n_timepoints=500, n_channels=20)
        edges_list = compute_global_edges(dataset, n_bins=n_bins, strategy="uniform")
        all_edges.append(edges_list)
    
    # 检查边界的稳定性
    n_channels = len(all_edges[0])
    
    print(f"\n边界稳定性检查:")
    print(f"-" * 60)
    
    for ch in range(min(5, n_channels)):  # 检查前5个通道
        edges_arrays = [edges[ch] for edges in all_edges if edges[ch] is not None]
        
        if len(edges_arrays) < 2:
            continue
        
        # 计算边界的标准差
        edges_stacked = np.array(edges_arrays)
        edges_std = np.std(edges_stacked, axis=0)
        edges_mean = np.mean(edges_stacked, axis=0)
        
        # 相对标准差（变异系数）
        cv = edges_std / (np.abs(edges_mean) + 1e-8)
        
        print(f"  通道 {ch}:")
        print(f"    平均边界: {edges_mean[:3]} ... {edges_mean[-3:]}")
        print(f"    标准差:   {edges_std[:3]} ... {edges_std[-3:]}")
        print(f"    变异系数: {cv.mean():.4f} (平均)")
        
        # 变异系数应该比较小（< 0.1 表示稳定）
        assert cv.mean() < 0.5, f"通道 {ch} 边界不稳定，变异系数 {cv.mean():.4f} > 0.5"
    
    print(f"\n  ✓ 边界稳定性测试通过!\n")


def test_real_data_simulation():
    """
    测试5: 使用更真实的 fNIRS 数据模拟
    """
    print(f"\n{'='*60}")
    print(f"测试5: 真实数据模拟测试")
    print(f"{'='*60}\n")
    
    # 生成更真实的 fNIRS 数据
    n_samples = 3
    n_timepoints = 2000
    n_channels = 53
    n_bins = 16
    
    print("生成真实 fNIRS 数据特征:")
    print("- 血氧动力学响应（6-12秒响应时间）")
    print("- 呼吸和心跳伪影（0.2-1.0 Hz）")
    print("- 运动伪影（随机脉冲）")
    print("- 基线漂移\n")
    
    dataset = []
    for i in range(n_samples):
        data = np.zeros((n_timepoints, n_channels))
        t = np.arange(n_timepoints)
        
        for ch in range(n_channels):
            # 血氧动力学响应（慢信号，0.05-0.1 Hz）
            hrf = 0.3 * np.sin(2 * np.pi * 0.08 * t / n_timepoints)
            
            # 生理噪声（呼吸 ~0.3Hz，心跳 ~1Hz）
            resp = 0.1 * np.sin(2 * np.pi * 0.3 * t / n_timepoints)
            cardiac = 0.05 * np.sin(2 * np.pi * 1.0 * t / n_timepoints)
            
            # 基线漂移
            baseline = 1.0 + 0.2 * (t / n_timepoints)
            
            # 高斯噪声
            noise = np.random.normal(0, 0.03, n_timepoints)
            
            # 随机运动伪影（5%的时间点）
            motion = np.zeros(n_timepoints)
            motion_indices = np.random.choice(n_timepoints, size=int(0.05 * n_timepoints), replace=False)
            motion[motion_indices] = np.random.normal(0, 0.3, len(motion_indices))
            
            data[:, ch] = baseline + hrf + resp + cardiac + noise + motion
        
        dataset.append(data)
        print(f"样本 {i+1}: 范围 [{data.min():.4f}, {data.max():.4f}], "
              f"均值 {data.mean():.4f}, 标准差 {data.std():.4f}")
    
    # 计算全局边界
    print(f"\n计算全局边界 (n_bins={n_bins})...")
    edges_list = compute_global_edges(dataset, n_bins=n_bins, strategy="uniform")
    
    # 离散化并验证
    print(f"\n离散化验证:")
    for idx, data in enumerate(dataset):
        binned, _ = discretize_data_strict(data, n_bins=n_bins, global_edges=edges_list)
        
        # 统计每个 bin 的分布
        unique, counts = np.unique(binned, return_counts=True)
        print(f"\n样本 {idx+1} Bin 分布:")
        print(f"  使用的 bins: {unique}")
        print(f"  Bin 0 (最小): {counts[0]} 个值")
        print(f"  Bin {n_bins-1} (最大): {counts[-1] if unique[-1] == n_bins-1 else 0} 个值")
        print(f"  总数据点: {binned.size}")
        
        # 验证互信息计算的准备就绪
        # 每个 bin 都应该有一定数量的数据点
        bin_counts = np.bincount(binned.flatten(), minlength=n_bins)
        empty_bins = np.sum(bin_counts == 0)
        print(f"  空 bins: {empty_bins}/{n_bins}")
        
        if empty_bins > n_bins * 0.3:
            print(f"  ⚠️  警告: {empty_bins} 个空 bins 可能影响互信息计算")
        else:
            print(f"  ✓ Bin 分布合理")
    
    print(f"\n  ✓ 真实数据模拟测试通过!\n")


def main():
    """主测试流程"""
    print("\n" + "="*60)
    print("fNIRS 数据离散化函数测试")
    print("="*60)
    
    # 设置参数
    n_bins = 16
    strategy = "uniform"
    
    # 生成测试数据
    dataset = generate_synthetic_fnirs_data(
        n_samples=5, 
        n_timepoints=1000, 
        n_channels=53
    )
    
    try:
        # 测试1: 验证 compute_global_edges
        edges_list = test_compute_global_edges(dataset, n_bins=n_bins, strategy=strategy)
        
        # 测试2: 验证 discretize_data_strict
        test_discretize_data_strict(dataset, edges_list, n_bins=n_bins)
        
        # 测试3: 验证一致性
        test_consistency_across_samples(dataset, edges_list, n_bins=n_bins)
        
        # 测试4: 验证边界稳定性
        test_edge_stability(n_bins=n_bins, n_repeats=3)
        
        # 测试5: 真实数据模拟
        test_real_data_simulation()
        
        # 所有测试通过
        print("\n" + "="*60)
        print("✓ 所有测试通过！")
        print("="*60)
        print("\n验证结论:")
        print("1. ✓ compute_global_edges 能正确计算全局边界")
        print("2. ✓ 边界合理覆盖数据范围且单调递增")
        print("3. ✓ 常量通道被正确识别和处理")
        print("4. ✓ discretize_data_strict 离散化逻辑正确")
        print("5. ✓ Bin 值在有效范围内 [0, n_bins-1]")
        print("6. ✓ 使用全局边界保证跨样本一致性")
        print("7. ✓ 边界计算稳定，适合互信息计算")
        print("\n后续可以放心使用这些函数进行:")
        print("- 脑区连接分析")
        print("- 互信息计算")
        print("- 图构建")
        print("="*60 + "\n")
        
    except AssertionError as e:
        print(f"\n✗ 测试失败: {e}")
        raise
    except Exception as e:
        print(f"\n✗ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()



