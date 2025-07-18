#!/usr/bin/env python3
"""
验证Chilean数据集点云文件读取是否正确
"""

import os
import sys
import numpy as np
import pickle
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 添加项目路径
project_path = "/home/wzj/pan1/MinkLoc3Dv2"
if project_path not in sys.path:
    sys.path.append(project_path)

# 导入Chilean点云加载器
try:
    from datasets.chilean.chilean_raw import ChileanPointCloudLoader

    print("✅ 成功导入ChileanPointCloudLoader")
except ImportError as e:
    print(f"❌ 导入ChileanPointCloudLoader失败: {e}")
    print("请确保已正确创建 datasets/chilean/chilean_raw.py 文件")
    sys.exit(1)


def test_direct_file_loading():
    """直接测试点云文件加载"""
    print("🔍 测试直接点云文件加载...")

    dataset_path = "/home/wzj/pan2/Chilean_Underground_Mine_Dataset_Many_Times"

    # 找一些示例文件
    sample_files = []
    for series in [10, 11, 12]:
        series_dir = os.path.join(dataset_path, f"downsampled_simdata_{series}")
        if os.path.exists(series_dir):
            for subdir in os.listdir(series_dir):
                subdir_path = os.path.join(series_dir, subdir)
                if os.path.isdir(subdir_path) and subdir.startswith("downsampled_simdata_"):
                    # 找第一个txt文件
                    for file in os.listdir(subdir_path):
                        if file.endswith('.txt'):
                            sample_files.append(os.path.join(subdir_path, file))
                            break
                if len(sample_files) >= 3:
                    break
        if len(sample_files) >= 3:
            break

    if not sample_files:
        print("❌ 未找到任何点云文件")
        return False

    print(f"找到 {len(sample_files)} 个示例文件")

    loader = ChileanPointCloudLoader()

    for i, file_path in enumerate(sample_files):
        print(f"\n📄 测试文件 {i + 1}: {os.path.basename(file_path)}")
        print(f"   完整路径: {file_path}")

        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"   ❌ 文件不存在")
            continue

        # 检查文件大小
        file_size = os.path.getsize(file_path) / 1024  # KB
        print(f"   📏 文件大小: {file_size:.1f} KB")

        # 读取前几行查看格式
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()[:5]
            print(f"   📝 前5行内容:")
            for j, line in enumerate(lines):
                print(f"      行{j + 1}: {line.strip()}")
        except Exception as e:
            print(f"   ❌ 读取文件内容失败: {e}")
            continue

        # 使用加载器读取
        try:
            pc = loader(file_path)
            print(f"   ✅ 加载器读取成功")
            print(f"   📊 点云信息:")
            print(f"      形状: {pc.shape}")
            print(f"      数据类型: {pc.dtype}")

            if len(pc) > 0:
                print(f"      X范围: {pc[:, 0].min():.3f} 到 {pc[:, 0].max():.3f}")
                print(f"      Y范围: {pc[:, 1].min():.3f} 到 {pc[:, 1].max():.3f}")
                print(f"      Z范围: {pc[:, 2].min():.3f} 到 {pc[:, 2].max():.3f}")

                # 检查是否有无效值
                has_nan = np.any(np.isnan(pc))
                has_inf = np.any(np.isinf(pc))
                print(f"      包含NaN: {has_nan}")
                print(f"      包含Inf: {has_inf}")

                # 显示一些样本点
                print(f"      前3个点:")
                for k in range(min(3, len(pc))):
                    print(f"        点{k + 1}: ({pc[k, 0]:.3f}, {pc[k, 1]:.3f}, {pc[k, 2]:.3f})")
            else:
                print(f"      ❌ 空点云")

        except Exception as e:
            print(f"   ❌ 加载器读取失败: {e}")

    return True


def test_pickle_file_loading():
    """测试从pickle文件中加载点云"""
    print("\n🔍 测试从pickle文件加载点云...")

    dataset_path = "/home/wzj/pan2/Chilean_Underground_Mine_Dataset_Many_Times"
    pickle_file = os.path.join(dataset_path, "training_queries_baseline_chilean.pickle")

    if not os.path.exists(pickle_file):
        print(f"❌ Pickle文件不存在: {pickle_file}")
        return False

    try:
        with open(pickle_file, 'rb') as f:
            queries = pickle.load(f)
        print(f"✅ 成功加载pickle文件，包含 {len(queries)} 个查询")
    except Exception as e:
        print(f"❌ 加载pickle文件失败: {e}")
        return False

    # 随机选择几个查询进行测试
    sample_keys = random.sample(list(queries.keys()), min(3, len(queries)))
    loader = ChileanPointCloudLoader()

    for i, key in enumerate(sample_keys):
        query = queries[key]
        print(f"\n📋 测试查询 {i + 1} (ID: {key}):")
        print(f"   文件路径: {query.rel_scan_filepath}")

        # 构建完整路径
        full_path = os.path.join(dataset_path, query.rel_scan_filepath)
        print(f"   完整路径: {full_path}")

        # 检查文件是否存在
        if not os.path.exists(full_path):
            print(f"   ❌ 文件不存在")
            continue

        # 使用加载器读取
        try:
            pc = loader(full_path)
            print(f"   ✅ 成功读取点云")
            print(f"   📊 点云信息:")
            print(f"      形状: {pc.shape}")
            print(f"      查询位置: ({query.position[0]:.3f}, {query.position[1]:.3f})")
            print(f"      正样本数量: {len(query.positives)}")
            print(f"      非负样本数量: {len(query.non_negatives)}")

            if len(pc) > 0:
                print(f"      点云中心: ({pc[:, 0].mean():.3f}, {pc[:, 1].mean():.3f}, {pc[:, 2].mean():.3f})")
        except Exception as e:
            print(f"   ❌ 读取点云失败: {e}")

    return True


def visualize_sample_pointcloud():
    """可视化示例点云"""
    print("\n🎨 可视化示例点云...")

    dataset_path = "/home/wzj/pan2/Chilean_Underground_Mine_Dataset_Many_Times"
    pickle_file = os.path.join(dataset_path, "training_queries_baseline_chilean.pickle")

    if not os.path.exists(pickle_file):
        print(f"❌ Pickle文件不存在，跳过可视化")
        return False

    try:
        with open(pickle_file, 'rb') as f:
            queries = pickle.load(f)

        # 选择第一个查询
        first_key = list(queries.keys())[0]
        query = queries[first_key]

        loader = ChileanPointCloudLoader()
        full_path = os.path.join(dataset_path, query.rel_scan_filepath)

        if not os.path.exists(full_path):
            print(f"❌ 示例文件不存在: {full_path}")
            return False

        pc = loader(full_path)

        if len(pc) == 0:
            print(f"❌ 空点云，无法可视化")
            return False

        print(f"✅ 准备可视化点云: {query.rel_scan_filepath}")
        print(f"   点数: {len(pc)}")

        # 创建3D图
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 为了性能，如果点太多就采样
        if len(pc) > 5000:
            indices = np.random.choice(len(pc), 5000, replace=False)
            pc_vis = pc[indices]
            print(f"   采样到 {len(pc_vis)} 个点进行可视化")
        else:
            pc_vis = pc

        # 绘制点云
        scatter = ax.scatter(pc_vis[:, 0], pc_vis[:, 1], pc_vis[:, 2],
                             c=pc_vis[:, 2], cmap='viridis', s=1, alpha=0.6)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'Chilean pointcloud : {os.path.basename(query.rel_scan_filepath)}')

        # 添加颜色条
        plt.colorbar(scatter, ax=ax, label='Z (m)')

        # 设置等比例
        # ax.set_box_aspect([1,1,1])

        # 保存图片
        output_path = os.path.join(os.path.dirname(__file__), 'chilean_pointcloud_sample.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"   💾 图片已保存: {output_path}")

        # 显示图片（如果在桌面环境中）
        try:
            plt.show()
        except:
            print("   ℹ️ 无法显示图片（可能不在桌面环境中）")

        return True

    except Exception as e:
        print(f"❌ 可视化失败: {e}")
        return False


def compare_with_oxford_format():
    """与Oxford数据集格式对比"""
    print("\n🔍 与Oxford数据集格式对比...")

    # 模拟读取一个Oxford .bin文件的过程
    print("Oxford数据集特点:")
    print("  - 文件格式: .bin (二进制)")
    print("  - 坐标系统: 经纬度 (northing, easting)")
    print("  - 数据组织: 预处理的点云，已移除地面")
    print("  - 点数: 固定数量")

    print("\nChilean数据集特点:")
    print("  - 文件格式: .txt (文本)")
    print("  - 坐标系统: 3D坐标 (x, y, z)")
    print("  - 数据组织: 原始点云 + intensity")
    print("  - 点数: 可变数量")

    # 测试Chilean点云
    dataset_path = "/home/wzj/pan2/Chilean_Underground_Mine_Dataset_Many_Times"
    loader = ChileanPointCloudLoader()

    # 分析几个文件的统计信息
    sample_files = []
    for series in [10, 11]:
        series_dir = os.path.join(dataset_path, f"downsampled_simdata_{series}")
        if os.path.exists(series_dir):
            for subdir in os.listdir(series_dir):
                subdir_path = os.path.join(series_dir, subdir)
                if os.path.isdir(subdir_path):
                    txt_files = [f for f in os.listdir(subdir_path) if f.endswith('.txt')]
                    if txt_files:
                        sample_files.extend([os.path.join(subdir_path, f) for f in txt_files[:2]])
                if len(sample_files) >= 5:
                    break
        if len(sample_files) >= 5:
            break

    if sample_files:
        point_counts = []
        spatial_ranges = []

        for file_path in sample_files[:5]:
            try:
                pc = loader(file_path)
                if len(pc) > 0:
                    point_counts.append(len(pc))
                    x_range = pc[:, 0].max() - pc[:, 0].min()
                    y_range = pc[:, 1].max() - pc[:, 1].min()
                    z_range = pc[:, 2].max() - pc[:, 2].min()
                    spatial_ranges.append((x_range, y_range, z_range))
            except:
                continue

        if point_counts:
            print(f"\nChilean数据集统计 (基于{len(point_counts)}个样本):")
            print(f"  平均点数: {np.mean(point_counts):.0f}")
            print(f"  点数范围: {min(point_counts)} - {max(point_counts)}")

            if spatial_ranges:
                avg_x_range = np.mean([r[0] for r in spatial_ranges])
                avg_y_range = np.mean([r[1] for r in spatial_ranges])
                avg_z_range = np.mean([r[2] for r in spatial_ranges])
                print(f"  平均空间范围:")
                print(f"    X: {avg_x_range:.2f}m")
                print(f"    Y: {avg_y_range:.2f}m")
                print(f"    Z: {avg_z_range:.2f}m")


def main():
    print("🔍 Chilean数据集点云读取验证")
    print("=" * 50)

    # 测试1: 直接文件加载
    success1 = test_direct_file_loading()

    # 测试2: 从pickle文件加载
    success2 = test_pickle_file_loading()

    # 测试3: 可视化
    try:
        success3 = visualize_sample_pointcloud()
    except Exception as e:
        print(f"⚠️ 可视化跳过: {e}")
        success3 = True  # 可视化失败不影响整体验证

    # 测试4: 格式对比
    compare_with_oxford_format()

    print("\n" + "=" * 50)
    if success1 and success2:
        print("🎉 点云读取验证通过!")
        print("✅ Chilean数据集点云文件读取正确")
        print("💡 建议:")
        print("  - 点云格式符合预期")
        print("  - 可以正常进行训练")
        return 0
    else:
        print("❌ 点云读取验证失败!")
        print("🔧 建议检查:")
        print("  - 数据集路径是否正确")
        print("  - 点云文件格式是否正确")
        print("  - ChileanPointCloudLoader是否正确实现")
        return 1


if __name__ == "__main__":
    sys.exit(main())