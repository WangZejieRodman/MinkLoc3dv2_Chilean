import pandas as pd
import numpy as np
import os
import pandas as pd
from sklearn.neighbors import KDTree
import pickle
import random
import glob

# 导入TrainingTuple类 - 关键修复点1
from datasets.base_datasets import TrainingTuple

# Chilean Underground Mine Dataset configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
base_path = "/home/wzj/pan2/Chilean_Underground_Mine_Dataset_Many_Times"

# 定义输出目录
OUTPUT_DIR = "/home/wzj/pan2/Chilean_Underground_Mine_Dataset_Many_Times"


def load_trajectory_data(trajectory_file):
    """
    加载轨迹数据文件
    格式: timestamp x y z qx qy qz qw
    """
    try:
        with open(trajectory_file, 'r') as f:
            lines = f.readlines()
            # 跳过头部行
            data_lines = [line.strip() for line in lines[1:] if line.strip()]

        trajectory_data = []
        for line in data_lines:
            values = line.split()
            if len(values) >= 4:  # 至少需要 timestamp x y z
                timestamp = int(float(values[0]))
                x, y, z = float(values[1]), float(values[2]), float(values[3])
                trajectory_data.append([timestamp, x, y, z])

        return np.array(trajectory_data)
    except Exception as e:
        print(f"Error loading {trajectory_file}: {e}")
        return np.array([])


def find_all_trajectory_files():
    """
    查找所有轨迹文件并按时段分组
    """
    trajectory_files = {}

    # 扫描 downsampled_simdata_10 到 downsampled_simdata_20 目录
    for i in range(10, 21):
        folder_pattern = os.path.join(base_path, f"downsampled_simdata_{i}")
        if os.path.exists(folder_pattern):
            # 查找该目录下的所有轨迹文件
            trajectory_pattern = os.path.join(folder_pattern, "*_trajectory.txt")
            files = glob.glob(trajectory_pattern)

            for file_path in files:
                # 提取基础名称，例如 downsampled_simdata_100
                base_name = os.path.basename(file_path).replace("_trajectory.txt", "")

                # 确保对应的点云目录存在
                pointcloud_dir = os.path.join(folder_pattern, base_name)
                if os.path.exists(pointcloud_dir):
                    trajectory_files[base_name] = {
                        'trajectory_file': file_path,
                        'pointcloud_dir': pointcloud_dir,
                        'series': i  # 10, 11, 12, ..., 20
                    }

    return trajectory_files


def create_dataframe_from_trajectories(trajectory_files):
    """
    从轨迹文件创建DataFrame
    """
    all_data = []

    for session_name, info in trajectory_files.items():
        print(f"Processing {session_name}...")

        # 加载轨迹数据
        trajectory_data = load_trajectory_data(info['trajectory_file'])

        if len(trajectory_data) == 0:
            print(f"  Warning: No valid trajectory data found for {session_name}")
            continue

        # 检查对应的点云文件是否存在
        pointcloud_dir = info['pointcloud_dir']
        for traj_point in trajectory_data:
            timestamp = int(traj_point[0])
            x, y, z = traj_point[1], traj_point[2], traj_point[3]

            # 构造点云文件路径 - 使用相对路径，相对于数据集根目录
            relative_pointcloud_file = os.path.relpath(
                os.path.join(pointcloud_dir, f"{session_name}_{timestamp:03d}.txt"),
                base_path
            )

            # 检查文件是否存在（使用绝对路径检查）
            absolute_pointcloud_file = os.path.join(pointcloud_dir, f"{session_name}_{timestamp:03d}.txt")
            if os.path.exists(absolute_pointcloud_file):
                all_data.append({
                    'file': relative_pointcloud_file,  # 使用相对路径
                    'session': session_name,
                    'series': info['series'],
                    'timestamp': timestamp,
                    'x': x,
                    'y': y,
                    'z': z
                })

        print(f"  Found {len([d for d in all_data if d['session'] == session_name])} valid point clouds")

    return pd.DataFrame(all_data)


def split_train_test_for_underground_mine(df):
    """
    为地下巷道数据集划分训练集和测试集
    """

    # 1. 定义完全用于测试的时段 (选择 series 15 和 18)
    test_series = [15, 18]

    # 2. 定义完全用于训练的时段
    all_series = df['series'].unique()
    train_series = [s for s in all_series if s not in test_series]

    print(f"Test series (complete sessions): {test_series}")
    print(f"Train series: {train_series}")

    # 初始化训练集和测试集
    df_train = pd.DataFrame(columns=df.columns)
    df_test = pd.DataFrame(columns=df.columns)

    # 3. 处理测试时段 - 完整轨迹加入测试集
    for series in test_series:
        series_data = df[df['series'] == series].copy()
        df_test = pd.concat([df_test, series_data], ignore_index=True)
        print(f"Added {len(series_data)} samples from series {series} to test set")

    # 4. 处理训练时段
    for series in train_series:
        series_data = df[df['series'] == series].copy()

        # 按session分组处理
        for session in series_data['session'].unique():
            session_data = series_data[series_data['session'] == session].copy()
            session_data = session_data.sort_values('timestamp')  # 按时间戳排序

            # 策略: 80%用于训练，20%用于测试（模拟轨迹段间隔）
            n_total = len(session_data)
            n_train = int(n_total * 0.8)

            # 随机选择训练样本（非连续，模拟实际场景）
            all_indices = list(range(n_total))
            random.shuffle(all_indices)

            train_indices = sorted(all_indices[:n_train])
            test_indices = sorted(all_indices[n_train:])

            # 添加到训练集和测试集
            train_samples = session_data.iloc[train_indices]
            test_samples = session_data.iloc[test_indices]

            df_train = pd.concat([df_train, train_samples], ignore_index=True)
            df_test = pd.concat([df_test, test_samples], ignore_index=True)

            print(f"Session {session}: {len(train_samples)} to train, {len(test_samples)} to test")

    return df_train, df_test


def construct_query_dict_underground(df_centroids, filename, pos_radius=5.0, neg_radius=25.0):
    """
    为地下巷道环境构建查询字典
    关键修复点2：使用TrainingTuple对象而不是字典

    参数:
    - pos_radius: 正样本半径 (7米，适合地下巷道的密集采样)
    - neg_radius: 负样本排除半径 (35米，考虑地下巷道相对较小的空间)
    - filename: 输出文件名（完整路径）
    """

    # 使用3D坐标构建KDTree (x, y, z)
    coords = df_centroids[['x', 'y', 'z']].values
    tree = KDTree(coords)

    # 查找邻居
    ind_pos = tree.query_radius(coords, r=pos_radius)
    ind_neg_exclude = tree.query_radius(coords, r=neg_radius)

    queries = {}
    positive_counts = []
    negative_counts = []

    for i in range(len(df_centroids)):
        row = df_centroids.iloc[i]
        query_file = row["file"]

        # 正样本: pos_radius内的样本 - 查询样本本身
        positives = np.setdiff1d(ind_pos[i], [i])

        # 负样本: 全部样本 - neg_radius内的样本
        negatives = np.setdiff1d(df_centroids.index.values.tolist(), ind_neg_exclude[i])

        # 计算non_negatives（正样本 + 中间区域样本）
        # non_negatives包含所有不应该被视为负样本的样本
        non_negatives = ind_neg_exclude[i]

        # 随机打乱负样本顺序
        random.shuffle(negatives)

        # 创建TrainingTuple对象 - 关键修复点3
        training_tuple = TrainingTuple(
            id=i,
            timestamp=int(row['timestamp']),
            rel_scan_filepath=query_file,
            positives=positives,
            non_negatives=non_negatives,
            position=np.array([row['x'], row['y']])  # 使用2D位置以兼容原框架
        )

        queries[i] = training_tuple

        positive_counts.append(len(positives))
        negative_counts.append(len(negatives))

    # 保存查询字典
    with open(filename, 'wb') as handle:
        pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # 打印统计信息
    print(f"Saved {filename}")
    print(f"  Total queries: {len(queries)}")
    print(f"  Positive samples per query - Mean: {np.mean(positive_counts):.1f}, "
          f"Min: {np.min(positive_counts)}, Max: {np.max(positive_counts)}")
    print(f"  Negative samples per query - Mean: {np.mean(negative_counts):.1f}, "
          f"Min: {np.min(negative_counts)}, Max: {np.max(negative_counts)}")

    return queries


def analyze_dataset_distribution(df):
    """
    分析数据集分布
    """
    print("\n" + "=" * 60)
    print("DATASET ANALYSIS")
    print("=" * 60)

    # 按时段统计
    print("\nSamples per series:")
    series_counts = df.groupby('series').size().sort_index()
    for series, count in series_counts.items():
        print(f"  Series {series}: {count} samples")

    # 按session统计
    print(f"\nTotal sessions: {df['session'].nunique()}")
    print("Samples per session:")
    session_counts = df.groupby('session').size().sort_values(ascending=False)
    for session, count in session_counts.head(10).items():
        print(f"  {session}: {count} samples")

    if len(session_counts) > 10:
        print(f"  ... and {len(session_counts) - 10} more sessions")

    # 空间分布
    print(f"\nSpatial distribution:")
    print(f"  X range: {df['x'].min():.2f} to {df['x'].max():.2f} "
          f"(span: {df['x'].max() - df['x'].min():.2f}m)")
    print(f"  Y range: {df['y'].min():.2f} to {df['y'].max():.2f} "
          f"(span: {df['y'].max() - df['y'].min():.2f}m)")
    print(f"  Z range: {df['z'].min():.2f} to {df['z'].max():.2f} "
          f"(span: {df['z'].max() - df['z'].min():.2f}m)")


def main():
    """
    主函数
    """
    print("=" * 80)
    print("CHILEAN UNDERGROUND MINE DATASET - Training Tuples Generation")
    print("=" * 80)
    print(f"Output directory: {OUTPUT_DIR}")

    # 检查输出目录是否存在
    if not os.path.exists(OUTPUT_DIR):
        print(f"Error: Output directory does not exist: {OUTPUT_DIR}")
        return

    # 设置随机种子以确保可重现性
    random.seed(42)
    np.random.seed(42)

    # 1. 查找所有轨迹文件
    print("\n1. Discovering trajectory files...")
    trajectory_files = find_all_trajectory_files()
    print(f"Found {len(trajectory_files)} trajectory files")

    if len(trajectory_files) == 0:
        print("Error: No trajectory files found!")
        return

    # 显示发现的文件
    print("Discovered sessions:")
    for session_name, info in sorted(trajectory_files.items()):
        print(f"  {session_name} (series {info['series']})")

    # 2. 创建DataFrame
    print("\n2. Loading trajectory data and creating DataFrame...")
    df_all = create_dataframe_from_trajectories(trajectory_files)

    if len(df_all) == 0:
        print("Error: No valid data found!")
        return

    print(f"Total valid samples: {len(df_all)}")

    # 3. 分析数据集
    analyze_dataset_distribution(df_all)

    # 4. 划分训练集和测试集
    print("\n3. Splitting into train and test sets...")
    df_train, df_test = split_train_test_for_underground_mine(df_all)

    print(f"\nFinal split:")
    print(f"  Training samples: {len(df_train)}")
    print(f"  Test samples: {len(df_test)}")
    print(f"  Train/Test ratio: {len(df_train) / len(df_test):.2f}")

    # 5. 构建查询字典
    print("\n4. Constructing query dictionaries...")

    # 为地下巷道调整参数
    pos_radius = 7.0
    neg_radius = 35.0

    print(f"Using parameters for underground environment:")
    print(f"  Positive radius: {pos_radius}m")
    print(f"  Negative exclusion radius: {neg_radius}m")

    # 构建训练查询
    print(f"\nBuilding training queries...")
    train_output_path = os.path.join(OUTPUT_DIR, "training_queries_baseline_chilean.pickle")
    construct_query_dict_underground(
        df_train,
        train_output_path,
        pos_radius=pos_radius,
        neg_radius=neg_radius
    )

    # 构建测试查询
    print(f"\nBuilding test queries...")
    test_output_path = os.path.join(OUTPUT_DIR, "test_queries_baseline_chilean.pickle")
    construct_query_dict_underground(
        df_test,
        test_output_path,
        pos_radius=pos_radius,
        neg_radius=neg_radius
    )

    # 6. 保存数据集信息
    print("\n5. Saving dataset information...")

    # 保存训练集和测试集的文件列表
    train_csv_path = os.path.join(OUTPUT_DIR, "train_set_chilean.csv")
    test_csv_path = os.path.join(OUTPUT_DIR, "test_set_chilean.csv")

    df_train[['file', 'session', 'series', 'timestamp', 'x', 'y', 'z']].to_csv(
        train_csv_path, index=False
    )
    df_test[['file', 'session', 'series', 'timestamp', 'x', 'y', 'z']].to_csv(
        test_csv_path, index=False
    )

    print(f"Saved {train_csv_path}")
    print(f"Saved {test_csv_path}")

    print("\n" + "=" * 80)
    print("GENERATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\nGenerated files:")
    print(f"  - {train_output_path}")
    print(f"  - {test_output_path}")
    print(f"  - {train_csv_path}")
    print(f"  - {test_csv_path}")
    print(f"\nDataset summary:")
    print(f"  Total samples: {len(df_all)}")
    print(f"  Training samples: {len(df_train)}")
    print(f"  Test samples: {len(df_test)}")
    print(f"  Sessions used: {df_all['session'].nunique()}")
    print(f"  Series covered: {sorted(df_all['series'].unique())}")

    # 验证生成的数据结构 - 关键修复点4
    print(f"\n6. Verifying generated data structure...")
    try:
        with open(train_output_path, 'rb') as f:
            train_queries = pickle.load(f)

        # 检查第一个查询的结构
        if len(train_queries) > 0:
            first_query = train_queries[0]
            print(f"✅ First training query structure:")
            print(f"   Type: {type(first_query)}")
            print(f"   ID: {first_query.id}")
            print(f"   Timestamp: {first_query.timestamp}")
            print(f"   File: {first_query.rel_scan_filepath}")
            print(f"   Positives: {len(first_query.positives)} samples")
            print(f"   Non-negatives: {len(first_query.non_negatives)} samples")
            print(f"   Position shape: {first_query.position.shape}")

        print("✅ Data structure verification passed!")

    except Exception as e:
        print(f"❌ Data structure verification failed: {e}")


if __name__ == "__main__":
    main()