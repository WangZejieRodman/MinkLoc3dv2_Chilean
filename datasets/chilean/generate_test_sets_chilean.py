import pandas as pd
import numpy as np
import os
import pickle
import random
import glob
from sklearn.neighbors import KDTree

# Chilean Underground Mine Dataset configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
base_path = "/home/wzj/pan2/Chilean_Underground_Mine_Dataset_Many_Times"

# 定义输出目录
OUTPUT_DIR = "/home/wzj/pan2/Chilean_Underground_Mine_Dataset_Many_Times"

# Search radius for positive matches
SEARCH_RADIUS = 10.0  # 10m search radius


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
                # 提取基础名称
                base_name = os.path.basename(file_path).replace("_trajectory.txt", "")
                # 确保对应的点云目录存在
                pointcloud_dir = os.path.join(folder_pattern, base_name)
                if os.path.exists(pointcloud_dir):
                    trajectory_files[base_name] = {
                        'trajectory_file': file_path,
                        'pointcloud_dir': pointcloud_dir,
                        'series': i
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

            # 构造点云文件路径 - 使用相对路径
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

    return pd.DataFrame(all_data)


def split_train_test_for_underground_mine(df):
    """
    按照Chilean数据集的划分策略分割训练集和测试集
    """
    # 1. 定义完全用于测试的时段
    test_series = [15, 18]

    # 2. 定义用于训练的时段
    all_series = df['series'].unique()
    train_series = [s for s in all_series if s not in test_series]

    # 初始化数据集
    df_database = pd.DataFrame(columns=df.columns)  # 数据库集(训练集)
    df_query_1 = pd.DataFrame(columns=df.columns)  # 查询集1 (series 15, 18)
    df_query_2 = pd.DataFrame(columns=df.columns)  # 查询集2 (其他时段的测试部分)

    # 3. 处理测试时段 - 完整轨迹加入查询集1
    for series in test_series:
        series_data = df[df['series'] == series].copy()
        df_query_1 = pd.concat([df_query_1, series_data], ignore_index=True)
        print(f"Added {len(series_data)} samples from series {series} to query set 1")

    # 4. 处理训练时段
    for series in train_series:
        series_data = df[df['series'] == series].copy()

        # 按session分组处理
        for session in series_data['session'].unique():
            session_data = series_data[series_data['session'] == session].copy()
            session_data = session_data.sort_values('timestamp')

            # 策略: 80%用于训练(数据库集)，20%用于测试(查询集2)
            n_total = len(session_data)
            n_train = int(n_total * 0.8)

            # 随机选择训练样本
            all_indices = list(range(n_total))
            random.shuffle(all_indices)

            train_indices = sorted(all_indices[:n_train])
            test_indices = sorted(all_indices[n_train:])

            # 添加到对应数据集
            train_samples = session_data.iloc[train_indices]
            test_samples = session_data.iloc[test_indices]

            df_database = pd.concat([df_database, train_samples], ignore_index=True)
            df_query_2 = pd.concat([df_query_2, test_samples], ignore_index=True)

    return df_database, df_query_1, df_query_2


def output_to_file(output, filename):
    """
    保存pickle文件
    """
    with open(filename, 'wb') as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {filename}")


def construct_query_and_database_sets_chilean(df_database, df_query_1, df_query_2):
    """
    构建Chilean数据集的查询集和数据库集
    修复：保持与原始评估协议的兼容性
    """

    # 构建数据库集的KDTree (使用3D坐标)
    print("Building database KDTree...")
    database_coords = df_database[['x', 'y', 'z']].values
    database_tree = KDTree(database_coords)

    # 构建数据库集字典 - 保持原有格式用于评估
    database_dict = {}
    for idx, row in df_database.iterrows():
        database_dict[len(database_dict)] = {
            'query': row['file'],
            'x': row['x'],
            'y': row['y'],
            'z': row['z'],
            'session': row['session'],
            'series': row['series'],
            'timestamp': row['timestamp']
        }

    # 构建查询集1 (series 15, 18) - 保持原有格式用于评估
    print("Building query set 1...")
    query_set_1 = {}
    for idx, row in df_query_1.iterrows():
        query_coords = np.array([[row['x'], row['y'], row['z']]])
        # 在数据库集中查找10m范围内的正样本
        indices = database_tree.query_radius(query_coords, r=SEARCH_RADIUS)
        positive_indices = indices[0].tolist()

        query_set_1[len(query_set_1)] = {
            'query': row['file'],
            'x': row['x'],
            'y': row['y'],
            'z': row['z'],
            'session': row['session'],
            'series': row['series'],
            'timestamp': row['timestamp'],
            0: positive_indices  # 在数据库集中的正样本索引
        }

    # 构建查询集2 (其他时段的测试部分) - 保持原有格式用于评估
    print("Building query set 2...")
    query_set_2 = {}
    for idx, row in df_query_2.iterrows():
        query_coords = np.array([[row['x'], row['y'], row['z']]])
        # 在数据库集中查找10m范围内的正样本
        indices = database_tree.query_radius(query_coords, r=SEARCH_RADIUS)
        positive_indices = indices[0].tolist()

        query_set_2[len(query_set_2)] = {
            'query': row['file'],
            'x': row['x'],
            'y': row['y'],
            'z': row['z'],
            'session': row['session'],
            'series': row['series'],
            'timestamp': row['timestamp'],
            0: positive_indices  # 在数据库集中的正样本索引
        }

    # 输出统计信息
    print(f"\nDataset Statistics:")
    print(f"Database set size: {len(database_dict)}")
    print(f"Query set 1 size: {len(query_set_1)}")
    print(f"Query set 2 size: {len(query_set_2)}")

    # 计算正样本统计
    if len(query_set_1) > 0:
        pos_counts_1 = [len(query_set_1[i][0]) for i in range(len(query_set_1))]
        print(
            f"Query set 1 - Mean: {np.mean(pos_counts_1):.1f}, Min: {np.min(pos_counts_1)}, Max: {np.max(pos_counts_1)}")

    if len(query_set_2) > 0:
        pos_counts_2 = [len(query_set_2[i][0]) for i in range(len(query_set_2))]
        print(
            f"Query set 2 - Mean: {np.mean(pos_counts_2):.1f}, Min: {np.min(pos_counts_2)}, Max: {np.max(pos_counts_2)}")

    # 保存文件 - 使用完整输出路径
    database_output_path = os.path.join(OUTPUT_DIR, "chilean_evaluation_database.pickle")
    query_1_output_path = os.path.join(OUTPUT_DIR, "chilean_evaluation_query_1.pickle")
    query_2_output_path = os.path.join(OUTPUT_DIR, "chilean_evaluation_query_2.pickle")

    output_to_file([database_dict], database_output_path)
    output_to_file([query_set_1], query_1_output_path)
    output_to_file([query_set_2], query_2_output_path)


def verify_no_data_leakage(df_database, df_query_1, df_query_2):
    """
    验证数据库集和查询集之间没有数据泄漏
    """
    print("\nVerifying no data leakage...")

    # 检查文件路径是否有重复
    database_files = set(df_database['file'].tolist())
    query_1_files = set(df_query_1['file'].tolist())
    query_2_files = set(df_query_2['file'].tolist())

    # 检查database和query_1之间的重叠
    overlap_1 = database_files & query_1_files
    if overlap_1:
        print(f"WARNING: Found {len(overlap_1)} overlapping files between database and query_1")
        print(f"Sample overlaps: {list(overlap_1)[:5]}")
    else:
        print("✓ No file overlap between database and query_1")

    # 检查database和query_2之间的重叠
    overlap_2 = database_files & query_2_files
    if overlap_2:
        print(f"WARNING: Found {len(overlap_2)} overlapping files between database and query_2")
        print(f"Sample overlaps: {list(overlap_2)[:5]}")
    else:
        print("✓ No file overlap between database and query_2")

    # 检查query_1和query_2之间的重叠
    overlap_12 = query_1_files & query_2_files
    if overlap_12:
        print(f"INFO: Found {len(overlap_12)} overlapping files between query_1 and query_2")
    else:
        print("✓ No file overlap between query_1 and query_2")

    return len(overlap_1) == 0 and len(overlap_2) == 0


def main():
    """
    主函数
    """
    print("=" * 80)
    print("CHILEAN UNDERGROUND MINE DATASET - Test Sets Generation")
    print("=" * 80)
    print(f"Search radius: {SEARCH_RADIUS}m")
    print(f"Output directory: {OUTPUT_DIR}")

    # 检查输出目录是否存在
    if not os.path.exists(OUTPUT_DIR):
        print(f"Error: Output directory does not exist: {OUTPUT_DIR}")
        return

    # 设置随机种子
    random.seed(42)
    np.random.seed(42)

    # 1. 查找所有轨迹文件
    print("\n1. Discovering trajectory files...")
    trajectory_files = find_all_trajectory_files()
    print(f"Found {len(trajectory_files)} trajectory files")

    if len(trajectory_files) == 0:
        print("Error: No trajectory files found!")
        return

    # 2. 创建DataFrame
    print("\n2. Loading trajectory data...")
    df_all = create_dataframe_from_trajectories(trajectory_files)

    if len(df_all) == 0:
        print("Error: No valid data found!")
        return

    print(f"Total valid samples: {len(df_all)}")

    # 3. 按照Chilean数据集策略分割数据
    print("\n3. Splitting data according to Chilean dataset strategy...")
    df_database, df_query_1, df_query_2 = split_train_test_for_underground_mine(df_all)

    print(f"\nData split results:")
    print(f"  Database set (training): {len(df_database)} samples")
    print(f"  Query set 1 (series 15, 18): {len(df_query_1)} samples")
    print(f"  Query set 2 (other series test): {len(df_query_2)} samples")

    # 4. 验证数据完整性
    total_expected = len(df_database) + len(df_query_1) + len(df_query_2)
    print(f"  Total samples check: {len(df_all)} original -> {total_expected} split")

    # 5. 验证无数据泄漏
    no_leakage = verify_no_data_leakage(df_database, df_query_1, df_query_2)
    if not no_leakage:
        print("ERROR: Data leakage detected! Please check the split logic.")
        return

    # 6. 构建查询集和数据库集
    print(f"\n4. Constructing query and database sets...")
    construct_query_and_database_sets_chilean(df_database, df_query_1, df_query_2)

    # 7. 保存分割信息 - CSV文件也保存到输出目录
    print(f"\n5. Saving split information...")
    database_csv_path = os.path.join(OUTPUT_DIR, "chilean_database_set.csv")
    query_1_csv_path = os.path.join(OUTPUT_DIR, "chilean_query_1_set.csv")
    query_2_csv_path = os.path.join(OUTPUT_DIR, "chilean_query_2_set.csv")

    df_database.to_csv(database_csv_path, index=False)
    df_query_1.to_csv(query_1_csv_path, index=False)
    df_query_2.to_csv(query_2_csv_path, index=False)

    print(f"Saved {database_csv_path}")
    print(f"Saved {query_1_csv_path}")
    print(f"Saved {query_2_csv_path}")

    print("\n" + "=" * 80)
    print("GENERATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\nGenerated files:")
    print(f"  - {os.path.join(OUTPUT_DIR, 'chilean_evaluation_database.pickle')}")
    print(f"  - {os.path.join(OUTPUT_DIR, 'chilean_evaluation_query_1.pickle')}")
    print(f"  - {os.path.join(OUTPUT_DIR, 'chilean_evaluation_query_2.pickle')}")
    print(f"  - {database_csv_path}")
    print(f"  - {query_1_csv_path}")
    print(f"  - {query_2_csv_path}")

    print(f"\nFinal summary:")
    print(f"  Database samples: {len(df_database)} (from training set)")
    print(f"  Query 1 samples: {len(df_query_1)} (series 15, 18)")
    print(f"  Query 2 samples: {len(df_query_2)} (other series test subset)")
    print(f"  Search radius: {SEARCH_RADIUS}m")
    print(f"  Data leakage: {'None detected' if no_leakage else 'DETECTED - CHECK REQUIRED'}")

    # 验证生成的数据结构
    print(f"\n6. Verifying generated evaluation data structure...")
    try:
        # 检查数据库文件
        database_path = os.path.join(OUTPUT_DIR, "chilean_evaluation_database.pickle")
        with open(database_path, 'rb') as f:
            database_data = pickle.load(f)

        if len(database_data) > 0 and len(database_data[0]) > 0:
            first_db_entry = database_data[0][0]
            print(f"✅ Database structure verification:")
            print(f"   Type: {type(first_db_entry)}")
            print(f"   Keys: {list(first_db_entry.keys())}")
            print(f"   Sample file: {first_db_entry.get('query', 'N/A')}")

        # 检查查询文件
        query_1_path = os.path.join(OUTPUT_DIR, "chilean_evaluation_query_1.pickle")
        with open(query_1_path, 'rb') as f:
            query_1_data = pickle.load(f)

        if len(query_1_data) > 0 and len(query_1_data[0]) > 0:
            first_query = query_1_data[0][0]
            print(f"✅ Query set 1 structure verification:")
            print(f"   Type: {type(first_query)}")
            print(f"   Keys: {list(first_query.keys())}")
            print(f"   Sample file: {first_query.get('query', 'N/A')}")
            print(f"   Positive matches: {len(first_query.get(0, []))} samples")

        print("✅ Evaluation data structure verification passed!")

    except Exception as e:
        print(f"❌ Evaluation data structure verification failed: {e}")


if __name__ == "__main__":
    main()