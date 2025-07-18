import numpy as np
import os

from datasets.base_datasets import PointCloudLoader


class ChileanPointCloudLoader(PointCloudLoader):
    def set_properties(self):
        # Chilean地下巷道数据集特性
        self.remove_zero_points = True
        self.remove_ground_plane = False  # 地下巷道可能不需要移除地面
        self.ground_plane_level = None

    def read_pc(self, file_pathname: str) -> np.ndarray:
        """
        读取Chilean数据集的点云文件 (.txt格式)
        格式: [x y z intensity] 每行一个点，无文件头
        返回: Nx3 matrix (x, y, z)
        """
        try:
            # 加载txt文件，每行格式为: x y z intensity
            pc = np.loadtxt(file_pathname)

            if len(pc) == 0:
                print(f"Warning: Empty point cloud file: {file_pathname}")
                return np.array([]).reshape(0, 3)

            # 检查数据格式
            if pc.ndim == 1:
                # 只有一个点的情况
                if len(pc) >= 3:
                    pc = pc.reshape(1, -1)
                else:
                    print(f"Error: Insufficient coordinates in {file_pathname}")
                    return np.array([]).reshape(0, 3)

            if pc.shape[1] < 3:
                print(f"Error: Expected at least 3 columns (x,y,z), got {pc.shape[1]} in {file_pathname}")
                return np.array([]).reshape(0, 3)

            # 只返回xyz坐标（忽略intensity列）
            pc_xyz = pc[:, :3].astype(np.float32)

            # 检查是否有无效值
            if np.any(np.isnan(pc_xyz)) or np.any(np.isinf(pc_xyz)):
                print(f"Warning: Found NaN or Inf values in {file_pathname}")
                # 移除包含NaN或Inf的行
                valid_mask = np.all(np.isfinite(pc_xyz), axis=1)
                pc_xyz = pc_xyz[valid_mask]

            return pc_xyz

        except Exception as e:
            print(f"Error loading {file_pathname}: {e}")
            return np.array([]).reshape(0, 3)