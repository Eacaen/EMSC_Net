"""
EMSC窗口采样器
实现从完整序列中采样窗口的功能，确保数据的物理意义和连续性
"""

import numpy as np
from typing import List, Tuple, Optional
import os
import joblib
from EMSC_data import load_dataset_from_npz

class EMSCWindowSampler:
    """EMSC窗口采样器，用于从完整序列中采样窗口数据"""
    
    def __init__(self, 
                 dataset_path: str,
                 scalers_dir: Optional[str] = None,
                 window_size: int = 1000, 
                 stride: Optional[int] = None, 
                 augment_factor: int = 1, 
                 state_dim: int = 8,
                 normalize: bool = True,  # 添加归一化控制参数
                 ):
        """
        初始化窗口采样器
        
        Args:
            dataset_path: 数据集npz文件路径
            scalers_dir: 标准化器文件夹路径，如果为None则使用默认路径
            window_size: 窗口大小（包含起始点）
            stride: 采样步长，如果为None则使用window_size
            augment_factor: 数据增强因子，每个序列生成多少个窗口
            state_dim: 状态向量维度
            normalize: 是否进行数据归一化，默认为True
        """
        self.window_size = window_size
        self.stride = stride if stride is not None else window_size
        self.augment_factor = augment_factor
        self.state_dim = state_dim
        self.normalize = normalize  # 保存归一化控制参数
        
        # 设置数据集路径
        self.dataset_path = dataset_path
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"数据集文件不存在: {dataset_path}")
        
        # 设置标准化器目录
        if scalers_dir is None:
            self.scalers_dir = os.path.join(os.path.dirname(self.dataset_path), 'scalers')
        else:
            self.scalers_dir = scalers_dir
        
        # 检查标准化器目录
        if not os.path.exists(self.scalers_dir):
            print(f"警告: 标准化器目录不存在: {self.scalers_dir}")
            if self.normalize:
                print("警告: 归一化已启用但标准化器目录不存在，将使用未归一化的数据")
                self.normalize = False  # 如果目录不存在且需要归一化，则禁用归一化
        else:
            print(f"标准化器目录: {self.scalers_dir}")
            # 检查标准化器文件
            x_scaler_path = os.path.join(self.scalers_dir, 'x_scaler.save')
            y_scaler_path = os.path.join(self.scalers_dir, 'y_scaler.save')
            if self.normalize:
                if not os.path.exists(x_scaler_path):
                    print(f"警告: X标准化器文件不存在: {x_scaler_path}")
                    self.normalize = False
                if not os.path.exists(y_scaler_path):
                    print(f"警告: Y标准化器文件不存在: {y_scaler_path}")
                    self.normalize = False
        
        # 标准化器
        self.x_scaler = None
        self.y_scaler = None
        
        # 如果需要归一化，尝试加载标准化器
        if self.normalize:
            self._try_load_scalers()
            if self.x_scaler is None or self.y_scaler is None:
                print("警告: 标准化器加载失败，将使用未归一化的数据")
                self.normalize = False
        
        # 加载数据集
        self.X_sequences, self.Y_sequences = self._load_dataset()
        if self.X_sequences is None or self.Y_sequences is None:
            raise ValueError("数据集加载失败")
        
        # 物理量序列（初始为None，需要显式调用recover_physical_quantities来恢复）
        self.physical_sequences = None
        
        print(f"窗口采样器初始化完成:")
        print(f"- 数据集路径: {dataset_path}")
        print(f"- 标准化器目录: {self.scalers_dir}")
        print(f"- 窗口大小: {window_size}")
        print(f"- 采样步长: {self.stride}")
        print(f"- 增强因子: {augment_factor}")
        print(f"- 序列数量: {len(self.X_sequences)}")
        print(f"- 序列长度范围: {min(len(x) for x in self.X_sequences)} - {max(len(x) for x in self.X_sequences)}")
        print(f"- 数据归一化: {'启用' if self.normalize else '禁用'}")
    
    def _try_load_scalers(self) -> bool:
        """
        尝试加载标准化器
        
        Returns:
            bool: 是否成功加载标准化器
        """
        x_scaler_path = os.path.join(self.scalers_dir, 'x_scaler.save')
        y_scaler_path = os.path.join(self.scalers_dir, 'y_scaler.save')
        
        try:
            if os.path.exists(x_scaler_path) and os.path.exists(y_scaler_path):
                self.x_scaler = joblib.load(x_scaler_path)
                self.y_scaler = joblib.load(y_scaler_path)
                print(f"成功加载标准化器:")
                print(f"- X标准化器: {x_scaler_path}")
                print(f"- Y标准化器: {y_scaler_path}")
                return True
            else:
                missing_files = []
                if not os.path.exists(x_scaler_path):
                    missing_files.append('x_scaler.save')
                if not os.path.exists(y_scaler_path):
                    missing_files.append('y_scaler.save')
                print(f"警告: 以下标准化器文件不存在: {', '.join(missing_files)}")
                return False
        except Exception as e:
            print(f"加载标准化器时出错: {e}")
            return False
    
    def load_scalers(self) -> bool:
        """
        加载标准化器
        
        Returns:
            bool: 是否成功加载标准化器
        """
        return self._try_load_scalers()
    
    def _load_dataset(self) -> Tuple[Optional[List[np.ndarray]], Optional[List[np.ndarray]]]:
        """
        加载数据集
        
        Returns:
            Tuple[Optional[List[np.ndarray]], Optional[List[np.ndarray]]]: 
                (X_sequences, Y_sequences) 或 (None, None)如果加载失败
        """
        try:
            X_sequences, Y_sequences = load_dataset_from_npz(self.dataset_path)
            if X_sequences is None or Y_sequences is None:
                print(f"警告: 数据集加载失败: {self.dataset_path}")
                return None, None
            
            # 打印数据集信息
            print(f"数据集加载成功:")
            print(f"- X序列数量: {len(X_sequences)}")
            print(f"- Y序列数量: {len(Y_sequences)}")
            print(f"- X序列类型: {type(X_sequences[0])}")
            print(f"- Y序列类型: {type(Y_sequences[0])}")
            if len(X_sequences) > 0:
                print(f"- X序列形状: {X_sequences[0].shape if hasattr(X_sequences[0], 'shape') else 'unknown'}")
                print(f"- Y序列形状: {Y_sequences[0].shape if hasattr(Y_sequences[0], 'shape') else 'unknown'}")
            
            return X_sequences, Y_sequences
        except Exception as e:
            print(f"加载数据集时出错: {e}")
            return None, None
    
    def recover_physical_quantities(self) -> List[np.ndarray]:
        """
        恢复物理量序列
        
        Returns:
            List[np.ndarray]: 物理量序列列表
        """
        if self.physical_sequences is None:
            print("恢复物理量序列...")
            self.physical_sequences = []
            for x_seq in self.X_sequences:
                # 确保x_seq是numpy数组
                x_seq = np.array(x_seq, dtype=np.float32)
                
                # 提取增量和初值
                delta_strain = x_seq[:, 0]
                delta_time = x_seq[:, 1]
                delta_temperature = x_seq[:, 2]
                init_strain = x_seq[0, 3]
                init_time = x_seq[0, 4]
                init_temp = x_seq[0, 5]
                
                # 计算累积值得到物理量
                strain = np.cumsum(delta_strain) + init_strain
                time = np.cumsum(delta_time) + init_time
                temperature = np.cumsum(delta_temperature) + init_temp
                
                physical_seq = np.column_stack([strain, time, temperature])
                self.physical_sequences.append(physical_seq)
            print(f"物理量序列恢复完成，共 {len(self.physical_sequences)} 条序列")
        return self.physical_sequences
    
    def sample_windows(self, 
                      physical_sequences: List[np.ndarray],
                      Y_sequences: List[np.ndarray],
                      shuffle: bool = True) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        从物理量序列中采样窗口
        
        Args:
            physical_sequences: 物理量序列列表
            Y_sequences: 输出序列列表
            shuffle: 是否随机打乱窗口顺序
            
        Returns:
            Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]: 
                - 窗口输入数据列表
                - 窗口输出数据列表
                - 初始状态向量列表
        """
        windows_X = []
        windows_Y = []
        init_states = []
        
        for i, (physical_seq, Y_seq) in enumerate(zip(physical_sequences, Y_sequences)):
            # 确保序列是numpy数组
            physical_seq = np.array(physical_seq, dtype=np.float32)
            Y_seq = np.array(Y_seq, dtype=np.float32)
            
            seq_len = len(physical_seq)
            if seq_len < self.window_size:
                continue
                
            # 计算可采样的窗口数量
            max_windows = (seq_len - self.window_size) // self.stride + 1
            num_windows = min(max_windows, self.augment_factor)
            
            # 随机选择起始位置
            if shuffle:
                start_indices = np.random.choice(
                    range(0, seq_len - self.window_size + 1, self.stride),
                    size=num_windows,
                    replace=False
                )
            else:
                start_indices = range(0, seq_len - self.window_size + 1, self.stride)[:num_windows]
            
            for start_idx in start_indices:
                end_idx = start_idx + self.window_size
                
                # 提取窗口的物理量数据
                window_physical = physical_seq[start_idx:end_idx]
                window_Y = Y_seq[start_idx:end_idx]
                
                # 计算窗口内的增量
                delta_strain = np.zeros(self.window_size, dtype=np.float32)
                delta_time = np.zeros(self.window_size, dtype=np.float32)
                delta_temperature = np.zeros(self.window_size, dtype=np.float32)
                
                for j in range(1, self.window_size):
                    delta_strain[j] = window_physical[j, 0] - window_physical[j-1, 0]
                    delta_time[j] = window_physical[j, 1] - window_physical[j-1, 1]
                    delta_temperature[j] = window_physical[j, 2] - window_physical[j-1, 2]
                
                # 获取窗口起始点的值作为初始值
                init_strain = window_physical[0, 0]
                init_time = window_physical[0, 1]
                init_temp = window_physical[0, 2]
                
                # 构造窗口特征
                window_X = np.column_stack([
                    delta_strain,
                    delta_time,
                    delta_temperature,
                    np.full(self.window_size, init_strain),
                    np.full(self.window_size, init_time),
                    np.full(self.window_size, init_temp)
                ])
                
                # 如果有标准化器，进行归一化
                if self.normalize:
                    window_X, window_Y = self._normalize_window(window_X, window_Y)
                
                # 构造初始状态（使用零初始化）
                init_state = np.zeros(self.state_dim, dtype=np.float32)
                
                windows_X.append(window_X)
                windows_Y.append(window_Y)
                init_states.append(init_state)
        
        print(f"窗口采样完成:")
        print(f"- 总窗口数: {len(windows_X)}")
        print(f"- 窗口大小: {self.window_size}")
        print(f"- 数据格式: X={type(windows_X[0])}, Y={type(windows_Y[0])}")
        print(f"- 数据形状: X={windows_X[0].shape}, Y={windows_Y[0].shape}")
        print(f"- 数据归一化: {'启用' if self.normalize else '禁用'}")
        
        return windows_X, windows_Y, init_states
    
    def _normalize_window(self, window_X: np.ndarray, window_Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        对窗口数据进行归一化
        
        Args:
            window_X: 窗口输入数据 (window_size, 6)
            window_Y: 窗口输出数据 (window_size, 1)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: 归一化后的窗口数据
        """
        if not self.normalize or self.x_scaler is None or self.y_scaler is None:
            return window_X, window_Y
            
        # 直接对整个窗口数据进行归一化
        normalized_X = self.x_scaler.transform(window_X)
        normalized_Y = self.y_scaler.transform(window_Y)
        
        return normalized_X, normalized_Y

def test_window_sampler():
    """测试窗口采样器功能"""
    # 创建采样器实例
    dataset_path = "/Users/tianyunhu/Documents/temp/code/Test_app/EMSC_Model/msc_models/dataset_EMSC_tt/dataset_EMSC_tt.npz"
    scalers_dir = os.path.join(os.path.dirname(dataset_path), 'scalers')
    
    print(f"数据集路径: {dataset_path}")
    print(f"标准化器目录: {scalers_dir}")

    
    # 测试归一化的情况

    sampler_with_norm = EMSCWindowSampler(
        dataset_path=dataset_path,
        scalers_dir=scalers_dir,
        window_size=1000,
        stride=2,
        augment_factor=1,
        state_dim=8,
        normalize=True  # 启用归一化
    )

    # 选择要测试的采样器
    sampler = sampler_with_norm  # 或 sampler_no_norm
    
    # 显式恢复物理量序列（按需调用）
    physical_sequences = sampler.recover_physical_quantities()
    print("\n物理量序列类型检查:")
    print(f"物理量序列类型: {type(physical_sequences)}")
    print(f"物理量序列长度: {len(physical_sequences)}")
    print(f"单个物理量序列类型: {type(physical_sequences[0])}")
    print(f"单个物理量序列形状: {physical_sequences[0].shape}")


    # 使用恢复后的序列进行窗口采样
    windows_X, windows_Y, init_states = sampler.sample_windows(
        physical_sequences,
        sampler.Y_sequences,
        shuffle=True
    )
    
    print("\n窗口采样结果类型检查:")
    print(f"输入窗口列表类型: {type(windows_X)}")
    print(f"输出窗口列表类型: {type(windows_Y)}")
    print(f"初始状态列表类型: {type(init_states)}")
    print(f"输入窗口列表长度: {len(windows_X)}")
    print(f"单个输入窗口类型: {type(windows_X[0])}")
    print(f"单个输入窗口形状: {windows_X[0].shape}")
    print(f"单个输出窗口类型: {type(windows_Y[0])}")
    print(f"单个输出窗口形状: {windows_Y[0].shape}")
    print(f"单个初始状态类型: {type(init_states[0])}")
    print(f"单个初始状态形状: {init_states[0].shape}")
    print(f"数据归一化状态: {'启用' if sampler.normalize else '禁用'}")
    

    for i in range(min(3, len(physical_sequences))):
        org_physical_seq = physical_sequences[i]
        print(f"\n窗口 {i} 的物理量范围:")
        print(f"应变: {org_physical_seq[:, 0][:10]}")
        print(f"时间: {org_physical_seq[:, 1][:10]}")
        print(f"温度: {org_physical_seq[:, 2][:10]}")
        print(f"应变范围: [{org_physical_seq[:, 0].min():.3f}, {org_physical_seq[:, 0].max():.3f}]")
        print(f"时间范围: [{org_physical_seq[:, 1].min():.3f}, {org_physical_seq[:, 1].max():.3f}]")
        print(f"温度范围: [{org_physical_seq[:, 2].min():.3f}, {org_physical_seq[:, 2].max():.3f}]")
  

    # 验证窗口数据的物理意义
    for i in range(min(3, len(windows_X))):
        window_X = windows_X[i]
        print(f"\n窗口 {i} 的物理量范围:")
        print(f"应变增量: [{window_X[:, 0].min():.3f}, {window_X[:, 0].max():.3f}]")
        print(f"时间增量: [{window_X[:, 1].min():.3f}, {window_X[:, 1].max():.3f}]")
        print(f"温度增量: [{window_X[:, 2].min():.3f}, {window_X[:, 2].max():.3f}]")
        print(f"初始应变: {window_X[0, 3]:.3f}")
        print(f"初始时间: {window_X[0, 4]:.3f}")
        print(f"初始温度: {window_X[0, 5]:.3f}")

if __name__ == "__main__":
    test_window_sampler() 