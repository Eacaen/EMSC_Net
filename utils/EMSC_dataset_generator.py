import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import joblib
from datetime import datetime
import glob
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import matplotlib as mpl
import tensorflow as tf
import json
from tqdm import tqdm
import time

# 设置中文字体
def set_chinese_font():
    """设置中文字体"""
    # 尝试设置不同的中文字体
    chinese_fonts = [
        'SimHei',  # 黑体
        'Microsoft YaHei',  # 微软雅黑
        'PingFang SC',  # 苹方
        'STHeiti',  # 华文黑体
        'Arial Unicode MS'  # Arial Unicode
    ]
    
    # 检查系统字体
    system_fonts = set([f.name for f in mpl.font_manager.fontManager.ttflist])
    
    # 选择第一个可用的中文字体
    for font in chinese_fonts:
        if font in system_fonts:
            plt.rcParams['font.family'] = font
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
            print(f"使用中文字体: {font}")
            return True
    
    print("警告: 未找到合适的中文字体，图表中文可能无法正确显示")
    return False

class EMSCDatasetGenerator:
    """
    EMSC数据集生成器类，用于处理和生成训练数据
    支持NPZ和TFRecord两种格式
    """
    def __init__(self, target_sequence_length=1000, window_size=None, stride=None, max_subsequences=200,
                 normalize=True, scaler_type='minmax', output_format='npz'):
        """
        初始化数据集生成器
        
        参数:
        target_sequence_length: 目标序列长度
        window_size: 滑动窗口大小，默认为target_sequence_length
        stride: 滑动窗口步长，默认为target_sequence_length//10
        max_subsequences: 每个序列最多生成的子序列数
        normalize: 是否对数据进行归一化，默认True
        scaler_type: 归一化方法类型，可选值：
                    'minmax' - MinMaxScaler (默认)
                    'standard' - StandardScaler (Z-score标准化)
                    'robust' - RobustScaler (基于中位数和四分位数的标准化)
                    'maxabs' - MaxAbsScaler (基于最大绝对值的标准化)
        output_format: 输出格式，可选值：'npz', 'tfrecord', 'both'
        """
        self.target_sequence_length = target_sequence_length
        self.window_size = window_size if window_size is not None else target_sequence_length
        self.stride = stride if stride is not None else target_sequence_length // 10
        self.max_subsequences = max_subsequences
        
        # 归一化选项
        self.normalize = normalize
        self.scaler_type = scaler_type.lower()
        
        # 输出格式选项
        self.output_format = output_format.lower()
        valid_formats = ['npz', 'tfrecord', 'both']
        if self.output_format not in valid_formats:
            raise ValueError(f"output_format必须是以下之一: {valid_formats}")
        
        # 验证scaler_type
        valid_scalers = ['minmax', 'standard', 'robust', 'maxabs']
        if self.scaler_type not in valid_scalers:
            raise ValueError(f"scaler_type必须是以下之一: {valid_scalers}")
        
        # 初始化标准化器
        if self.normalize:
            self.x_scaler = self._create_scaler()
            self.y_scaler = self._create_scaler()
        else:
            self.x_scaler = None
            self.y_scaler = None
        
        # 初始化数据存储
        self.X_paths = []
        self.Y_paths = []
        
        # 数据集基础路径
        self.base_dir = "/Users/tianyunhu/Documents/temp/code/Test_app/EMSC_Model/msc_models"
        self.dataset_name = None  # 将在prepare_and_save_dataset中设置
        
        # 数据存储
        self.sequence_lengths = []
        self.temperature_stats = {}
        self.strain_rate_stats = {}  # 新增：应变率统计
        
        # 定义列名映射
        self.column_mapping = {
            'time': 'time',
            'true_strain': 'true_strain',
            'true_stress': 'true_stress',
            'temperature': 'temperature',
            'delta_strain': 'delta_strain',  # 原 Δε
            'delta_time': 'delta_time',      # 原 Δt
            'delta_temperature': 'delta_temperature',  # 原 ΔT

            'init_strain': 'init_strain',    # 初始应变
            'init_time': 'init_time',        # 初始时间
            'init_temp': 'init_temp'         # 初始温度
        }
        
    def _create_scaler(self):
        """
        根据scaler_type创建相应的标准化器
        
        返回:
        scaler: 标准化器实例
        """
        if self.scaler_type == 'minmax':
            return MinMaxScaler()
        elif self.scaler_type == 'standard':
            return StandardScaler()
        elif self.scaler_type == 'robust':
            return RobustScaler()
        elif self.scaler_type == 'maxabs':
            return MaxAbsScaler()
        else:
            raise ValueError(f"不支持的scaler_type: {self.scaler_type}")
    
    def get_scaler_info(self):
        """
        获取当前标准化器信息
        
        返回:
        dict: 包含标准化器信息的字典
        """
        return {
            'normalize': self.normalize,
            'scaler_type': self.scaler_type if self.normalize else None,
            'scaler_name': self.x_scaler.__class__.__name__ if self.normalize else None
        }

    def get_dataset_paths(self, dataset_name):
        """
        获取数据集相关的所有路径
        
        参数:
        dataset_name: 数据集名称
        
        返回:
        dict: 包含所有相关路径的字典
        """
        # 使用数据集名称作为顶层目录
        dataset_dir = os.path.join(self.base_dir, dataset_name)
        
        return {
            'dataset_dir': dataset_dir,                    # 数据集目录
            'dataset_file': os.path.join(dataset_dir, f'{dataset_name}.npz'),  # NPZ数据集文件
            'tfrecord_file': os.path.join(dataset_dir, f'{dataset_name}.tfrecord'),  # TFRecord数据集文件
            'tfrecord_info': os.path.join(dataset_dir, f'{dataset_name}.tfrecord.info.json'),  # TFRecord信息文件
            'scaler_dir': os.path.join(dataset_dir, 'scalers'),  # 标准化器目录
            'x_scaler_file': os.path.join(dataset_dir, 'scalers', 'x_scaler.save'),  # X标准化器文件
            'y_scaler_file': os.path.join(dataset_dir, 'scalers', 'y_scaler.save'),  # Y标准化器文件
            'stats_plot': os.path.join(dataset_dir, f'{dataset_name}_statistics.png')  # 统计图表
        }

    def extract_strain_rate_from_filename(self, filename):
        """从文件名中提取应变率信息"""
        try:
            strain_rate = os.path.splitext(filename)[0].split('_')[2]
            return float(strain_rate)
        except (IndexError, ValueError) as e:
            print(f"警告: 无法从文件名 {filename} 提取应变率信息: {e}")
            return None
                
    def load_and_preprocess_data(self, file_list):
        """
        加载和预处理数据文件
        
        参数:
        file_list: 数据文件路径列表
        
        返回:
        X_paths: 输入序列列表
        Y_paths: 目标序列列表
        """
        self.X_paths = []
        self.Y_paths = []
        self.sequence_lengths = []
        self.temperature_stats = {}
        self.strain_rate_stats = {}  # 重置应变率统计
        
        for file_idx, file in enumerate(file_list):
            print(f"文件 {file_idx+1}/{len(file_list)}: {file} 开始处理")
            try:
                df = pd.read_excel(file)
                df = df.rename(columns=lambda x: x.strip())
                
                # 验证必要的列是否存在
                required_columns = {'time', 'true_strain', 'true_stress', 'temperature'}
                if not required_columns.issubset({col.lower() for col in df.columns}):
                    print(f"文件 {file} 缺少必要列，已跳过")
                    continue

                # 提取数据
                time = df[self.column_mapping['time']]
                true_strain = df[self.column_mapping['true_strain']]
                true_stress = df[self.column_mapping['true_stress']]
                temperature = df[self.column_mapping['temperature']]

                # 处理压缩数据
                if 'com' in file:
                    print(f"文件 '{file}' 检测到压缩数据，已将应力和应变取反")
                    true_strain = -true_strain
                    true_stress = -true_stress

                # 提取温度和应变率信息
                strain_rate = self.extract_strain_rate_from_filename(file)
                
                if temperature is not None:
                    if temperature[0] not in self.temperature_stats:
                        self.temperature_stats[temperature[0]] = 0
                    self.temperature_stats[temperature[0]] += 1
                
                if strain_rate is not None:
                    if strain_rate not in self.strain_rate_stats:
                        self.strain_rate_stats[strain_rate] = 0
                    self.strain_rate_stats[strain_rate] += 1

                # 计算增量
                df[self.column_mapping['delta_strain']] = true_strain.diff().fillna(0)
                df[self.column_mapping['delta_time']] = time.diff().fillna(1e-5)
                df[self.column_mapping['delta_temperature']] = temperature.diff().fillna(0)
                
                # 获取初始值（只取第一个值）
                init_strain = true_strain.iloc[0]
                init_time = time.iloc[0]
                init_temp = temperature.iloc[0]
                
                # 准备特征和目标
                # 增量特征
                delta_features = df[[
                    self.column_mapping['delta_strain'],
                    self.column_mapping['delta_time'],
                    self.column_mapping['delta_temperature']
                ]].values
                
                # 将初始值作为额外特征添加到每个时间步
                init_features = np.array([init_strain, init_time, init_temp])
                x_full = np.column_stack([delta_features, np.tile(init_features, (len(delta_features), 1))])
                y_full = true_stress.values.reshape(-1, 1)
                
                full_len = len(x_full)
                self.sequence_lengths.append(full_len)
                
                # 处理长序列
                if full_len > self.window_size:
                    num_subsequences = 0
                    for i in range(0, full_len - self.window_size + 1, self.stride):
                        if num_subsequences >= self.max_subsequences:
                            break
                        
                        x_sub = x_full[i : i + self.window_size]
                        y_sub = y_full[i : i + self.window_size]
                        
                        self.X_paths.append(x_sub)
                        self.Y_paths.append(y_sub)
                        num_subsequences += 1
                    print(f"文件 {file} (长度 {full_len}) 分割为 {num_subsequences} 个子序列")
                else:
                    # 处理短序列
                    if full_len < self.target_sequence_length:
                        x_full, y_full = self.augment_short_sequence(x_full, y_full)
                    self.X_paths.append(x_full)
                    self.Y_paths.append(y_full)
                    
            except Exception as e:
                print(f"处理文件 {file} 时出错: {e}")
                continue
        
        self._print_dataset_statistics()
        return self.X_paths, self.Y_paths
    
    def augment_short_sequence(self, x, y, target_length=None):
        """
        对短序列进行数据增强
        
        参数:
        x: 输入序列
        y: 目标序列
        target_length: 目标长度，默认使用self.target_sequence_length
        
        返回:
        x_aug: 增强后的输入序列
        y_aug: 增强后的目标序列
        """
        target_length = target_length or self.target_sequence_length
        if len(x) >= target_length:
            return x, y
        
        # 使用插值方法生成更多数据点
        t = np.linspace(0, 1, len(x))
        t_new = np.linspace(0, 1, target_length)
        
        x_aug = np.array([interp1d(t, x[:, i])(t_new) for i in range(x.shape[1])]).T
        y_aug = interp1d(t, y.flatten())(t_new).reshape(-1, 1)
        
        return x_aug, y_aug
    
    def is_normalized(self, data):
        """
        检查数据是否已经标准化
        
        参数:
        data: 输入数据（可以是列表或numpy数组）
        
        返回:
        bool: 数据是否已标准化
        """
        if not self.normalize:
            return True  # 如果不需要归一化，则认为已经"标准化"
            
        if isinstance(data, list):
            data = np.concatenate(data)
        
        data_min = np.min(data, axis=0)
        data_max = np.max(data, axis=0)
        data_range = np.ptp(data, axis=0)
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        
        if self.scaler_type == 'minmax':
            # 检查数据是否在[0,1]范围内
            is_normalized = (np.all(data_min >= -1e-6) and 
                           np.all(data_max <= 1 + 1e-6) and 
                           np.all(data_range >= 0.1))
        elif self.scaler_type == 'standard':
            # 检查是否接近0均值1标准差
            is_normalized = (np.all(np.abs(mean) < 0.1) and 
                           np.all(np.abs(std - 1) < 0.1))
        elif self.scaler_type == 'robust':
            # 对于RobustScaler，检查中位数是否接近0
            median = np.median(data, axis=0)
            is_normalized = np.all(np.abs(median) < 0.1)
        elif self.scaler_type == 'maxabs':
            # 检查最大绝对值是否接近1
            max_abs = np.max(np.abs(data), axis=0)
            is_normalized = np.all(max_abs <= 1 + 1e-6) and np.all(max_abs >= 0.1)
        else:
            # 如果是其他类型，尝试通用检查
            is_minmax = (np.all(data_min >= -1e-6) and 
                        np.all(data_max <= 1 + 1e-6) and 
                        np.all(data_range >= 0.1))
            is_standard = (np.all(np.abs(mean) < 0.1) and 
                          np.all(np.abs(std - 1) < 0.1))
            is_normalized = is_minmax or is_standard
        
        return is_normalized
    
    def prepare_and_save_dataset(self, dataset_name, force_normalize=False):
        """
        准备训练序列并保存数据集，包括标准化和掩码生成
        
        参数:
        dataset_name: 数据集名称
        force_normalize: 是否强制重新标准化数据（即使数据已经标准化）
        
        返回:
        tuple: (X_seq, Y_seq, masks) 标准化后的序列和掩码
        """
        try:
            # 设置数据集名称并获取路径
            self.dataset_name = dataset_name
            paths = self.get_dataset_paths(dataset_name)
            
            # 确保数据集目录存在
            os.makedirs(paths['dataset_dir'], exist_ok=True)
            os.makedirs(paths['scaler_dir'], exist_ok=True)
            
            # 处理归一化
            if not self.normalize:
                print("归一化已禁用，直接使用原始数据...")
                x_scaled = self.X_paths
                y_scaled = self.Y_paths
                print("X数据范围:", np.min(self.X_paths[0]), "到", np.max(self.X_paths[0]))
                print("Y数据范围:", np.min(self.Y_paths[0]), "到", np.max(self.Y_paths[0]))
            else:
                # 检查数据是否已经标准化
                x_is_normalized = self.is_normalized(self.X_paths)
                y_is_normalized = self.is_normalized(self.Y_paths)
                
                if x_is_normalized and y_is_normalized and not force_normalize:
                    print(f"数据已经使用{self.scaler_type}方法标准化，直接使用...")
                    print("X数据范围:", np.min(self.X_paths[0]), "到", np.max(self.X_paths[0]))
                    print("Y数据范围:", np.min(self.Y_paths[0]), "到", np.max(self.Y_paths[0]))
                    x_scaled = self.X_paths
                    y_scaled = self.Y_paths
                else:
                    print(f"数据未标准化或需要重新标准化，开始使用{self.scaler_type}方法标准化...")
                    print("原始数据范围:")
                    print("X数据范围:", np.min(self.X_paths[0]), "到", np.max(self.X_paths[0]))
                    print("Y数据范围:", np.min(self.Y_paths[0]), "到", np.max(self.Y_paths[0]))
                    
                    # 首先将所有数据合并以进行标准化器拟合
                    all_x = np.vstack(self.X_paths)
                    all_y = np.vstack(self.Y_paths)
                    
                    # 拟合标准化器
                    print(f"拟合{self.x_scaler.__class__.__name__}标准化器...")
                    self.x_scaler.fit(all_x)
                    self.y_scaler.fit(all_y)
                    print("标准化器拟合完成")
                    
                    # 标准化数据
                    print("标准化数据...")
                    x_scaled = [self.x_scaler.transform(x) for x in self.X_paths]
                    y_scaled = [self.y_scaler.transform(y) for y in self.Y_paths]
                    print("数据标准化完成")
                    print("标准化后数据范围:")
                    print("X数据范围:", np.min(x_scaled[0]), "到", np.max(x_scaled[0]))
                    print("Y数据范围:", np.min(y_scaled[0]), "到", np.max(y_scaled[0]))
            
            # 准备序列和掩码
            print("准备序列和掩码...")
            X_seq = []
            Y_seq = []
            masks = []
            
            for x, y in zip(x_scaled, y_scaled):
                seq_len = len(x)
                
                # 生成掩码（1表示有效数据，0表示填充）
                mask = np.ones(min(seq_len, self.window_size), dtype=np.float32)
                
                # 填充或截断序列
                x_padded = np.pad(x[:self.window_size], 
                                ((0, max(0, self.window_size - seq_len)), (0, 0)), 
                                mode='constant', constant_values=0)
                y_padded = np.pad(y[:self.window_size], 
                                ((0, max(0, self.window_size - seq_len)), (0, 0)), 
                                mode='constant', constant_values=0)
                mask_padded = np.pad(mask, 
                                   (0, max(0, self.window_size - len(mask))), 
                                   mode='constant', constant_values=0)
                
                X_seq.append(x_padded)
                Y_seq.append(y_padded)
                masks.append(mask_padded)
            
            # 转换为numpy数组
            X_seq = np.array(X_seq, dtype=np.float32)
            Y_seq = np.array(Y_seq, dtype=np.float32)
            masks = np.array(masks, dtype=np.float32)
            
            print("序列和掩码准备完成")
            
            # 保存数据集
            print(f"保存数据集 (格式: {self.output_format})...")
            
            # 保存NPZ格式
            if self.output_format in ['npz', 'both']:
                print("保存NPZ格式...")
                save_data = {
                    'X_paths': np.array(x_scaled, dtype=object),
                    'Y_paths': np.array(y_scaled, dtype=object),
                    'normalize': self.normalize,
                    'scaler_type': self.scaler_type if self.normalize else None,
                    'output_format': self.output_format,
                    'save_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                np.savez_compressed(paths['dataset_file'], **save_data)
                print(f"NPZ数据集已保存: {paths['dataset_file']}")
            
            # 保存TFRecord格式
            if self.output_format in ['tfrecord', 'both']:
                print("保存TFRecord格式...")
                self._save_tfrecord_dataset(x_scaled, y_scaled, paths)
                print(f"TFRecord数据集已保存: {paths['tfrecord_file']}")
            
            # 保存标准化器（如果启用了归一化）
            if self.normalize and self.x_scaler is not None and self.y_scaler is not None:
                print("保存标准化器...")
                joblib.dump(self.x_scaler, paths['x_scaler_file'])
                joblib.dump(self.y_scaler, paths['y_scaler_file'])
                print(f"数据集和{self.scaler_type}标准化器已保存至: {paths['dataset_dir']}")
            else:
                print(f"数据集已保存至: {paths['dataset_dir']} (未使用归一化)")
            
            return X_seq, Y_seq, masks
            
        except Exception as e:
            print(f"准备和保存数据集时出错: {e}")
            return None, None, None

    def load_dataset(self, dataset_name):
        """
        加载数据集和标准化器
        
        参数:
        dataset_name: 数据集名称
        
        返回:
        bool: 是否成功加载
        """
        try:
            # 获取路径
            paths = self.get_dataset_paths(dataset_name)
            
            # 检查数据集文件是否存在
            if not os.path.exists(paths['dataset_file']):
                print(f"数据集文件不存在: {paths['dataset_file']}")
                return False
            
            # 加载数据集
            print(f"加载数据集: {paths['dataset_file']}")
            data = np.load(paths['dataset_file'], allow_pickle=True)
            self.X_paths = data['X_paths'].tolist()
            self.Y_paths = data['Y_paths'].tolist()
            
            # 加载归一化设置（如果存在）
            if 'normalize' in data:
                saved_normalize = bool(data['normalize'])
                saved_scaler_type = str(data['scaler_type']) if data['scaler_type'] is not None else None
                
                print(f"数据集归一化设置: normalize={saved_normalize}, scaler_type={saved_scaler_type}")
                
                # 检查当前设置是否与保存的设置一致
                if self.normalize != saved_normalize:
                    print(f"警告: 当前归一化设置({self.normalize})与数据集设置({saved_normalize})不一致")
                if self.normalize and saved_scaler_type and self.scaler_type != saved_scaler_type:
                    print(f"警告: 当前归一化方法({self.scaler_type})与数据集方法({saved_scaler_type})不一致")
            else:
                print("数据集未包含归一化设置信息，使用当前设置")
            
            # 加载标准化器（如果启用了归一化且文件存在）
            if self.normalize:
                if os.path.exists(paths['x_scaler_file']) and os.path.exists(paths['y_scaler_file']):
                    print("加载标准化器...")
                    self.x_scaler = joblib.load(paths['x_scaler_file'])
                    self.y_scaler = joblib.load(paths['y_scaler_file'])
                    print(f"标准化器加载完成: {self.x_scaler.__class__.__name__}")
                else:
                    print("警告: 启用了归一化但未找到标准化器文件")
                    # 重新创建标准化器
                    self.x_scaler = self._create_scaler()
                    self.y_scaler = self._create_scaler()
            else:
                print("归一化已禁用，跳过标准化器加载")
            
            self.dataset_name = dataset_name
            return True
            
        except Exception as e:
            print(f"加载数据集时出错: {e}")
            return False

    def _print_dataset_statistics(self):
        """打印数据集统计信息"""
        if not self.sequence_lengths:
            self.sequence_lengths = [len(x) for x in self.X_paths]
        
        print("\n数据集统计信息:")
        print("="*50)
        print(f"序列数量: {len(self.X_paths)}")
        print(f"归一化设置: {'启用' if self.normalize else '禁用'}")
        if self.normalize:
            print(f"归一化方法: {self.scaler_type} ({self.x_scaler.__class__.__name__ if self.x_scaler else 'None'})")
        
        # 打印X_paths和Y_paths的维度信息
        if self.X_paths:
            print("\nX_paths前5个序列的前5个元素:")
            for i, x_seq in enumerate(self.X_paths[:5]):
                print(f"序列 {i+1}:")
                print(x_seq[:5])
            
            print("\nY_paths前5个序列的前5个元素:")
            for i, y_seq in enumerate(self.Y_paths[:5]):
                print(f"序列 {i+1}:")
                print(y_seq[:5])
            x_shapes = [x.shape for x in self.X_paths]
            y_shapes = [y.shape for y in self.Y_paths]
            
            print("\nX_paths维度统计:")
            print(f"样本数量: {len(x_shapes)}")
            print(f"形状分布:")
            shape_counts = {}
            for shape in x_shapes:
                shape_str = str(shape)
                if shape_str not in shape_counts:
                    shape_counts[shape_str] = 0
                shape_counts[shape_str] += 1
            for shape_str, count in sorted(shape_counts.items()):
                print(f"  {shape_str}: {count}个样本")
            
            print("\nY_paths维度统计:")
            print(f"样本数量: {len(y_shapes)}")
            print(f"形状分布:")
            shape_counts = {}
            for shape in y_shapes:
                shape_str = str(shape)
                if shape_str not in shape_counts:
                    shape_counts[shape_str] = 0
                shape_counts[shape_str] += 1
            for shape_str, count in sorted(shape_counts.items()):
                print(f"  {shape_str}: {count}个样本")
            
            # 打印特征维度信息
            if x_shapes:
                print(f"\n特征维度: {x_shapes[0][1]}")
                print("特征列表:")
                print("动态特征（每个时间步）:")
                delta_features = [
                    self.column_mapping['delta_strain'],
                    self.column_mapping['delta_time'],
                    self.column_mapping['delta_temperature']
                ]
                for i, col in enumerate(delta_features):
                    print(f"  {i+1}. {col}")
                print("\n静态特征（序列初始值）:")
                init_features = [
                    self.column_mapping['init_strain'],
                    self.column_mapping['init_time'],
                    self.column_mapping['init_temp']
                ]
                for i, col in enumerate(init_features):
                    print(f"  {i+4}. {col}")
        
        print("\n序列长度统计:")
        print(f"最短: {min(self.sequence_lengths)}")
        print(f"最长: {max(self.sequence_lengths)}")
        print(f"平均: {np.mean(self.sequence_lengths):.2f}")
        print(f"中位数: {np.median(self.sequence_lengths)}")
        
        if self.temperature_stats:
            print("\n温度分布:")
            for temp, count in sorted(self.temperature_stats.items()):
                print(f"温度 {temp}°C: {count} 个序列")
        
        if self.strain_rate_stats:
            print("\n应变率分布:")
            for rate, count in sorted(self.strain_rate_stats.items()):
                print(f"应变率 {rate:.2e} s⁻¹: {count} 个序列")
        
        print("="*50)
    
    def plot_dataset_statistics(self, dataset_name=None):
        """
        绘制并保存数据集统计图表
        
        参数:
        dataset_name: 数据集名称，如果为None则使用当前数据集名称
        """
        try:
            if dataset_name is None:
                if self.dataset_name is None:
                    raise ValueError("未指定数据集名称")
                dataset_name = self.dataset_name
            
            # 获取路径
            paths = self.get_dataset_paths(dataset_name)
            
            # 确保目录存在
            os.makedirs(paths['dataset_dir'], exist_ok=True)
            
            # 设置中文字体
            set_chinese_font()
            
            # 创建图形
            fig = plt.figure(figsize=(15, 10))
            
            # 1. 序列长度分布
            plt.subplot(2, 2, 1)
            plt.hist(self.sequence_lengths, bins=30, alpha=0.7)
            plt.title('序列长度分布')
            plt.xlabel('序列长度')
            plt.ylabel('频数')
            plt.grid(True)
            
            # 2. 温度分布
            plt.subplot(2, 2, 2)
            temps = list(self.temperature_stats.keys())
            counts = list(self.temperature_stats.values())
            plt.bar(temps, counts, alpha=0.7)
            plt.title('温度分布')
            plt.xlabel('温度 (°C)')
            plt.ylabel('样本数')
            plt.grid(True)
            
            # 3. 应变率分布
            plt.subplot(2, 2, 3)
            strain_rates = list(self.strain_rate_stats.keys())
            counts = list(self.strain_rate_stats.values())
            plt.bar(strain_rates, counts, alpha=0.7)
            plt.title('应变率分布')
            plt.xlabel('应变率 (s^-1)')
            plt.ylabel('样本数')
            plt.xticks(rotation=45)
            plt.grid(True)
            
            # 4. 应力范围分布 - 由于没有stress_stats，暂时移除这个图
            plt.subplot(2, 2, 4)
            plt.text(0.5, 0.5, '应力范围分布\n(数据未收集)', 
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=plt.gca().transAxes)
            plt.title('应力范围分布')
            plt.axis('off')
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图表
            plt.savefig(paths['stats_plot'], dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"数据集统计图表已保存至: {paths['stats_plot']}")
            
        except Exception as e:
            print(f"绘制统计图表时出错: {e}")

    @staticmethod
    def print_usage_examples():
        """
        打印使用示例和说明文档
        """
        print("\n" + "="*70)
        print("         EMSCDatasetGenerator 使用说明")
        print("="*70)
        
        print("\n1. 归一化选项说明:")
        print("   normalize=True/False  : 是否启用数据归一化")
        print("   scaler_type选项:")
        print("   - 'minmax'   : MinMaxScaler - 将数据缩放到[0,1]范围")
        print("   - 'standard' : StandardScaler - Z-score标准化，均值0，标准差1")
        print("   - 'robust'   : RobustScaler - 基于中位数和四分位数的标准化")
        print("   - 'maxabs'   : MaxAbsScaler - 基于最大绝对值的标准化")
        
        print("\n2. 输出格式选项说明:")
        print("   output_format选项:")
        print("   - 'npz'      : 保存为NPZ格式（默认，兼容性好）")
        print("   - 'tfrecord' : 保存为TFRecord格式（TensorFlow优化，加载更快）")
        print("   - 'both'     : 同时保存NPZ和TFRecord格式")
        
        print("\n3. 使用示例:")
        print("\n   # 示例1: 生成NPZ格式数据集（默认）")
        print("   generator = EMSCDatasetGenerator(")
        print("       target_sequence_length=1000,")
        print("       normalize=True,")
        print("       scaler_type='minmax',")
        print("       output_format='npz'")
        print("   )")
        
        print("\n   # 示例2: 生成TFRecord格式数据集（推荐用于大数据集）")
        print("   generator = EMSCDatasetGenerator(")
        print("       target_sequence_length=1000,")
        print("       normalize=True,")
        print("       scaler_type='minmax',")
        print("       output_format='tfrecord'")
        print("   )")
        
        print("\n   # 示例3: 同时生成两种格式")
        print("   generator = EMSCDatasetGenerator(")
        print("       target_sequence_length=1000,")
        print("       normalize=True,")
        print("       scaler_type='minmax',")
        print("       output_format='both'")
        print("   )")
        
        print("\n   # 示例4: 禁用归一化，生成TFRecord")
        print("   generator = EMSCDatasetGenerator(")
        print("       target_sequence_length=1000,")
        print("       normalize=False,")
        print("       output_format='tfrecord'")
        print("   )")
        
        print("\n4. 格式选择建议:")
        print("   - 小数据集(<100MB): 推荐使用 'npz'")
        print("   - 大数据集(>100MB): 推荐使用 'tfrecord'")
        print("   - 需要兼容性: 推荐使用 'both'")
        print("   - TensorFlow训练: 推荐使用 'tfrecord'")
        
        print("\n5. 归一化方法选择建议:")
        print("   - 神经网络模型: 推荐使用 'minmax'")
        print("   - 数据包含异常值: 推荐使用 'robust'")
        print("   - 传统机器学习: 推荐使用 'standard'")
        print("   - 数据已预处理: 可选择 normalize=False")
        
        print("\n6. 注意事项:")
        print("   - 数据集会保存归一化和格式设置")
        print("   - 标准化器会自动保存和加载")
        print("   - TFRecord格式加载速度更快，但文件较大")
        print("   - 可使用 force_normalize=True 强制重新归一化")
        print("   - 使用 get_scaler_info() 查看当前设置")
        
        print("="*70)

    def recover_physical_quantities(self, X_paths):
        """
        从增量和初值恢复全物理量序列
        
        Args:
            X_paths: 输入序列列表，每个序列包含 [delta_strain, delta_time, delta_temperature, 
                    init_strain, init_time, init_temp]
        
        Returns:
            physical_sequences: 恢复后的物理量序列列表，每个序列包含 [strain, time, temperature]
        """
        physical_sequences = []
        
        for x_seq in X_paths:
            # 提取增量和初值
            delta_strain = x_seq[:, 0]
            delta_time = x_seq[:, 1]
            delta_temperature = x_seq[:, 2]
            init_strain = x_seq[0, 3]  # 使用第一个点的初值
            init_time = x_seq[0, 4]
            init_temp = x_seq[0, 5]
            
            # 计算累积值
            strain = np.cumsum(delta_strain) + init_strain
            time = np.cumsum(delta_time) + init_time
            temperature = np.cumsum(delta_temperature) + init_temp
            
            # 组合成物理量序列
            physical_seq = np.column_stack([strain, time, temperature])
            physical_sequences.append(physical_seq)
        
        return physical_sequences

    def normalize_window_features(self, window_X, window_Y, x_scaler, y_scaler):
        """
        使用全局归一化参数对窗口特征进行归一化
        
        Args:
            window_X: 窗口输入数据 (window_size, 6)
            window_Y: 窗口输出数据 (window_size, 1)
            x_scaler: X数据的标准化器
            y_scaler: Y数据的标准化器
        
        Returns:
            normalized_X: 归一化后的窗口输入数据
            normalized_Y: 归一化后的窗口输出数据
        """
        # 分离动态特征和静态特征
        dynamic_features = window_X[:, :3]  # delta_strain, delta_time, delta_temperature
        static_features = window_X[:, 3:]   # init_strain, init_time, init_temp
        
        # 对动态特征进行归一化
        normalized_dynamic = x_scaler.transform(dynamic_features)
        
        # 对静态特征进行归一化（使用相同的scaler，但只取第一个点）
        normalized_static = x_scaler.transform(static_features[0:1])
        normalized_static = np.tile(normalized_static, (len(window_X), 1))
        
        # 组合归一化后的特征
        normalized_X = np.column_stack([normalized_dynamic, normalized_static])
        
        # 对输出进行归一化
        normalized_Y = y_scaler.transform(window_Y)
        
        return normalized_X, normalized_Y

    def prepare_window_data(self, physical_sequences, window_size, stride=None, 
                          x_scaler=None, y_scaler=None):
        """
        从物理量序列生成训练窗口数据
        
        Args:
            physical_sequences: 物理量序列列表
            window_size: 窗口大小
            stride: 滑动步长，默认为window_size//10
            x_scaler: X数据的标准化器
            y_scaler: Y数据的标准化器
        
        Returns:
            windows_X: 窗口输入数据列表
            windows_Y: 窗口输出数据列表
        """
        if stride is None:
            stride = window_size // 10
            
        windows_X = []
        windows_Y = []
        
        for seq_idx, physical_seq in enumerate(physical_sequences):
            seq_len = len(physical_seq)
            
            # 对每个可能的起始位置
            for start_idx in range(0, seq_len - window_size + 1, stride):
                end_idx = start_idx + window_size
                
                # 提取窗口的物理量数据
                window_physical = physical_seq[start_idx:end_idx]
                
                # 计算窗口内的增量
                delta_strain = np.zeros(window_size)
                delta_time = np.zeros(window_size)
                delta_temperature = np.zeros(window_size)
                
                for i in range(1, window_size):
                    delta_strain[i] = window_physical[i, 0] - window_physical[i-1, 0]
                    delta_time[i] = window_physical[i, 1] - window_physical[i-1, 1]
                    delta_temperature[i] = window_physical[i, 2] - window_physical[i-1, 2]
                
                # 获取窗口起始点的值作为初始值
                init_strain = window_physical[0, 0]
                init_time = window_physical[0, 1]
                init_temp = window_physical[0, 2]
                
                # 构造窗口特征
                window_X = np.column_stack([
                    delta_strain,
                    delta_time,
                    delta_temperature,
                    np.full(window_size, init_strain),
                    np.full(window_size, init_time),
                    np.full(window_size, init_temp)
                ])
                
                # 如果有标准化器，进行归一化
                if x_scaler is not None and y_scaler is not None:
                    window_X, window_Y = self.normalize_window_features(
                        window_X, window_Y, x_scaler, y_scaler)
                
                windows_X.append(window_X)
                windows_Y.append(window_Y)
        
        return windows_X, windows_Y

    def get_scalers(self, dataset_name):
        """
        加载数据集对应的标准化器
        
        Args:
            dataset_name: 数据集名称
        
        Returns:
            x_scaler: X数据的标准化器
            y_scaler: Y数据的标准化器
        """
        paths = self.get_dataset_paths(dataset_name)
        
        if os.path.exists(paths['x_scaler_file']) and os.path.exists(paths['y_scaler_file']):
            x_scaler = joblib.load(paths['x_scaler_file'])
            y_scaler = joblib.load(paths['y_scaler_file'])
            return x_scaler, y_scaler
        else:
            print("警告: 未找到标准化器文件")
            return None, None

    def _bytes_feature(self, value):
        """将numpy数组转换为bytes特征"""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.tobytes()]))

    def _float_feature(self, value):
        """将float值转换为float特征"""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def _int64_feature(self, value):
        """将int值转换为int64特征"""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _save_tfrecord_dataset(self, x_scaled, y_scaled, paths):
        """
        保存TFRecord格式的数据集
        
        参数:
        x_scaled: 标准化后的X数据列表
        y_scaled: 标准化后的Y数据列表
        paths: 路径字典
        """
        start_time = time.time()
        total_samples = len(x_scaled)
        
        print(f"开始转换 {total_samples} 个样本到TFRecord格式...")
        
        # 创建TFRecord写入器
        with tf.io.TFRecordWriter(paths['tfrecord_file']) as writer:
            for i, (x_sample, y_sample) in enumerate(tqdm(zip(x_scaled, y_scaled), 
                                                          total=total_samples, 
                                                          desc="转换TFRecord")):
                # 确保数据是numpy数组并转换为float32
                x_array = np.array(x_sample, dtype=np.float32)
                y_array = np.array(y_sample, dtype=np.float32)
                
                # 创建特征
                feature = {
                    'X_paths': self._bytes_feature(x_array),
                    'Y_paths': self._bytes_feature(y_array),
                    'sample_id': self._int64_feature(i)
                }
                
                # 创建Example并写入
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
        
        # 创建TFRecord信息文件 - 与现有格式兼容
        if x_scaled and y_scaled:
            sample_x = np.array(x_scaled[0], dtype=np.float32)
            sample_y = np.array(y_scaled[0], dtype=np.float32)
            
            # 计算展平后的大小
            x_flattened_size = sample_x.size
            y_flattened_size = sample_y.size
            
            dataset_info = {
                'keys': ['X_paths', 'Y_paths'],
                'total_samples': total_samples,
                'shapes': {
                    'X_paths': [x_flattened_size],  # 展平后的大小
                    'Y_paths': [y_flattened_size]   # 展平后的大小
                },
                'dtypes': {
                    'X_paths': 'float32',
                    'Y_paths': 'float32'
                },
                'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'source_file': paths['dataset_file'],
                'note': 'TFRecord格式数据集 - 由EMSCDatasetGenerator生成',
                'normalize': self.normalize,
                'scaler_type': self.scaler_type if self.normalize else None,
                'output_format': self.output_format,
                'conversion_time': time.time() - start_time,
                'conversion_method': 'dataset_generator_tfrecord',
                'original_shapes': {
                    'X_paths': {
                        'outer_shape': [total_samples] + list(sample_x.shape),
                        'inner_shape': list(sample_x.shape),
                        'note': f'嵌套数组: 外层({total_samples}, {sample_x.shape[0]}, {sample_x.shape[1]}), 内层{sample_x.shape}'
                    },
                    'Y_paths': {
                        'outer_shape': [total_samples] + list(sample_y.shape),
                        'inner_shape': list(sample_y.shape),
                        'note': f'嵌套数组: 外层({total_samples}, {sample_y.shape[0]}, {sample_y.shape[1]}), 内层{sample_y.shape}'
                    }
                }
            }
        else:
            # 备用信息
            dataset_info = {
                'keys': ['X_paths', 'Y_paths'],
                'total_samples': total_samples,
                'shapes': {'X_paths': 'unknown', 'Y_paths': 'unknown'},
                'dtypes': {'X_paths': 'float32', 'Y_paths': 'float32'},
                'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'note': 'TFRecord格式数据集 - 由EMSCDatasetGenerator生成',
                'conversion_time': time.time() - start_time,
                'conversion_method': 'dataset_generator_tfrecord'
            }
        
        # 保存信息文件
        with open(paths['tfrecord_info'], 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        elapsed_time = time.time() - start_time
        conversion_rate = total_samples / elapsed_time if elapsed_time > 0 else 0
        
        print(f"TFRecord转换完成:")
        print(f"  ⏱️ 耗时: {elapsed_time:.2f}秒")
        print(f"  📈 转换速度: {conversion_rate:.1f} 样本/秒")
        print(f"  📋 信息文件: {paths['tfrecord_info']}")

def main():
    """主函数，用于测试数据集生成器"""
    # 打印使用说明
    EMSCDatasetGenerator.print_usage_examples()
    
    # 示例用法
    data_dir = "/Users/tianyunhu/Documents/temp/CTC/PPCC"  # 替换为实际的数据目录
    save_dir = "/Users/tianyunhu/Documents/temp/code/Test_app/EMSC_Model/msc_models"  # 替换为实际的保存目录
    dataset_name = 'dataset_EMSC_big'

    # 获取数据文件列表
    file_list = glob.glob(os.path.join(data_dir, "*.xlsx"))
    
    if not file_list:
        print(f"在 {data_dir} 中未找到数据文件")
        return
    
    # 创建数据集生成器 - 修改为匹配训练脚本期望的格式
    # 获取第一个Excel文件的长度作为目标序列长度
    first_file = file_list[0]
    df = pd.read_excel(first_file)
    target_sequence_length = len(df)
    window_size = target_sequence_length
    stride = 100  # 减小步长以生成更多重叠序列
    max_subsequences = 50  # 减少每个文件的子序列数量
    
    # 示例1: 生成TFRecord格式数据集（推荐用于大数据集）
    generator = EMSCDatasetGenerator(
        target_sequence_length=target_sequence_length,
        window_size=window_size,
        stride=stride,
        max_subsequences=max_subsequences,
        normalize=True,          # 启用归一化
        scaler_type='minmax',     # 使用MinMax归一化
        output_format='both'      # 同时生成NPZ和TFRecord以确保兼容性
    )
    
    # 示例2: 生成NPZ格式数据集（兼容性好）
    # generator = EMSCDatasetGenerator(
    #     target_sequence_length=target_sequence_length,
    #     window_size=window_size,
    #     stride=stride,
    #     max_subsequences=max_subsequences,
    #     normalize=True,
    #     scaler_type='minmax',
    #     output_format='npz'      # 输出格式为NPZ
    # )
    
    # 示例3: 同时生成两种格式
    # generator = EMSCDatasetGenerator(
    #     target_sequence_length=target_sequence_length,
    #     window_size=window_size,
    #     stride=stride,
    #     max_subsequences=max_subsequences,
    #     normalize=True,
    #     scaler_type='minmax',
    #     output_format='both'     # 同时生成NPZ和TFRecord
    # )
    
    # 示例4: 禁用归一化，生成TFRecord
    # generator = EMSCDatasetGenerator(
    #     target_sequence_length=target_sequence_length,
    #     window_size=window_size,
    #     stride=stride,
    #     max_subsequences=max_subsequences,
    #     normalize=False,         # 禁用归一化
    #     output_format='tfrecord' # 输出格式为TFRecord
    # )
    
    # 生成数据集
    X_paths, Y_paths = generator.load_and_preprocess_data(file_list)
    
    # 打印当前归一化设置信息
    print("\n当前归一化设置:", generator.get_scaler_info())
    
    # 准备训练序列  
    dataset_path = os.path.join(save_dir, dataset_name + '.npz')
    X_seq, Y_seq, masks = generator.prepare_and_save_dataset(dataset_name)
    
    # 绘制统计图表
    generator.plot_dataset_statistics(dataset_name)
    
    # 演示如何测试不同的归一化方法
    # print("\n" + "="*50)
    # print("演示不同归一化方法的效果:")
    # print("="*50)
    
    # if X_paths and Y_paths:
    #     sample_x = X_paths[0][:100]  # 取第一个序列的前100个点作为示例
    #     sample_y = Y_paths[0][:100]
        
    #     scaler_types = ['minmax', 'standard', 'robust', 'maxabs']
        
    #     for scaler_type in scaler_types:
    #         print(f"\n{scaler_type.upper()}标准化效果:")
    #         temp_generator = EMSCDatasetGenerator(normalize=True, scaler_type=scaler_type)
            
    #         # 创建临时标准化器并拟合
    #         temp_x_scaler = temp_generator._create_scaler()
    #         temp_y_scaler = temp_generator._create_scaler()
            
    #         temp_x_scaler.fit(sample_x)
    #         temp_y_scaler.fit(sample_y)
            
    #         # 转换数据
    #         scaled_x = temp_x_scaler.transform(sample_x)
    #         scaled_y = temp_y_scaler.transform(sample_y)
            
    #         print(f"  X - 原始范围: [{sample_x.min():.3f}, {sample_x.max():.3f}] -> 标准化后: [{scaled_x.min():.3f}, {scaled_x.max():.3f}]")
    #         print(f"  Y - 原始范围: [{sample_y.min():.3f}, {sample_y.max():.3f}] -> 标准化后: [{scaled_y.min():.3f}, {scaled_y.max():.3f}]")
            
    #         if scaler_type == 'standard':
    #             print(f"      X均值: {scaled_x.mean():.3f}, 标准差: {scaled_x.std():.3f}")
    #             print(f"      Y均值: {scaled_y.mean():.3f}, 标准差: {scaled_y.std():.3f}")


if __name__ == "__main__":
    main()

