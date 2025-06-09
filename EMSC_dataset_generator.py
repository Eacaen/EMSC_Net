import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
from datetime import datetime
import glob
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import matplotlib as mpl

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
    """
    def __init__(self, target_sequence_length=1000, window_size=None, stride=None, max_subsequences=200):
        """
        初始化数据集生成器
        
        参数:
        target_sequence_length: 目标序列长度
        window_size: 滑动窗口大小，默认为target_sequence_length
        stride: 滑动窗口步长，默认为target_sequence_length//10
        max_subsequences: 每个序列最多生成的子序列数
        """
        self.target_sequence_length = target_sequence_length
        self.window_size = window_size if window_size is not None else target_sequence_length
        self.stride = stride if stride is not None else target_sequence_length // 10
        self.max_subsequences = max_subsequences
        
        # 初始化标准化器
        self.x_scaler = MinMaxScaler()
        self.y_scaler = MinMaxScaler()
        
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
            'dataset_file': os.path.join(dataset_dir, f'{dataset_name}.npz'),  # 数据集文件
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
        if isinstance(data, list):
            data = np.concatenate(data)
        # 检查数据是否在[0,1]范围内（MinMaxScaler的默认范围）
        data_range = np.ptp(data, axis=0)  # 计算每个特征的范围
        data_min = np.min(data, axis=0)    # 计算每个特征的最小值
        data_max = np.max(data, axis=0)    # 计算每个特征的最大值
        
        # 检查是否所有特征都在[0,1]范围内，且范围接近1
        is_minmax = np.all(data_min >= -1e-6) and np.all(data_max <= 1 + 1e-6) and np.all(data_range >= 0.1)
        
        # 检查是否所有特征都接近0均值（StandardScaler的特征）
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        is_standard = np.all(np.abs(mean) < 1e-6) and np.all(np.abs(std - 1) < 1e-6)
        
        return is_minmax or is_standard
    
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
            
            # 检查数据是否已经标准化
            x_is_normalized = self.is_normalized(self.X_paths)
            y_is_normalized = self.is_normalized(self.Y_paths)
            
            if x_is_normalized and y_is_normalized and not force_normalize:
                print("数据已经标准化，直接使用...")
                print("X数据范围:", np.min(self.X_paths[0]), "到", np.max(self.X_paths[0]))
                print("Y数据范围:", np.min(self.Y_paths[0]), "到", np.max(self.Y_paths[0]))
                x_scaled = self.X_paths
                y_scaled = self.Y_paths
            else:
                print("数据未标准化或需要重新标准化，开始标准化过程...")
                print("X数据范围:", np.min(self.X_paths[0]), "到", np.max(self.X_paths[0]))
                print("Y数据范围:", np.min(self.Y_paths[0]), "到", np.max(self.Y_paths[0]))
                
                # 首先将所有数据合并以进行标准化器拟合
                all_x = np.vstack(self.X_paths)
                all_y = np.vstack(self.Y_paths)
                
                # 拟合标准化器
                print("拟合标准化器...")
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
            
            # 保存数据集和标准化器
            print("保存数据集和标准化器...")
            np.savez_compressed(
                paths['dataset_file'],
                X_paths=np.array(x_scaled, dtype=object),
                Y_paths=np.array(y_scaled, dtype=object),
                save_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            
            # 保存标准化器
            joblib.dump(self.x_scaler, paths['x_scaler_file'])
            joblib.dump(self.y_scaler, paths['y_scaler_file'])
            
            print(f"数据集和标准化器已保存至: {paths['dataset_dir']}")
            
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
            
            # 加载标准化器
            if os.path.exists(paths['x_scaler_file']) and os.path.exists(paths['y_scaler_file']):
                print("加载标准化器...")
                self.x_scaler = joblib.load(paths['x_scaler_file'])
                self.y_scaler = joblib.load(paths['y_scaler_file'])
                print("标准化器加载完成")
            else:
                print("警告: 未找到标准化器文件")
            
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

def main():
    """主函数，用于测试数据集生成器"""
    # 示例用法
    data_dir = "/Users/tianyunhu/Documents/temp/CTC/PPCC"  # 替换为实际的数据目录
    save_dir = "/Users/tianyunhu/Documents/temp/code/Test_app/EMSC_Model/msc_models"  # 替换为实际的保存目录
    dataset_name = 'dataset_EMSC_big'

    # 获取数据文件列表
    file_list = glob.glob(os.path.join(data_dir, "*.xlsx"))
    
    if not file_list:
        print(f"在 {data_dir} 中未找到数据文件")
        return
    
    # 创建数据集生成器
    target_sequence_length = 1000
    window_size = 1000
    stride = 500
    max_subsequences = 200
    generator = EMSCDatasetGenerator(
        target_sequence_length=target_sequence_length,
        window_size=window_size,
        stride=stride,
        max_subsequences=max_subsequences
    )
    
    # 生成数据集
    X_paths, Y_paths = generator.load_and_preprocess_data(file_list)
    
    # 准备训练序列  
    dataset_path = os.path.join(save_dir, dataset_name + '.npz')
    X_seq, Y_seq, masks = generator.prepare_and_save_dataset(dataset_name)
    
    # 绘制统计图表
    generator.plot_dataset_statistics(dataset_name)
if __name__ == "__main__":
    main()

