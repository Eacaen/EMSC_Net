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
    def __init__(self, 
                 target_sequence_length=5000,
                 window_size=None,
                 stride=None,
                 max_subsequences=200,
                 random_seed=42):
        """
        初始化数据集生成器
        
        参数:
        target_sequence_length: 目标序列长度
        window_size: 滑动窗口大小，默认等于target_sequence_length
        stride: 滑动窗口步长，默认等于target_sequence_length/10
        max_subsequences: 每条长路径最多截取的子序列数
        random_seed: 随机种子
        """
        # 设置中文字体
        set_chinese_font()
        
        self.target_sequence_length = target_sequence_length
        self.window_size = window_size or target_sequence_length
        self.stride = stride or target_sequence_length // 10
        self.max_subsequences = max_subsequences
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # 初始化标准化器
        self.x_scaler = MinMaxScaler()
        self.y_scaler = MinMaxScaler()
        
        # 数据存储
        self.X_paths = []
        self.Y_paths = []
        self.sequence_lengths = []
        self.temperature_stats = {}
        self.strain_rate_stats = {}  # 新增：应变率统计
        
        # 定义列名映射
        self.column_mapping = {
            'Time': 'time',
            'True_Strain': 'true_strain',
            'True_Stress': 'true_stress',
            'delta_strain': 'delta_strain',  # 原 Δε
            'delta_time': 'delta_time',      # 原 Δt
            'Temperature': 'temperature'     # 原 T
        }
        
    def extract_temperature_from_filename(self, filename):
        """从文件名中提取温度信息"""
        try:
            temperature = os.path.splitext(filename)[0].split('_')[3]
            return float(temperature)
        except (IndexError, ValueError) as e:
            print(f"警告: 无法从文件名 {filename} 提取温度信息: {e}")
            return None

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
            try:
                df = pd.read_excel(file)
                df = df.rename(columns=lambda x: x.strip())
                
                # 验证必要的列是否存在
                required_columns = {'time', 'true_strain', 'true_stress'}
                if not required_columns.issubset({col.lower() for col in df.columns}):
                    print(f"文件 {file} 缺少必要列，已跳过")
                    continue

                # 提取数据
                true_strain = df[self.column_mapping['True_Strain']]
                true_stress = df[self.column_mapping['True_Stress']]

                # 处理压缩数据
                if 'com' in file:
                    print(f"文件 '{file}' 检测到压缩数据，已将应力和应变取反")
                    true_strain = -true_strain
                    true_stress = -true_stress

                # 提取温度和应变率信息
                temperature = self.extract_temperature_from_filename(file)
                strain_rate = self.extract_strain_rate_from_filename(file)
                
                if temperature is not None:
                    if temperature not in self.temperature_stats:
                        self.temperature_stats[temperature] = 0
                    self.temperature_stats[temperature] += 1
                
                if strain_rate is not None:
                    if strain_rate not in self.strain_rate_stats:
                        self.strain_rate_stats[strain_rate] = 0
                    self.strain_rate_stats[strain_rate] += 1

                # 计算增量
                df[self.column_mapping['delta_strain']] = true_strain.diff().fillna(0)
                df[self.column_mapping['delta_time']] = df[self.column_mapping['Time']].diff().fillna(1e-5)
                df[self.column_mapping['Temperature']] = temperature
                
                # 准备特征和目标
                feature_columns = [
                    self.column_mapping['delta_strain'],
                    self.column_mapping['delta_time'],
                    self.column_mapping['Temperature']
                ]
                x_full = df[feature_columns].values
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
    
    def prepare_sequences(self):
        """
        准备训练序列，包括标准化和掩码生成
        
        返回:
        X_seq: 标准化后的输入序列
        Y_seq: 标准化后的目标序列
        masks: 序列掩码
        """
        # 首先将所有数据合并以进行标准化器拟合
        all_x = np.vstack(self.X_paths)
        all_y = np.vstack(self.Y_paths)
        
        # 拟合标准化器
        print("拟合标准化器...")
        self.x_scaler.fit(all_x)
        self.y_scaler.fit(all_y)
        print("标准化器拟合完成")
        
        # 标准化数据
        print("开始标准化数据...")
        x_scaled = [self.x_scaler.transform(x) for x in self.X_paths]
        y_scaled = [self.y_scaler.transform(y) for y in self.Y_paths]
        print("数据标准化完成")
        
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
        
        print("序列和掩码准备完成")
        return np.array(X_seq, dtype=np.float32), np.array(Y_seq, dtype=np.float32), np.array(masks, dtype=np.float32)
    
    def save_dataset(self, save_path):
        """
        保存数据集和标准化器
        
        参数:
        save_path: 保存路径
        
        返回:
        bool: 是否保存成功
        """
        try:
            # 确保保存目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # 保存数据集
            np.savez_compressed(
                save_path,
                X_paths=np.array(self.X_paths, dtype=object),
                Y_paths=np.array(self.Y_paths, dtype=object),
                save_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            
            # 保存标准化器
            scaler_path = os.path.join(os.path.dirname(save_path), 'scalers')
            os.makedirs(scaler_path, exist_ok=True)
            joblib.dump(self.x_scaler, os.path.join(scaler_path, 'x_scaler.save'))
            joblib.dump(self.y_scaler, os.path.join(scaler_path, 'y_scaler.save'))
            
            print(f"数据集和标准化器已保存至: {save_path}")
            return True
            
        except Exception as e:
            print(f"保存数据集时出错: {e}")
            return False
    
    def load_dataset(self, load_path):
        """
        加载数据集和标准化器
        
        参数:
        load_path: 加载路径
        
        返回:
        bool: 是否加载成功
        """
        try:
            if not os.path.exists(load_path):
                print(f"数据集文件不存在: {load_path}")
                return False
            
            # 加载数据集
            data = np.load(load_path, allow_pickle=True)
            self.X_paths = data['X_paths'].tolist()
            self.Y_paths = data['Y_paths'].tolist()
            
            # 加载标准化器
            scaler_path = os.path.join(os.path.dirname(load_path), 'scalers')
            self.x_scaler = joblib.load(os.path.join(scaler_path, 'x_scaler.save'))
            self.y_scaler = joblib.load(os.path.join(scaler_path, 'y_scaler.save'))
            
            self._print_dataset_statistics()
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
    
    def plot_dataset_statistics(self, save_path=None):
        """
        绘制数据集统计图表
        
        参数:
        save_path: 图表保存路径，如果为None则显示图表
        """
        # 确保中文字体设置
        set_chinese_font()
        
        plt.figure(figsize=(15, 15))  # 增加图形高度以容纳新的子图
        
        # 序列长度分布
        plt.subplot(3, 2, 1)
        plt.hist(self.sequence_lengths, bins=30, color='skyblue', edgecolor='black')
        plt.title("序列长度分布", fontsize=12, pad=15)
        plt.xlabel("序列长度", fontsize=10)
        plt.ylabel("样本数量", fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 温度分布
        if self.temperature_stats:
            plt.subplot(3, 2, 2)
            temps = list(self.temperature_stats.keys())
            counts = list(self.temperature_stats.values())
            bars = plt.bar(temps, counts, color='lightgreen', edgecolor='black')
            plt.title("温度分布", fontsize=12, pad=15)
            plt.xlabel("温度 (°C)", fontsize=10)
            plt.ylabel("序列数量", fontsize=10)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # 在柱状图上添加数值标签
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom')
        
        # 应变率分布（新增）
        if self.strain_rate_stats:
            plt.subplot(3, 2, 3)
            rates = list(self.strain_rate_stats.keys())
            counts = list(self.strain_rate_stats.values())
            
            # 使用对数刻度显示应变率
            plt.bar(range(len(rates)), counts, color='lightcoral', edgecolor='black')
            plt.title("应变率分布", fontsize=12, pad=15)
            plt.xlabel("应变率 (s⁻¹)", fontsize=10)
            plt.ylabel("序列数量", fontsize=10)
            plt.xticks(range(len(rates)), [f'{rate:.2e}' for rate in rates], rotation=45)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # 在柱状图上添加数值标签
            for i, count in enumerate(counts):
                plt.text(i, count, f'{int(count)}', ha='center', va='bottom')
        
        # 应力-应变关系示例（随机选择几个序列）
        plt.subplot(3, 2, 4)
        num_samples = min(5, len(self.X_paths))
        sample_indices = np.random.choice(len(self.X_paths), num_samples, replace=False)
        
        colors = plt.cm.tab10(np.linspace(0, 1, num_samples))
        for idx, color in zip(sample_indices, colors):
            strain = np.cumsum(self.X_paths[idx][:, 0])
            stress = self.Y_paths[idx].flatten()
            plt.plot(strain, stress, label=f'样本 {idx+1}', color=color, linewidth=2)
        
        plt.title("应力-应变关系示例", fontsize=12, pad=15)
        plt.xlabel("应变", fontsize=10)
        plt.ylabel("应力 (MPa)", fontsize=10)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 温度和应变率组合分布（新增）
        if self.temperature_stats and self.strain_rate_stats:
            plt.subplot(3, 2, 5)
            
            # 创建温度和应变率的组合统计
            temp_rate_stats = {}
            for file_idx, file in enumerate(self.X_paths):
                temp = list(self.temperature_stats.keys())[file_idx % len(self.temperature_stats)]
                rate = list(self.strain_rate_stats.keys())[file_idx % len(self.strain_rate_stats)]
                key = (temp, rate)
                if key not in temp_rate_stats:
                    temp_rate_stats[key] = 0
                temp_rate_stats[key] += 1
            
            # 准备热力图数据
            temps = sorted(self.temperature_stats.keys())
            rates = sorted(self.strain_rate_stats.keys())
            heatmap_data = np.zeros((len(temps), len(rates)))
            
            for (temp, rate), count in temp_rate_stats.items():
                i = temps.index(temp)
                j = rates.index(rate)
                heatmap_data[i, j] = count
            
            # 绘制热力图
            plt.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
            plt.colorbar(label='序列数量')
            plt.title("温度-应变率分布", fontsize=12, pad=15)
            plt.xlabel("应变率 (s⁻¹)", fontsize=10)
            plt.ylabel("温度 (°C)", fontsize=10)
            
            # 设置刻度标签
            plt.xticks(range(len(rates)), [f'{rate:.2e}' for rate in rates], rotation=45)
            plt.yticks(range(len(temps)), [f'{temp:.0f}' for temp in temps])
            
            # 在热力图上添加数值标签
            for i in range(len(temps)):
                for j in range(len(rates)):
                    if heatmap_data[i, j] > 0:
                        plt.text(j, i, f'{int(heatmap_data[i, j])}',
                                ha='center', va='center', color='black')
        
        # 添加数据集基本信息
        plt.subplot(3, 2, 6)
        plt.axis('off')
        info_text = (
            f"数据集统计信息:\n\n"
            f"总序列数: {len(self.X_paths)}\n"
            f"最短序列: {min(self.sequence_lengths)}\n"
            f"最长序列: {max(self.sequence_lengths)}\n"
            f"平均长度: {np.mean(self.sequence_lengths):.1f}\n"
            f"中位长度: {np.median(self.sequence_lengths):.1f}\n"
            f"温度范围: {min(self.temperature_stats.keys()):.1f}°C - "
            f"{max(self.temperature_stats.keys()):.1f}°C\n"
            f"应变率范围: {min(self.strain_rate_stats.keys()):.2e} - "
            f"{max(self.strain_rate_stats.keys()):.2e} s⁻¹"
        )
        plt.text(0.1, 0.5, info_text, fontsize=10, 
                bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"统计图表已保存至: {save_path}")
            plt.close()
        else:
            plt.show()

def main():
    """主函数，用于测试数据集生成器"""
    # 示例用法
    data_dir = "/Users/tianyunhu/Documents/temp/CTC/PPCC"  # 替换为实际的数据目录
    save_dir = "/Users/tianyunhu/Documents/temp/code/Test_app/EMSC_Model/msc_models"  # 替换为实际的保存目录
    
    # 获取数据文件列表
    file_list = glob.glob(os.path.join(data_dir, "*.xlsx"))
    
    if not file_list:
        print(f"在 {data_dir} 中未找到数据文件")
        return
    
    # 创建数据集生成器
    generator = EMSCDatasetGenerator(
        target_sequence_length=5000,
        window_size=5000,
        stride=500,
        max_subsequences=200
    )
    
    # 生成数据集
    X_paths, Y_paths = generator.load_and_preprocess_data(file_list)
    
    # 准备训练序列
    X_seq, Y_seq, masks = generator.prepare_sequences()
    
    # 保存数据集
    save_path = os.path.join(save_dir, "dataset.npz")
    generator.save_dataset(save_path)
    
    # 绘制统计图表
    plot_path = os.path.join(save_dir, "dataset_statistics.png")
    generator.plot_dataset_statistics(plot_path)

if __name__ == "__main__":
    main()

