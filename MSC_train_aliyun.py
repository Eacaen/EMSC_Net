# Minimal State Cell (MSC) training pipeline with multi-temperature, multi-rate experimental data

import os
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Concatenate, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import History, Callback
import tensorflow as tf
import matplotlib.pyplot as plt
import joblib
from scipy.interpolate import interp1d
import shutil
import glob
import h5py
from datetime import datetime
import time

print(f"当前工作目录: {os.getcwd()}")
print(tf.__version__)

# 状态维度：定义MSC单元内部状态向量的维度
# 这个参数决定了模型内部状态的复杂度，影响模型捕捉材料行为的能力
# 较大的值可以表示更复杂的材料状态，但会增加模型参数量和训练难度
STATE_DIM = 5

# 输入维度：定义模型输入特征的维度
# 对于材料本构模型，通常包括应变增量、应变率、温度等物理量
# 这个参数需要与实验数据的特征数量相匹配
INPUT_DIM = 3

# 隐藏层维度：定义神经网络隐藏层的神经元数量
# 这个参数影响模型的表达能力和复杂度
# 较大的值可以学习更复杂的映射关系，但可能导致过拟合和计算开销增加
HIDDEN_DIM = 32

# 学习率：定义优化算法的学习率
# 这个参数控制模型参数更新的步长，影响训练速度和稳定性
LEARNING_RATE = 1e-3

TARGET_SEQUENCE_LENGTH = 500  # 更新目标序列长度

# 添加滑动窗口参数
WINDOW_SIZE = TARGET_SEQUENCE_LENGTH
# 滑动窗口步长：定义生成子序列时的滑动步长
# 这个参数控制相邻子序列之间的重叠程度，影响训练数据的多样性和数量
# 较小的步长会产生更多重叠的子序列，增加训练数据量，但可能导致数据冗余
# 较大的步长会减少重叠，降低数据冗余，但可能减少训练样本数量
STRIDE = TARGET_SEQUENCE_LENGTH // 10  # 设置为目标序列长度的一半，平衡数据量和多样性
MAX_SUBSEQUENCES = 200  # 每条长路径最多截取的子序列数

class MSC_Cell(tf.keras.layers.Layer):
    def __init__(self, state_dim=5, hidden_dim=32):
        super().__init__()
        self.fc1 = Dense(hidden_dim, activation='tanh')
        self.fc2 = Dense(hidden_dim, activation='tanh')
        self.out = Dense(state_dim, activation='linear')

    def call(self, inputs):
        state_prev, delta_input = inputs
        x = Concatenate()([state_prev, delta_input])
        x = self.fc1(x)
        x = self.fc2(x)
        delta_state = self.out(x)
        return state_prev + delta_state

class MSC_Sequence(tf.keras.layers.Layer):
    def __init__(self, state_dim=5, hidden_dim=32, **kwargs):
        super().__init__(**kwargs)
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.msc_cell = MSC_Cell(state_dim=state_dim, hidden_dim=hidden_dim)

    def call(self, inputs):
        delta_seq, state_0 = inputs
        def step_fn(t, state, outputs):
            delta_t = delta_seq[:, t, :]
            state = self.msc_cell([state, delta_t])
            outputs = outputs.write(t, state)
            return t + 1, state, outputs

        outputs = tf.TensorArray(dtype=tf.float32, size=tf.shape(delta_seq)[1])
        _, final_state, outputs = tf.while_loop(
            lambda t, *_: t < tf.shape(delta_seq)[1],
            step_fn,
            [tf.constant(0), state_0, outputs]
        )
        return tf.transpose(outputs.stack(), [1, 0, 2])

def build_msc_model(state_dim=5, input_dim=3, output_dim=1):
    delta_input = Input(shape=(None, input_dim), name='delta_input')
    init_state = Input(shape=(state_dim,), name='init_state')

    state_seq = MSC_Sequence(state_dim=state_dim)([delta_input, init_state])
    stress_out = Dense(output_dim, name='stress_out')(state_seq)
    return Model(inputs=[delta_input, init_state], outputs=stress_out)

def build_msc_model_with_mask(state_dim=STATE_DIM, input_dim=INPUT_DIM, output_dim=1):
    """
    构建带有掩码机制的MSC模型，简化架构以避免形状问题
    """
    delta_input = Input(shape=(None, input_dim), name='delta_input')
    init_state = Input(shape=(state_dim,), name='init_state')
    mask = Input(shape=(None,), name='mask')
    
    # 使用掩码层处理序列
    state_seq = MSC_Sequence(state_dim=state_dim)([delta_input, init_state])
    
    # 使用掩码创建注意力掩码
    # 将1D掩码转换为2D掩码，适合注意力机制使用
    # 注意：为了避免形状问题，我们直接将掩码应用到输出
    masked_seq = tf.keras.layers.Multiply()([state_seq, tf.expand_dims(mask, -1)])
    
    # 输出层
    stress_out = Dense(output_dim, name='stress_out')(masked_seq)
    
    return Model(inputs=[delta_input, init_state, mask], outputs=stress_out)

def extract_temperature_from_filename(filename):
    temperature = filename.split('_')[3]
    print(temperature)
    return float(temperature)

def load_and_preprocess_data(file_list):
    """
    改进的数据加载和预处理函数，对长序列使用滑动窗口
    """
    paths = []
    targets = []
    sequence_lengths = []
    
    for file_idx, file in enumerate(file_list):
        df = pd.read_excel(file)
        df = df.rename(columns=lambda x: x.strip())
        if not {'Time', 'True_Strain', 'True_Stress'}.issubset(df.columns):
            print(f"文件 {file} 缺少必要列，已跳过")
            continue

        True_Strain = df['True_Strain']
        True_Stress = df['True_Stress']

        if 'com' in file:
            print(f"file'{file}'检测到压缩数据文件，已将应力和应变取反")
            True_Strain = -True_Strain
            True_Stress = -True_Stress

        T = extract_temperature_from_filename(file)
        df['Δε'] = True_Strain.diff().fillna(0)
        df['Δt'] = df['Time'].diff().fillna(1e-5)
        df['T'] = T
        
        x_full = df[['Δε', 'Δt', 'T']].values
        y_full = True_Stress.values.reshape(-1, 1)
        
        full_len = len(x_full)
        sequence_lengths.append(full_len)
        
        if full_len > WINDOW_SIZE:
            # 应用滑动窗口
            num_subsequences = 0
            for i in range(0, full_len - WINDOW_SIZE + 1, STRIDE):
                if num_subsequences >= MAX_SUBSEQUENCES:
                    break
                
                x_sub = x_full[i : i + WINDOW_SIZE]
                y_sub = y_full[i : i + WINDOW_SIZE]
                
                paths.append(x_sub)
                targets.append(y_sub)
                num_subsequences += 1
            print(f"文件 {file} (长度 {full_len}) 分割为 {num_subsequences} 个子序列")
        else:
            # 短序列直接添加
            paths.append(x_full)
            targets.append(y_full)
            
    # 序列长度统计（基于原始文件）
    print(f"序列长度统计 (原始文件):")
    print(f"最短: {min(sequence_lengths)}")
    print(f"最长: {max(sequence_lengths)}")
    print(f"平均: {np.mean(sequence_lengths):.2f}")
    print(f"中位数: {np.median(sequence_lengths)}")
    
    return paths, targets

class MSCProgressCallback(Callback):
    """
    自定义回调类，用于定期保存模型和训练进度
    """
    def __init__(self, save_path='./msc_models/', model_name='msc_model.h5', 
                 best_model_name='best_msc_model.h5', save_frequency=5,
                 x_scaler=None, y_scaler=None):
        super().__init__()
        self.save_path = save_path
        self.model_name = model_name
        self.best_model_name = best_model_name
        self.save_frequency = save_frequency
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.best_val_loss = float('inf')
        self.training_history = {
            'loss': [],
            'val_loss': [],
            'epoch_times': []
        }
        
        # 确保保存目录存在
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(os.path.join(save_path, 'backups'), exist_ok=True)
        
        # 加载已有的训练历史
        self.training_history = self._load_training_history()
        
    def _load_training_history(self):
        """加载已有的训练历史"""
        try:
            history_file = os.path.join(self.save_path, 'training_history.xlsx')
            if os.path.exists(history_file):
                print(f"Loading existing training history from: {history_file}")
                df = pd.read_excel(history_file)
                history = {}
                for column in df.columns:
                    if column != 'epoch':
                        history[column] = df[column].tolist()
                print(f"Loaded history with {len(history.get('loss', []))} epochs")
                return history
            else:
                print("No existing training history found, creating new record")
                return {
                    'loss': [],
                    'val_loss': [],
                    'epoch_times': []
                }
        except Exception as e:
            print(f"Error loading training history: {str(e)}")
            return {
                'loss': [],
                'val_loss': [],
                'epoch_times': []
            }
    
    def _save_training_history(self):
        """保存训练历史到文件，支持多种格式"""
        try:
            # 获取最大epoch数
            max_epochs = max(len(v) for v in self.training_history.values() if isinstance(v, list))
            
            # 创建数据字典
            data_dict = {
                'epoch': list(range(1, max_epochs + 1))
            }
            
            # 添加所有指标
            for key, values in self.training_history.items():
                if isinstance(values, list) and values:
                    data_dict[key] = values
            
            # 创建DataFrame
            df = pd.DataFrame(data_dict)
            
            # 尝试保存为不同的格式
            save_formats = [
                ('csv', lambda p: df.to_csv(p, index=False)),
                ('json', lambda p: df.to_json(p, orient='records', indent=4)),
                ('excel', lambda p: df.to_excel(p, index=False))
            ]
            
            saved = False
            for format_name, save_func in save_formats:
                try:
                    save_path = os.path.join(self.save_path, f'training_history.{format_name}')
                    save_func(save_path)
                    print(f"Training history saved to: {save_path}")
                    saved = True
                    break
                except Exception as e:
                    print(f"Failed to save as {format_name}: {e}")
                    continue
            
            if not saved:
                print("Warning: Failed to save training history in any format")
            
        except Exception as e:
            print(f"Error saving training history: {e}")
    
    def _plot_training_history(self):
        """绘制并保存训练历史图"""
        try:
            plt.switch_backend('Agg')  # 使用非交互式后端
            
            # 创建图形
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # 绘制损失曲线
            if self.training_history['loss']:
                ax1.plot(self.training_history['loss'], label='Training Loss', color='blue')
            if self.training_history['val_loss']:
                ax1.plot(self.training_history['val_loss'], label='Validation Loss', color='red')
            
            ax1.set_title('Training and Validation Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)
            
            # 绘制训练时间
            if self.training_history['epoch_times']:
                ax2.plot(self.training_history['epoch_times'], label='Epoch Duration', color='green')
                ax2.set_title('Training Time per Epoch')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Time (seconds)')
                ax2.legend()
                ax2.grid(True)
            
            plt.tight_layout()
            
            # 尝试不同的保存格式
            save_formats = ['png', 'pdf', 'jpg']
            for fmt in save_formats:
                try:
                    save_path = os.path.join(self.save_path, f'training_history.{fmt}')
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    print(f"Training plot saved to: {save_path}")
                    break
                except Exception as e:
                    print(f"Failed to save plot as {fmt}: {e}")
                    continue
            
            plt.close()
            
        except Exception as e:
            print(f"Error plotting training history: {e}")
    
    def _validate_model_file(self, model_path):
        """验证SavedModel格式模型的完整性"""
        try:
            # 检查SavedModel目录结构
            if not os.path.exists(os.path.join(model_path, 'saved_model.pb')):
                return False, "Missing saved_model.pb file"
            
            if not os.path.exists(os.path.join(model_path, 'variables')):
                return False, "Missing variables directory"
            
            # 尝试加载模型进行验证
            with tf.keras.utils.custom_object_scope({
                'MSC_Sequence': MSC_Sequence,
                'MaskedMSELoss': MaskedMSELoss
            }):
                test_model = tf.keras.models.load_model(model_path)
                return True, None
                
        except Exception as e:
            return False, str(e)
    
    def _safe_save_model(self, model, is_best=False):
        """安全的模型保存函数，使用SavedModel格式"""
        try:
            # 构建文件路径
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = self.best_model_name if is_best else self.model_name
            model_filename = model_filename.replace('.h5', '')  # 移除.h5扩展名
            final_path = os.path.join(self.save_path, model_filename)
            backup_dir = os.path.join(self.save_path, "backups")
            
            # 创建备份目录
            os.makedirs(backup_dir, exist_ok=True)
            
            # 使用SavedModel格式保存模型，包含自定义对象
            print(f"Saving model to: {final_path}")
            with tf.keras.utils.custom_object_scope({
                'MSC_Sequence': MSC_Sequence,
                'MaskedMSELoss': MaskedMSELoss
            }):
                tf.keras.models.save_model(
                    model,
                    final_path,
                    save_format='tf',
                    include_optimizer=True
                )
            
            # 验证保存的模型
            is_valid, error_msg = self._validate_model_file(final_path)
            if not is_valid:
                raise ValueError(f"Model validation failed: {error_msg}")
            
            # 创建备份（如果当前文件存在且不是最佳模型）
            if os.path.exists(final_path) and not is_best:
                backup_name = f"backup_{timestamp}_current_model"
                backup_path = os.path.join(backup_dir, backup_name)
                try:
                    shutil.copytree(final_path, backup_path)
                    print(f"Created backup: {backup_path}")
                except Exception as e:
                    print(f"Warning: Failed to create backup: {e}")
            
            print(f"Successfully saved model to: {final_path}")
            
            # 同时保存标准化器
            if self.x_scaler is not None and self.y_scaler is not None:
                scaler_path = os.path.join(self.save_path, 'scalers')
                os.makedirs(scaler_path, exist_ok=True)
                try:
                    joblib.dump(self.x_scaler, os.path.join(scaler_path, 'x_scaler.save'))
                    joblib.dump(self.y_scaler, os.path.join(scaler_path, 'y_scaler.save'))
                    print("Scalers saved successfully")
                except Exception as e:
                    print(f"Warning: Failed to save scalers: {e}")
            
            return final_path
            
        except Exception as e:
            print(f"Error in safe_save_model: {str(e)}")
            return None
    
    def _cleanup_old_backups(self, backup_dir, keep_count=3):
        """清理旧的备份文件，保留最新的几个"""
        try:
            # 分别获取最佳模型和当前模型的备份
            backup_files = glob.glob(os.path.join(backup_dir, "backup_*.h5"))
            if not backup_files:
                return
            
            # 将备份文件按类型分组
            best_backups = [f for f in backup_files if 'best' in f]
            current_backups = [f for f in backup_files if 'current' in f]
            
            # 按修改时间排序并清理每种类型的备份
            for backup_list in [best_backups, current_backups]:
                if not backup_list:
                    continue
                
                # 按修改时间排序（最新的在前）
                backup_list.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                
                # 删除多余的备份
                files_to_delete = backup_list[keep_count:]
                for file_path in files_to_delete:
                    try:
                        os.remove(file_path)
                        print(f"Removed old backup: {os.path.basename(file_path)}")
                    except Exception as e:
                        print(f"Failed to remove backup {file_path}: {str(e)}")
            
            # 清理临时文件
            temp_files = glob.glob(os.path.join(backup_dir, "temp_*.h5"))
            for temp_file in temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                        print(f"Removed temporary file: {os.path.basename(temp_file)}")
                except Exception as e:
                    print(f"Failed to remove temp file {temp_file}: {str(e)}")
                    
        except Exception as e:
            print(f"Error in cleanup_old_backups: {str(e)}")
    
    def on_epoch_begin(self, epoch, logs=None):
        """记录epoch开始时间"""
        self.epoch_start_time = time.time()
    
    def on_epoch_end(self, epoch, logs=None):
        """epoch结束时的处理"""
        # 记录训练时间
        epoch_duration = time.time() - self.epoch_start_time
        
        # 更新训练历史
        self.training_history['loss'].append(logs.get('loss', 0))
        self.training_history['val_loss'].append(logs.get('val_loss', 0))
        self.training_history['epoch_times'].append(epoch_duration)
        
        # 检查是否是最佳模型
        current_val_loss = logs.get('val_loss', float('inf'))
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            print(f"\nNew best model found at epoch {epoch + 1} with val_loss: {current_val_loss:.6f}")
            self._safe_save_model(self.model, is_best=True)
        
        # 定期保存进度
        if (epoch + 1) % self.save_frequency == 0:
            print(f"\nSaving progress at epoch {epoch + 1}/{self.params['epochs']}")
            
            # 保存当前模型
            self._safe_save_model(self.model, is_best=False)
            
            # 保存训练历史
            self._save_training_history()
            
            # 绘制训练历史
            self._plot_training_history()
            
            print(f"Progress saved. Continuing training...")
        
        # 显示训练进度
        if (epoch + 1) % max(1, self.save_frequency // 2) == 0:
            avg_epoch_time = np.mean(self.training_history['epoch_times'][-10:]) if len(self.training_history['epoch_times']) > 0 else 0
            remaining_epochs = self.params['epochs'] - (epoch + 1)
            estimated_time = avg_epoch_time * remaining_epochs
            
            print(f"\nEpoch {epoch + 1}/{self.params['epochs']} - "
                  f"Loss: {logs.get('loss', 0):.6f} - "
                  f"Val_loss: {current_val_loss:.6f} - "
                  f"Best_val_loss: {self.best_val_loss:.6f}")
            print(f"Average epoch time: {avg_epoch_time:.2f}s - "
                  f"Estimated remaining time: {estimated_time/60:.1f}min")

def load_or_create_model_with_history(model_path='./msc_models/', model_name='msc_model', 
                                     best_model_name='best_msc_model',
                                     state_dim=STATE_DIM, input_dim=INPUT_DIM, output_dim=1):
    """
    加载已存在的SavedModel格式模型或创建新模型
    """
    model_file_path = os.path.join(model_path, best_model_name)
    fallback_path = os.path.join(model_path, model_name)
    
    # 首先尝试加载最佳模型
    if os.path.exists(model_file_path):
        print(f"Found existing best model: {model_file_path}")
        try:
            model = tf.keras.models.load_model(
                model_file_path,
                custom_objects={'MSC_Sequence': MSC_Sequence}
            )
            print("Successfully loaded existing best model")
            return model, False
        except Exception as e:
            print(f"Error loading best model: {e}")
    
    # 尝试加载当前模型
    if os.path.exists(fallback_path):
        print(f"Found existing current model: {fallback_path}")
        try:
            model = tf.keras.models.load_model(
                fallback_path,
                custom_objects={'MSC_Sequence': MSC_Sequence}
            )
            print("Successfully loaded existing current model")
            return model, False
        except Exception as e:
            print(f"Error loading current model: {e}")
    
    print("Creating new model")
    model = build_msc_model_with_mask(state_dim=state_dim, input_dim=input_dim, output_dim=output_dim)
    return model, True

def shuffle_sequences(X_seq, Y_seq, init_state_seq, random_state=42):
    """
    随机打乱训练序列
    
    参数:
    X_seq: 输入序列
    Y_seq: 目标序列
    init_state_seq: 初始状态序列
    random_state: 随机种子
    
    返回:
    打乱后的序列
    """
    # 设置随机种子
    np.random.seed(random_state)
    
    # 生成随机索引
    indices = np.random.permutation(len(X_seq))
    
    # 打乱所有序列
    X_shuffled = X_seq[indices]
    Y_shuffled = Y_seq[indices]
    init_state_shuffled = init_state_seq[indices]
    
    return X_shuffled, Y_shuffled, init_state_shuffled

def group_sequences_by_length(file_list):
    # 按序列长度分组
    length_groups = {}
    for file in file_list:
        df = pd.read_excel(file)
        length = len(df)
        if length not in length_groups:
            length_groups[length] = []
        length_groups[length].append(file)
    
    # 对每组使用合适的序列长度
    all_X = []
    all_Y = []
    for length, files in length_groups.items():
        if length <= TARGET_SEQUENCE_LENGTH:
            X, Y = process_group(files, length)
            all_X.append(X)
            all_Y.append(Y)
    
    return np.concatenate(all_X), np.concatenate(all_Y)

def augment_short_sequences(x, y, target_length):
    """对短序列进行数据增强"""
    if len(x) >= target_length:
        return x, y
    
    # 使用插值方法生成更多数据点
    t = np.linspace(0, 1, len(x))
    t_new = np.linspace(0, 1, target_length)
    
    x_aug = np.array([interp1d(t, x[:, i])(t_new) for i in range(x.shape[1])]).T
    y_aug = interp1d(t, y.flatten())(t_new).reshape(-1, 1)
    
    return x_aug, y_aug

def analyze_sequence_lengths(file_list):
    lengths = []
    for file in file_list:
        df = pd.read_excel(file)
        lengths.append(len(df))
    
    print(f"序列长度统计:")
    print(f"最短: {min(lengths)}")
    print(f"最长: {max(lengths)}")
    print(f"平均: {np.mean(lengths):.2f}")
    print(f"中位数: {np.median(lengths)}")
    
    # 绘制长度分布直方图
    plt.hist(lengths, bins=20)
    plt.title("序列长度分布")
    plt.xlabel("长度")
    plt.ylabel("数量")
    plt.show()

def prepare_sequences(X_paths, Y_paths, x_scaler, y_scaler):
    """
    准备训练序列，处理滑动窗口生成的子序列
    """
    X_norm = [x_scaler.transform(x) for x in X_paths]
    Y_norm = [y_scaler.transform(y) for y in Y_paths]
    
    # 所有序列都应填充到 WINDOW_SIZE
    max_len = WINDOW_SIZE
    
    print(f"使用统一序列长度 (填充/截断至): {max_len}")
    
    # 准备序列和掩码
    X_seq = []
    Y_seq = []
    masks = []
    
    for x, y in zip(X_norm, Y_norm):
        seq_len = len(x)
        
        # 生成掩码（1表示有效数据，0表示填充）
        mask = np.ones(min(seq_len, max_len), dtype=np.float32)
        
        # 填充或截断序列
        x_padded = np.pad(x[:max_len], ((0, max(0, max_len - seq_len)), (0, 0)), 
                         mode='constant', constant_values=0)
        y_padded = np.pad(y[:max_len], ((0, max(0, max_len - seq_len)), (0, 0)), 
                         mode='constant', constant_values=0)
        mask_padded = np.pad(mask, (0, max(0, max_len - len(mask))), 
                           mode='constant', constant_values=0)
        
        X_seq.append(x_padded)
        Y_seq.append(y_padded)
        masks.append(mask_padded)
    
    return np.array(X_seq, dtype=np.float32), np.array(Y_seq, dtype=np.float32), np.array(masks, dtype=np.float32)

class MaskedMSELoss(tf.keras.losses.Loss):
    """
    自定义掩码MSE损失类
    """
    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO, name='masked_mse_loss'):
        super().__init__(reduction=reduction, name=name)
    
    def call(self, y_true, y_pred, sample_weight=None):
        # 确保输入是float32类型
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # 计算MSE
        mse = tf.keras.losses.mean_squared_error(y_true, y_pred)
        
        # 如果提供了sample_weight（掩码），应用它
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32)
            # 确保掩码形状与预测值匹配
            sample_weight = tf.reshape(sample_weight, tf.shape(mse))
            mse = mse * sample_weight
            # 计算有效样本的平均损失
            return tf.reduce_sum(mse) / tf.reduce_sum(sample_weight)
        
        return tf.reduce_mean(mse)
    
    def get_config(self):
        """获取配置，用于序列化"""
        base_config = super().get_config()
        return base_config

def resume_training_from_checkpoint(model_path='./msc_models/', model_name='msc_model', 
                                   best_model_name='best_msc_model', resume_from_best=True):
    """
    从SavedModel格式的检查点恢复训练
    
    参数:
    model_path: 模型保存路径
    model_name: 当前模型目录名
    best_model_name: 最佳模型目录名
    resume_from_best: 是否从最佳模型恢复（默认True）
    
    返回:
    model: 加载的模型
    epoch_offset: 已训练的epoch数
    """
    # 确定要加载的模型目录
    if resume_from_best:
        model_dir = os.path.join(model_path, best_model_name)
        fallback_dir = os.path.join(model_path, model_name)
    else:
        model_dir = os.path.join(model_path, model_name)
        fallback_dir = os.path.join(model_path, best_model_name)
    
    # 尝试加载模型
    model = None
    if os.path.exists(model_dir):
        try:
            print(f"恢复训练，加载模型: {model_dir}")
            with tf.keras.utils.custom_object_scope({
                'MSC_Sequence': MSC_Sequence,
                'MaskedMSELoss': MaskedMSELoss
            }):
                model = tf.keras.models.load_model(model_dir)
            print("模型加载成功")
        except Exception as e:
            print(f"加载主模型失败: {e}")
    
    # 如果主模型加载失败，尝试后备模型
    if model is None and os.path.exists(fallback_dir):
        try:
            print(f"尝试加载后备模型: {fallback_dir}")
            with tf.keras.utils.custom_object_scope({
                'MSC_Sequence': MSC_Sequence,
                'MaskedMSELoss': MaskedMSELoss
            }):
                model = tf.keras.models.load_model(fallback_dir)
            print("后备模型加载成功")
        except Exception as e:
            print(f"加载后备模型也失败: {e}")
    
    # 如果仍然没有模型，返回None
    if model is None:
        print("未找到可用的检查点模型")
        return None, 0
    
    # 尝试加载训练历史以确定epoch偏移
    epoch_offset = 0
    try:
        history_file = os.path.join(model_path, 'training_history.xlsx')
        if os.path.exists(history_file):
            df = pd.read_excel(history_file)
            epoch_offset = len(df)
            print(f"从训练历史中检测到已完成 {epoch_offset} 个epochs")
        else:
            print("未找到训练历史文件，从epoch 0开始")
    except Exception as e:
        print(f"读取训练历史时出错: {e}")
        epoch_offset = 0
    
    return model, epoch_offset

def create_training_config():
    """
    创建训练配置字典，便于管理和保存训练参数
    """
    config = {
        'STATE_DIM': STATE_DIM,
        'INPUT_DIM': INPUT_DIM,
        'HIDDEN_DIM': HIDDEN_DIM,
        'LEARNING_RATE': LEARNING_RATE,
        'TARGET_SEQUENCE_LENGTH': TARGET_SEQUENCE_LENGTH,
        'WINDOW_SIZE': WINDOW_SIZE,
        'STRIDE': STRIDE,
        'MAX_SUBSEQUENCES': MAX_SUBSEQUENCES,
        'train_test_split_ratio': 0.8,
        'random_seed': 42
    }
    return config

def save_training_config(config, save_path):
    """保存训练配置"""
    try:
        import json
        config_file = './training_config.json'
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Training configuration saved to: {config_file}")
    except Exception as e:
        print(f"Error saving training config: {e}")

def load_training_config(save_path):
    """加载训练配置"""
    try:
        import json
        config_file = './training_config.json'
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
            print(f"Training configuration loaded from: {config_file}")
            return config
        else:
            print("No training configuration found, using defaults")
            return None
    except Exception as e:
        print(f"Error loading training config: {e}")
        return None

def save_dataset_to_npz(X_paths, Y_paths, save_path='./msc_models/dataset.npz'):
    """
    将预处理后的数据集保存为npz格式
    
    参数:
    X_paths: 输入序列列表
    Y_paths: 目标序列列表
    save_path: 保存路径
    
    返回:
    save_path: 保存的文件路径
    """
    try:
        # 确保保存目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 将列表转换为numpy数组
        X_array = np.array(X_paths, dtype=object)
        Y_array = np.array(Y_paths, dtype=object)
        
        # 保存为npz文件
        np.savez_compressed(
            save_path,
            X_paths=X_array,
            Y_paths=Y_array,
            save_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        print(f"数据集已保存至: {save_path}")
        return save_path
    except Exception as e:
        print(f"保存数据集时出错: {e}")
        return None

def load_dataset_from_npz(npz_path='./msc_models/dataset.npz'):
    """
    从npz文件加载数据集
    
    参数:
    npz_path: npz文件路径
    
    返回:
    X_paths: 输入序列列表
    Y_paths: 目标序列列表
    """
    try:
        if not os.path.exists(npz_path):
            print(f"数据集文件不存在: {npz_path}")
            return None, None
        
        # 加载npz文件
        data = np.load(npz_path, allow_pickle=True)
        X_paths = data['X_paths'].tolist()
        Y_paths = data['Y_paths'].tolist()
        
        # 打印数据集信息
        print(f"从 {npz_path} 加载数据集")
        print(f"保存时间: {data['save_time']}")
        print(f"序列数量: {len(X_paths)}")
        
        # 打印序列长度统计
        lengths = [len(x) for x in X_paths]
        print(f"序列长度统计:")
        print(f"最短: {min(lengths)}")
        print(f"最长: {max(lengths)}")
        print(f"平均: {np.mean(lengths):.2f}")
        print(f"中位数: {np.median(lengths)}")
        
        return X_paths, Y_paths
    except Exception as e:
        print(f"加载数据集时出错: {e}")
        return None, None

if __name__ == '__main__':
    # 设置TensorFlow的默认数据类型为float32
    tf.keras.backend.set_floatx('float32')
    data_dir = "/mnt/data/msc_models/"
    if os.path.exists(data_dir):
        # 设置模型保存路径
        save_model_path = data_dir
    else:
        data_dir = '/Users/tianyunhu/Documents/temp/code/Test_app/EMSC_Model'
        save_model_path = os.path.join(data_dir, 'msc_models')

    print(f"save_model_path: {save_model_path}")

    model_name = 'msc_model'  # SavedModel格式不需要文件扩展名
    best_model_name = 'best_msc_model'
    dataset_path = os.path.join(data_dir, 'dataset.npz')


    
    # 设置训练参数
    resume_training = True  # 设置为True以恢复训练，False从头开始
    epochs = 500  # 总的训练epochs数（包括之前已训练的）
    batch_size = 8
    save_frequency = 1  # 每5个epoch保存一次
    
    # 创建和保存训练配置
    training_config = create_training_config()
    training_config.update({
        'epochs': epochs,
        'batch_size': batch_size,
        'save_frequency': save_frequency
    })
    save_training_config(training_config, save_model_path)
    
    # 尝试从npz文件加载数据集，如果不存在则重新处理数据
    if os.path.exists(dataset_path):
        X_paths, Y_paths = load_dataset_from_npz(dataset_path)
    else:
        dataset_path = '/Users/tianyunhu/Documents/temp/code/Test_app/EMSC_Model/msc_models/dataset.npz'
        X_paths, Y_paths = load_dataset_from_npz(dataset_path)
        
    
    
    # 数据标准化
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    
    # 加载已有的标准化器（如果存在）
    scaler_path = os.path.join(save_model_path, 'scalers')
    x_scaler_file = os.path.join(scaler_path, 'x_scaler.save')
    y_scaler_file = os.path.join(scaler_path, 'y_scaler.save')
    
    if os.path.exists(x_scaler_file) and os.path.exists(y_scaler_file):
        print("Loading existing scalers...")
        x_scaler = joblib.load(x_scaler_file)
        y_scaler = joblib.load(y_scaler_file)
        print("Scalers loaded successfully")
    else:
        print("Creating new scalers...")
    # 注意：标准化器应在所有（包括子序列）数据上拟合
    x_scaler.fit(np.vstack(X_paths))
    y_scaler.fit(np.vstack(Y_paths))
    print("New scalers created")

    # 准备序列和掩码
    X_seq, Y_seq, masks = prepare_sequences(X_paths, Y_paths, x_scaler, y_scaler)
    init_state_seq = np.zeros((len(X_seq), STATE_DIM), dtype=np.float32)

    # 随机打乱序列
    print("随机打乱训练序列...")
    np.random.seed(training_config['random_seed'])  # 使用配置中的随机种子
    indices = np.random.permutation(len(X_seq))
    X_seq = X_seq[indices]
    Y_seq = Y_seq[indices]
    masks = masks[indices]
    init_state_seq = init_state_seq[indices]

    # 划分训练集和验证集
    train_size = int(training_config['train_test_split_ratio'] * len(X_seq))
    X_train = X_seq[:train_size]
    Y_train = Y_seq[:train_size]
    mask_train = masks[:train_size]
    init_state_train = init_state_seq[:train_size]
    
    X_val = X_seq[train_size:]
    Y_val = Y_seq[train_size:]
    mask_val = masks[train_size:]
    init_state_val = init_state_seq[train_size:]
    
    # 决定是恢复训练还是从头开始
    epoch_offset = 0
    if resume_training:
        print("尝试恢复训练...")
        resumed_model, epoch_offset = resume_training_from_checkpoint(
            model_path=save_model_path,
            model_name=model_name,
            best_model_name=best_model_name,
            resume_from_best=True
        )
        
        if resumed_model is not None:
            model = resumed_model
            is_new_model = False
            print(f"成功恢复训练，将从epoch {epoch_offset + 1} 开始")
        else:
            print("无法恢复训练，将创建新模型")
            model, is_new_model = load_or_create_model_with_history(
                model_path=save_model_path,
                model_name=model_name,
                best_model_name=best_model_name
            )
    else:
        print("从头开始训练...")
        model, is_new_model = load_or_create_model_with_history(
            model_path=save_model_path,
            model_name=model_name,
            best_model_name=best_model_name
        )
    
    # 编译模型
    optimizer = Adam(LEARNING_RATE)
    loss_fn = MaskedMSELoss()
    
    model.compile(
        optimizer=optimizer,
        loss=loss_fn
    )
    
    if is_new_model:
        model.summary()

    # 创建自定义回调，设置保存频率
    progress_callback = MSCProgressCallback(
        save_path=save_model_path,
        model_name=model_name,
        best_model_name=best_model_name,
        save_frequency=save_frequency,
        x_scaler=x_scaler,
        y_scaler=y_scaler
    )
    
    # 计算实际要训练的epochs
    remaining_epochs = epochs - epoch_offset
    if remaining_epochs <= 0:
        print(f"模型已经训练了 {epoch_offset} epochs，达到设定的总epochs {epochs}")
        print("如需继续训练，请增加总epochs数")
    else:
        print(f"\n开始训练 MSC 模型...")
        print(f"已完成epochs: {epoch_offset}")
        print(f"剩余epochs: {remaining_epochs}")
        print(f"总epochs目标: {epochs}")
        print(f"批次大小: {batch_size}")
        print(f"保存频率: 每 {save_frequency} epochs")
        print(f"模型保存路径: {save_model_path}")
        print(f"训练数据大小: {len(X_train)}")
        print(f"验证数据大小: {len(X_val)}")
        
        # 调整initial_epoch参数以支持恢复训练
        initial_epoch = epoch_offset
        
        # 使用 Keras fit 进行训练，包含自定义回调
        history = model.fit(
        x={
            'delta_input': X_train, 
            'init_state': init_state_train,
            'mask': mask_train
        },
        y=Y_train,
        validation_data=(
            {
                'delta_input': X_val,
                'init_state': init_state_val,
                'mask': mask_val
            },
            Y_val
        ),
            batch_size=batch_size,
            epochs=epochs,
            initial_epoch=initial_epoch,  # 从指定的epoch开始
        verbose=1,
            shuffle=True,  # 每个epoch打乱训练数据
            callbacks=[progress_callback]  # 添加自定义回调
        )
        
        # 训练完成后最终保存
        print("\n训练完成，执行最终保存...")
        final_model_path = progress_callback._safe_save_model(model, is_best=False)
        
        # 保存最终的训练历史和图表
        progress_callback._save_training_history()
        progress_callback._plot_training_history()
        
        print("="*60)
        print("训练完成总结:")
        print(f"最佳验证损失: {progress_callback.best_val_loss:.6f}")
        print(f"总训练时间: {sum(progress_callback.training_history['epoch_times']):.2f} 秒")
        print(f"平均每epoch时间: {np.mean(progress_callback.training_history['epoch_times']):.2f} 秒")
        print(f"模型保存路径: {save_model_path}")
        print(f"最佳模型: {os.path.join(save_model_path, best_model_name)}")
        print(f"当前模型: {os.path.join(save_model_path, model_name)}")
        print(f"训练历史: {os.path.join(save_model_path, 'training_history.xlsx')}")
        print(f"训练图表: {os.path.join(save_model_path, 'training_history.png')}")
        print(f"训练配置: {os.path.join(save_model_path, 'training_config.json')}")
        print("="*60)

        # 绘制最终的训练损失图（Keras内置历史）
        if len(history.history['loss']) > 0:
            plt.figure(figsize=(12, 6))
            
            # 子图1：损失曲线
            plt.subplot(1, 2, 1)
            plt.plot(range(initial_epoch + 1, epochs + 1), history.history['loss'], 
                    label='Training Loss', color='blue')
            plt.plot(range(initial_epoch + 1, epochs + 1), history.history['val_loss'], 
                    label='Validation Loss', color='red')
            plt.title('Model Loss During Training (Final)')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            
            # 子图2：训练时间
            if progress_callback.training_history['epoch_times']:
                plt.subplot(1, 2, 2)
                recent_times = progress_callback.training_history['epoch_times'][-remaining_epochs:]
                plt.plot(range(initial_epoch + 1, initial_epoch + 1 + len(recent_times)), 
                        recent_times, label='Epoch Duration', color='green')
                plt.title('Training Time per Epoch')
                plt.xlabel('Epoch')
                plt.ylabel('Time (seconds)')
                plt.legend()
                plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_model_path, 'final_training_summary.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"最终训练总结图保存至: {os.path.join(save_model_path, 'final_training_summary.png')}")
