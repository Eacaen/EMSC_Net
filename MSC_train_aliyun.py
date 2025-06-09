# Minimal State Cell (MSC) training pipeline with multi-temperature, multi-rate experimental data
# 输入: [delta_strain, delta_time, delta_temperature]
#     ↓
# 1. 方向向量计算: d_n = delta_strain / |delta_strain|
#     ↓
# 2. 内部层处理: l = tanh(W_a·l + b_a) * tanh(W_b·l + b_b)
#     ↓
# 3. 门控参数: α,β,γ = exp(W·l_d + b)
#     ↓
# 4. 候选状态: c_n = tanh(W_c·l_d + b_c)
#     ↓
# 5. 更新门: z = 1 - exp(-α|Δε| - βΔt - γ|T|)
#     ↓
# 6. 状态更新: h_n = (1-z)h_prev + z·c_n
#     ↓
# 7. 输出: σ_n = W_out·h_n (True_Stress)

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

class MSC_Cell(tf.keras.layers.Layer):
    """
    增强型最小状态单元 (EMSC) 实现
    
    参数:
    state_dim: 状态向量维度 (h)
    input_dim: 输入特征维度 [delta_strain, delta_time, delta_temperature, init_strain, init_time, init_temp]
    hidden_dim: 内部层维度 (l)
    num_internal_layers: 内部层数量
    """
    def __init__(self, state_dim=5, input_dim=3, hidden_dim=32, num_internal_layers=2):
        super().__init__()
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 1. 内部层 (tanh∘tanh 逐层)
        self.internal_layers = []
        for i in range(num_internal_layers):
            # 每层包含两个 Dense 层 (W_a, W_b)
            self.internal_layers.append([
                Dense(hidden_dim, activation='tanh', use_bias=True, name=f'W_a_{i}'),
                Dense(hidden_dim, activation='tanh', use_bias=True, name=f'W_b_{i}')
            ])
        
        # 2. 门控参数 (alpha, beta, gamma)
        self.W_alpha = Dense(1, use_bias=True, name='W_alpha')
        self.W_beta = Dense(1, use_bias=True, name='W_beta')
        self.W_gamma = Dense(1, use_bias=True, name='W_gamma')
        
        # 3. 候选状态
        self.W_c = Dense(state_dim, use_bias=True, name='W_c')
        
        # 4. 输出层 (True_Stress)
        self.W_out = Dense(1, use_bias=False, name='W_out')
    
    def calc_direction_vec(self, delta_features):
        """
        计算增量方向向量
        
        参数:
        delta_features: [delta_strain, delta_time, delta_temperature] (batch, 3)
        
        返回:
        direction: 归一化的方向向量 (batch, 3)
        """
        delta_norm = tf.sqrt(tf.reduce_sum(tf.square(delta_features), axis=-1, keepdims=True))
        delta_norm = tf.maximum(delta_norm, 1e-7)  # 避免除零
        direction = delta_features / delta_norm
        return direction
    
    def process_internal_layers(self, l_0):
        """
        处理内部层 (tanh∘tanh 逐层)
        
        参数:
        l_0: 初始特征向量 [h_prev, current_temp, direction]
        
        返回:
        l_d: 内部层输出
        """
        l = l_0
        for W_a, W_b in self.internal_layers:
            # 直接使用 Dense 层的 tanh 激活
            l_a = W_a(l)
            l_b = W_b(l)
            l = l_a * l_b
        return l
    
    def calc_gate_params(self, l_d):
        """
        计算门控参数 (exp激活确保非负)
        
        参数:
        l_d: 内部层输出
        
        返回:
        alpha, beta, gamma: 三个门控参数
        """
        alpha = tf.exp(self.W_alpha(l_d))  # 应变增量门控
        beta = tf.exp(self.W_beta(l_d))    # 时间增量门控
        gamma = tf.exp(self.W_gamma(l_d))  # 温度门控
        return alpha, beta, gamma
    
    def calc_update_gate(self, alpha, beta, gamma, delta_features):
        """
        计算更新门
        
        参数:
        alpha, beta, gamma: 门控参数
        delta_features: [delta_strain, delta_time, delta_temperature]
        
        返回:
        z: 更新门值
        """
        delta_strain = delta_features[..., 0:1]
        delta_time = delta_features[..., 1:2]
        delta_temperature = delta_features[..., 2:3]
        
        z = 1 - tf.exp(-alpha * tf.abs(delta_strain) - 
                       beta * delta_time - 
                       gamma * tf.abs(delta_temperature))
        return z
    
    def reconstruct_temp_seq(self, delta_x):
        """
        重建温度序列
        
        参数:
        delta_x: 输入特征 [delta_strain, delta_time, delta_temperature, init_strain, init_time, init_temp] (batch, seq_len, 6)
        
        返回:
        temp_seq: 重建后的温度序列 (batch, seq_len, 1)
        """
        # 分离动态特征和静态特征
        delta_features = delta_x[..., :3]  # [delta_strain, delta_time, delta_temperature]
        init_features = delta_x[..., 3:]   # [init_strain, init_time, init_temp]
        
        # 获取初始温度和温度增量
        init_temp = init_features[..., 0, 2:3]  # (batch, 1)
        delta_temp = delta_features[..., 2:3]   # (batch, seq_len, 1)
        
        # 计算温度序列
        temp_seq = tf.cumsum(delta_temp, axis=1)  # 累加温度增量
        temp_seq = temp_seq + init_temp  # 加上初始温度
        
        return temp_seq

    def call(self, inputs):
        """
        前向传播
        
        参数:
        inputs: [h_prev, delta_x]
            h_prev: 前一步状态向量 (batch, state_dim)
            delta_x: 输入特征 [delta_strain, delta_time, delta_temperature, init_strain, init_time, init_temp] (batch, seq_len, 6)
            
        返回:
        h_n: 新状态向量 (batch, state_dim)
        sigma_n: 输出应力 (batch, 1)
        """
        h_prev, delta_x = inputs
        
        # 分离动态特征和静态特征
        delta_features = delta_x[..., :3]  # [delta_strain, delta_time, delta_temperature]
        
        # 1. 重建温度序列
        temp_seq = self.reconstruct_temp_seq(delta_x)
        
        # 2. 计算方向向量 (仅对动态特征)
        direction = self.calc_direction_vec(delta_features)
        
        # 3. 内部层处理
        l_0 = tf.concat([h_prev, temp_seq, direction], axis=-1)
        l_d = self.process_internal_layers(l_0)
        
        # 4. 计算门控参数
        alpha, beta, gamma = self.calc_gate_params(l_d)
        
        # 5. 计算候选状态
        c_n = tf.tanh(self.W_c(l_d))
        
        # 6. 计算更新门
        z = self.calc_update_gate(alpha, beta, gamma, delta_features)
        
        # 7. 状态更新
        h_n = (1 - z) * h_prev + z * c_n
        
        # 8. 输出应力
        sigma_n = self.W_out(h_n)
        
        return h_n, sigma_n

class MSC_Sequence(tf.keras.layers.Layer):
    """
    EMSC 序列处理层
    
    参数:
    state_dim: 状态向量维度
    input_dim: 输入特征维度 [delta_strain, delta_time, delta_temperature]
    hidden_dim: 内部层维度
    num_internal_layers: 内部层数量
    """
    def __init__(self, state_dim=5, input_dim=3, hidden_dim=32, num_internal_layers=2, **kwargs):
        super().__init__(**kwargs)
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_internal_layers = num_internal_layers
        self.msc_cell = MSC_Cell(
            state_dim=state_dim,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_internal_layers=num_internal_layers
        )
        
    def call(self, inputs):
        """
        处理输入序列
        
        参数:
        inputs: [delta_seq, state_0]
            delta_seq: 输入序列 [delta_strain, delta_time, delta_temperature] (batch, seq_len, input_dim)
            state_0: 初始状态 (batch, state_dim)
            
        返回:
        outputs: 输出序列 (True_Stress) (batch, seq_len, 1)
        """
        delta_seq, state_0 = inputs
        
        # 使用 tf.while_loop 处理序列
        def step_fn(t, state, outputs):
            # 获取当前时间步的输入
            delta_x_t = delta_seq[:, t, :]  # (batch, input_dim)
            
            # 调用 MSC 单元
            new_state, output = self.msc_cell([state, delta_x_t])
            
            # 更新输出
            outputs = outputs.write(t, output)
            
            return t + 1, new_state, outputs
        
        # 初始化输出张量数组
        outputs = tf.TensorArray(dtype=tf.float32, size=tf.shape(delta_seq)[1])
        
        # 执行循环
        _, final_state, outputs = tf.while_loop(
            lambda t, *_: t < tf.shape(delta_seq)[1],
            step_fn,
            [tf.constant(0), state_0, outputs]
        )
        
        # 转置输出 (batch, seq_len, 1)
        return tf.transpose(outputs.stack(), [1, 0, 2])


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
            # 尝试按优先级加载不同格式的历史文件
            history_files = [
                ('training_history.csv', pd.read_csv),
                ('training_history.xlsx', pd.read_excel),
                ('training_history.excel', pd.read_excel)
            ]
            
            for filename, read_func in history_files:
                history_file = os.path.join(self.save_path, filename)
                if os.path.exists(history_file):
                    print(f"Loading existing training history from: {history_file}")
                    try:
                        df = read_func(history_file)
                        history = {}
                        for column in df.columns:
                            if column != 'epoch':
                                history[column] = df[column].tolist()
                        print(f"Loaded history with {len(history.get('loss', []))} epochs")
                        return history
                    except Exception as e:
                        print(f"Failed to load {filename}: {e}")
                        continue
            
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
            
            # 尝试保存为不同的格式，优先保存CSV格式以确保一致性
            save_formats = [
                ('csv', lambda p: df.to_csv(p, index=False, encoding='utf-8')),
                ('xlsx', lambda p: df.to_excel(p, index=False)),
                ('json', lambda p: df.to_json(p, orient='records', indent=4))
            ]
            
            saved_count = 0
            for format_name, save_func in save_formats:
                try:
                    save_path = os.path.join(self.save_path, f'training_history.{format_name}')
                    save_func(save_path)
                    print(f"Training history saved to: {save_path}")
                    saved_count += 1
                    # 只要CSV保存成功就继续保存其他格式作为备份
                    if format_name == 'csv':
                        continue
                except Exception as e:
                    print(f"Failed to save as {format_name}: {e}")
                    continue
            
            if saved_count == 0:
                print("Warning: Failed to save training history in any format")
            else:
                print(f"Training history saved in {saved_count} format(s)")
            
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


def create_training_config(state_dim=8, input_dim=6, hidden_dim=32, learning_rate=1e-3, 
                         target_sequence_length=5000, window_size=None, stride=None, 
                         max_subsequences=200, train_test_split_ratio=0.8, random_seed=42,
                         epochs=500, batch_size=8, save_frequency=1):
    """
    创建训练配置字典
    
    参数:
    state_dim: 状态向量维度
    input_dim: 输入特征维度
    hidden_dim: 隐藏层维度
    learning_rate: 学习率
    target_sequence_length: 目标序列长度
    window_size: 滑动窗口大小
    stride: 滑动窗口步长
    max_subsequences: 最大子序列数
    train_test_split_ratio: 训练集比例
    random_seed: 随机种子
    epochs: 训练轮数
    batch_size: 批次大小
    save_frequency: 保存频率
    """
    if window_size is None:
        window_size = target_sequence_length
    if stride is None:
        stride = target_sequence_length // 10
        
    config = {
        'STATE_DIM': state_dim,
        'INPUT_DIM': input_dim,
        'HIDDEN_DIM': hidden_dim,
        'LEARNING_RATE': learning_rate,
        'TARGET_SEQUENCE_LENGTH': target_sequence_length,
        'WINDOW_SIZE': window_size,
        'STRIDE': stride,
        'MAX_SUBSEQUENCES': max_subsequences,
        'train_test_split_ratio': train_test_split_ratio,
        'random_seed': random_seed,
        'epochs': epochs,
        'batch_size': batch_size,
        'save_frequency': save_frequency
    }
    return config

def build_msc_model(state_dim=8, input_dim=6, output_dim=1,
                   hidden_dim=32, num_internal_layers=2):
    """
    构建 EMSC 模型
    
    参数:
    state_dim: 状态向量维度
    input_dim: 输入特征维度 [delta_strain, delta_time, delta_temperature, init_strain, init_time, init_temp]
    output_dim: 输出维度
    hidden_dim: 内部层维度
    num_internal_layers: 内部层数量
    """
    # 输入层
    delta_input = Input(shape=(None, input_dim), name='delta_input')
    init_state = Input(shape=(state_dim,), name='init_state')
    
    # 使用 EMSC 序列层处理输入
    state_seq = MSC_Sequence(
        state_dim=state_dim,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_internal_layers=num_internal_layers
    )([delta_input, init_state])
    
    # 输出层
    stress_out = Dense(output_dim, name='stress_out')(state_seq)
    
    return Model(inputs=[delta_input, init_state], outputs=stress_out)

def load_or_create_model_with_history(model_path='./msc_models/', model_name='msc_model', 
                                     best_model_name='best_msc_model',
                                     state_dim=8, input_dim=6, output_dim=1,
                                     hidden_dim=32, num_internal_layers=2):
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
    model = build_msc_model(
        state_dim=state_dim,
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        num_internal_layers=num_internal_layers
    )
    return model, True

def resume_training_from_checkpoint(model_path='./msc_models/', model_name='msc_model', 
                                   best_model_name='best_msc_model', resume_from_best=True,
                                   state_dim=8, input_dim=6, output_dim=1,
                                   hidden_dim=32, num_internal_layers=2):
    """
    从SavedModel格式的检查点恢复训练
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
        # 尝试按优先级加载不同格式的历史文件
        history_files = [
            ('training_history.csv', pd.read_csv),
            ('training_history.xlsx', pd.read_excel),
            ('training_history.excel', pd.read_excel)
        ]
        
        for filename, read_func in history_files:
            history_file = os.path.join(model_path, filename)
            if os.path.exists(history_file):
                try:
                    df = read_func(history_file)
                    epoch_offset = len(df)
                    print(f"从训练历史中检测到已完成 {epoch_offset} 个epochs (来源: {filename})")
                    break
                except Exception as e:
                    print(f"读取 {filename} 时出错: {e}")
                    continue
        else:
            print("未找到训练历史文件，从epoch 0开始")
    except Exception as e:
        print(f"读取训练历史时出错: {e}")
        epoch_offset = 0
    
    return model, epoch_offset

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

    print(f"当前工作目录: {os.getcwd()}")
    print(tf.__version__)

    # 设置TensorFlow的默认数据类型为float32
    tf.keras.backend.set_floatx('float32')
    
    # 模型参数配置
    state_dim = 8
    input_dim = 6  # [delta_strain, delta_time, delta_temperature, init_strain, init_time, init_temp]
    hidden_dim = 32
    learning_rate = 1e-3
    target_sequence_length = 1000
    
    # 训练参数配置
    import argparse
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='EMSC模型训练参数')
    parser.add_argument('--epochs', type=int, default=2000, help='训练轮数 (默认: 2000)')
    parser.add_argument('--save_frequency', type=int, default=10, help='模型保存频率，每N个epoch保存一次 (默认: 10)')
    args = parser.parse_args()
    
    epochs = args.epochs
    batch_size = 8
    save_frequency = args.save_frequency
    resume_training = True
    
    # 路径配置
    data_dir = "/mnt/data/msc_models/"
    if os.path.exists(data_dir):
        save_model_path = data_dir
    else:
        data_dir = '/Users/tianyunhu/Documents/temp/code/Test_app/EMSC_Model'
        save_model_path = os.path.join(data_dir, 'msc_models')
    
    print(f"save_model_path: {save_model_path}")
    
    model_name = 'msc_model'
    best_model_name = 'best_msc_model'
    dataset_path = os.path.join(data_dir, 'dataset.npz')
    
    # 创建和保存训练配置
    training_config = create_training_config(
        state_dim=state_dim,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        target_sequence_length=target_sequence_length,
        epochs=epochs,
        batch_size=batch_size,
        save_frequency=save_frequency
    )
    save_training_config(training_config, save_model_path)
    
    # 加载数据集
    dataset_path = data_dir + '/dataset.npz'
    X_paths, Y_paths = load_dataset_from_npz(dataset_path)

    if X_paths is None or Y_paths is None:
        dataset_path = '/Users/tianyunhu/Documents/temp/code/Test_app/EMSC_Model/msc_models/dataset.npz'
        X_paths, Y_paths = load_dataset_from_npz(dataset_path)

    
    # 加载标准化器
    scaler_path = os.path.join(save_model_path, 'scalers')
    x_scaler_file = os.path.join(scaler_path, 'x_scaler.save')
    y_scaler_file = os.path.join(scaler_path, 'y_scaler.save')
    
    if not (os.path.exists(x_scaler_file) and os.path.exists(y_scaler_file)):
        raise ValueError("找不到标准化器文件，请确保数据集已正确生成并包含标准化器")
    
    print("加载标准化器...")
    x_scaler = joblib.load(x_scaler_file)
    y_scaler = joblib.load(y_scaler_file)
    print("标准化器加载成功")
    
    # 使用标准化器转换数据
    print("标准化数据...")
    x_scaled = [x_scaler.transform(x) for x in X_paths]
    y_scaled = [y_scaler.transform(y) for y in Y_paths]
    print("数据标准化完成")

    # 准备训练数据
    print("准备训练序列...")
    X_seq = []
    Y_seq = []
    init_states = []
    
    for x, y in zip(x_scaled, y_scaled):
        # 创建初始状态向量
        init_state = np.zeros(state_dim, dtype=np.float32)
        
        X_seq.append(x)
        Y_seq.append(y)
        init_states.append(init_state)
    
    # 转换为numpy数组
    X_seq = np.array(X_seq, dtype=np.float32)
    Y_seq = np.array(Y_seq, dtype=np.float32)
    init_states = np.array(init_states, dtype=np.float32)
    print("训练序列准备完成")

    # 随机打乱序列
    print("随机打乱训练序列...")
    np.random.seed(training_config['random_seed'])
    indices = np.random.permutation(len(X_seq))
    X_seq = X_seq[indices]
    Y_seq = Y_seq[indices]
    init_states = init_states[indices]

    # 划分训练集和验证集
    train_size = int(training_config['train_test_split_ratio'] * len(X_seq))
    X_train = X_seq[:train_size]
    Y_train = Y_seq[:train_size]
    init_states_train = init_states[:train_size]
    
    X_val = X_seq[train_size:]
    Y_val = Y_seq[train_size:]
    init_states_val = init_states[train_size:]
    
    print(f"训练集大小: {len(X_train)}")
    print(f"验证集大小: {len(X_val)}")
    print(f"输入维度: {input_dim} [delta_strain, delta_time, delta_temperature]")
    print(f"输出维度: 1 [True_Stress]")
    print(f"状态维度: {state_dim}")

    # 决定是恢复训练还是从头开始
    epoch_offset = 0
    if resume_training:
        print("尝试恢复训练...")
        resumed_model, epoch_offset = resume_training_from_checkpoint(
            model_path=save_model_path,
            model_name=model_name,
            best_model_name=best_model_name,
            resume_from_best=True,
            state_dim=state_dim,
            input_dim=input_dim,
            output_dim=1,
            hidden_dim=hidden_dim,
            num_internal_layers=2
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
                best_model_name=best_model_name,
                state_dim=state_dim,
                input_dim=input_dim,
                output_dim=1,
                hidden_dim=hidden_dim,
                num_internal_layers=2
            )
    else:
        print("从头开始训练...")
        model, is_new_model = load_or_create_model_with_history(
            model_path=save_model_path,
            model_name=model_name,
            best_model_name=best_model_name,
            state_dim=state_dim,
            input_dim=input_dim,
            output_dim=1,
            hidden_dim=hidden_dim,
            num_internal_layers=2
        )
    
    # 编译模型
    optimizer = Adam(learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='mse'  # 使用标准MSE损失，因为不再需要掩码
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
    remaining_epochs = epochs  # 直接使用新设置的轮数

    if remaining_epochs <= 0:
        print(f"模型已经训练了 {epoch_offset} epochs，达到设定的总epochs {epochs}")
        print("如需继续训练，请增加总epochs数")
    else:
        print(f"\n开始训练 MSC 模型...")
        print(f"已完成epochs: {epoch_offset}")
        print(f"剩余epochs: {remaining_epochs}")  # 显示新设置的轮数
        print(f"总epochs目标: {epochs + epoch_offset}")  # 总目标为已训练+新训练
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
                'init_state': init_states_train
            },
            y=Y_train,
            validation_data=(
                {
                    'delta_input': X_val,
                    'init_state': init_states_val
                },
                Y_val
            ),
            batch_size=batch_size,
            epochs=epochs,
            initial_epoch=initial_epoch,
            verbose=1,
            shuffle=True,
            callbacks=[progress_callback]
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
        print(f"训练历史: {os.path.join(save_model_path, 'training_history.csv')}")
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
