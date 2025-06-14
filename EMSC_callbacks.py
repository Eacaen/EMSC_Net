"""
EMSC回调模块
包含所有训练回调相关的类
"""

import os
import time
import shutil
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib
from datetime import datetime
from tensorflow.keras.callbacks import Callback, EarlyStopping
from EMSC_model import MSC_Sequence
from EMSC_losses import EMSCLoss

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
        
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(os.path.join(save_path, 'backups'), exist_ok=True)
        
        self.training_history = self._load_training_history()
    
    def _load_training_history(self):
        """加载已有的训练历史"""
        try:
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
        """保存训练历史到文件"""
        try:
            max_epochs = max(len(v) for v in self.training_history.values() if isinstance(v, list))
            
            data_dict = {
                'epoch': list(range(1, max_epochs + 1))
            }
            
            for key, values in self.training_history.items():
                if isinstance(values, list) and values:
                    data_dict[key] = values
            
            df = pd.DataFrame(data_dict)
            
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
            plt.switch_backend('Agg')
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            if self.training_history['loss']:
                ax1.plot(self.training_history['loss'], label='Training Loss', color='blue')
            if self.training_history['val_loss']:
                ax1.plot(self.training_history['val_loss'], label='Validation Loss', color='red')
            
            ax1.set_title('Training and Validation Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)
            
            if self.training_history['epoch_times']:
                ax2.plot(self.training_history['epoch_times'], label='Epoch Duration', color='green')
                ax2.set_title('Training Time per Epoch')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Time (seconds)')
                ax2.legend()
                ax2.grid(True)
            
            plt.tight_layout()
            
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
            if not os.path.exists(os.path.join(model_path, 'saved_model.pb')):
                return False, "Missing saved_model.pb file"
            
            if not os.path.exists(os.path.join(model_path, 'variables')):
                return False, "Missing variables directory"
            
            with tf.keras.utils.custom_object_scope({
                'MSC_Sequence': MSC_Sequence,
                'EMSCLoss': EMSCLoss
            }):
                test_model = tf.keras.models.load_model(model_path)
                return True, None
                
        except Exception as e:
            return False, str(e)
    
    def _safe_save_model(self, model, is_best=False):
        """安全地保存模型，处理可能的错误"""
        try:
            # 确保保存目录存在
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            
            # 确定保存路径
            if is_best:
                save_path = os.path.join(self.save_path, self.best_model_name)
            else:
                save_path = os.path.join(self.save_path, self.model_name)
            
            print(f"Saving model to: {save_path}")
            
            # 尝试追踪所有函数
            try:
                # 获取模型的所有层
                for layer in model.layers:
                    # 确保层的call方法被追踪
                    if hasattr(layer, 'call'):
                        layer.call = tf.function(layer.call)
                
                # 确保模型的call方法被追踪
                if hasattr(model, 'call'):
                    model.call = tf.function(model.call)
            except Exception as e:
                print(f"Warning: Failed to trace some functions: {e}")
            
            # 保存模型
            model.save(save_path, save_format='tf')
            print(f"Model saved successfully to: {save_path}")
            return save_path
            
        except Exception as e:
            print(f"Error in safe_save_model: {e}")
            return None
    
    def _cleanup_old_backups(self, backup_dir, keep_count=3):
        """清理旧的备份文件"""
        try:
            backup_files = glob.glob(os.path.join(backup_dir, "backup_*.h5"))
            if not backup_files:
                return
            
            best_backups = [f for f in backup_files if 'best' in f]
            current_backups = [f for f in backup_files if 'current' in f]
            
            for backup_list in [best_backups, current_backups]:
                if not backup_list:
                    continue
                
                backup_list.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                
                files_to_delete = backup_list[keep_count:]
                for file_path in files_to_delete:
                    try:
                        os.remove(file_path)
                        print(f"Removed old backup: {os.path.basename(file_path)}")
                    except Exception as e:
                        print(f"Failed to remove backup {file_path}: {str(e)}")
            
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
        epoch_duration = time.time() - self.epoch_start_time
        
        self.training_history['loss'].append(logs.get('loss', 0))
        self.training_history['val_loss'].append(logs.get('val_loss', 0))
        self.training_history['epoch_times'].append(epoch_duration)
        
        current_val_loss = logs.get('val_loss', float('inf'))
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            print(f"\nNew best model found at epoch {epoch + 1} with val_loss: {current_val_loss:.6f}")
            self._safe_save_model(self.model, is_best=True)
        
        if (epoch + 1) % self.save_frequency == 0:
            print(f"\nSaving progress at epoch {epoch + 1}/{self.params['epochs']}")
            
            self._safe_save_model(self.model, is_best=False)
            self._save_training_history()
            self._plot_training_history()
            
            print(f"Progress saved. Continuing training...")
        
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

class LearningRateScheduler(tf.keras.callbacks.Callback):
    """
    自定义学习率调度器回调
    支持多种学习率衰减策略：
    1. 指数衰减
    2. 余弦退火
    3. 基于验证损失的动态调整
    """
    def __init__(self, 
                 initial_learning_rate=0.001,
                 decay_type='exponential',
                 decay_steps=1000,
                 decay_rate=0.9,
                 min_learning_rate=1e-6,
                 patience=5,
                 factor=0.5,
                 verbose=1):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_type = decay_type
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.min_learning_rate = min_learning_rate
        self.patience = patience
        self.factor = factor
        self.verbose = verbose
        
        # 用于基于验证损失的动态调整
        self.best_val_loss = float('inf')
        self.wait = 0
        self.current_lr = initial_learning_rate
        
    def on_train_begin(self, logs=None):
        """训练开始时初始化"""
        self.optimizer = self.model.optimizer
        self.initial_learning_rate = float(self.optimizer.learning_rate.numpy())
        self.current_lr = self.initial_learning_rate
        
    def on_epoch_end(self, epoch, logs=None):
        """每个epoch结束时更新学习率"""
        # 初始化new_lr为当前学习率
        new_lr = self.current_lr
        
        if self.decay_type == 'exponential':
            # 指数衰减
            new_lr = self.initial_learning_rate * (self.decay_rate ** (epoch / self.decay_steps))
        elif self.decay_type == 'cosine':
            # 余弦退火
            progress = epoch / self.decay_steps
            new_lr = self.min_learning_rate + 0.5 * (self.initial_learning_rate - self.min_learning_rate) * \
                    (1 + np.cos(np.pi * progress))
        elif self.decay_type == 'validation':
            # 基于验证损失的动态调整
            current_val_loss = logs.get('val_loss')
            if current_val_loss is None:
                print("Warning: val_loss not found in logs, skipping learning rate adjustment")
                return
                
            if current_val_loss < self.best_val_loss:
                self.best_val_loss = current_val_loss
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    new_lr = max(self.current_lr * self.factor, self.min_learning_rate)
                    self.wait = 0
        else:
            raise ValueError(f"Unsupported decay type: {self.decay_type}")
        
        # 确保学习率不低于最小值
        new_lr = max(new_lr, self.min_learning_rate)
        
        # 更新优化器的学习率
        self.optimizer.learning_rate.assign(new_lr)
        self.current_lr = new_lr
        
        if self.verbose > 0:
            print(f"\nEpoch {epoch + 1}: Learning rate is {new_lr:.6f}")

def create_learning_rate_scheduler(initial_learning_rate=0.001,
                                 decay_type='exponential',
                                 decay_steps=1000,
                                 decay_rate=0.9,
                                 min_learning_rate=1e-6,
                                 patience=5,
                                 factor=0.5,
                                 verbose=1):
    """创建学习率调度器回调"""
    return LearningRateScheduler(
        initial_learning_rate=initial_learning_rate,
        decay_type=decay_type,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        min_learning_rate=min_learning_rate,
        patience=patience,
        factor=factor,
        verbose=verbose
    )

def create_early_stopping_callback():
    """创建早停回调"""
    return EarlyStopping(
        monitor='val_loss',      # 监控验证集损失值
        patience=15,             # 容忍验证集损失15个epoch不改善
        min_delta=1e-4,         # 最小改善阈值，损失改善需超过此值
        mode='min',             # 监控指标越小越好
        restore_best_weights=True,  # 训练结束时恢复最佳权重
        verbose=1               # 打印早停相关信息
    )

class NaNMonitorCallback(tf.keras.callbacks.Callback):
    """
    NaN监控回调，用于混合精度训练中的数值稳定性检查
    """
    def __init__(self, terminate_on_nan=True, patience=3):
        super().__init__()
        self.terminate_on_nan = terminate_on_nan
        self.patience = patience
        self.nan_count = 0
        
    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        
        if loss is not None and (np.isnan(loss) or np.isinf(loss)):
            print(f"\nWarning: NaN/Inf loss detected at batch {batch}!")
            self.nan_count += 1
            
            if self.terminate_on_nan and self.nan_count >= self.patience:
                print(f"Terminating training due to {self.nan_count} consecutive NaN losses")
                self.model.stop_training = True
        else:
            # 重置计数器如果损失正常
            self.nan_count = 0
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_loss = logs.get('val_loss')
        
        if val_loss is not None and (np.isnan(val_loss) or np.isinf(val_loss)):
            print(f"\nWarning: NaN/Inf validation loss at epoch {epoch}!")

def create_nan_monitor_callback(terminate_on_nan=True, patience=3):
    """创建NaN监控回调"""
    return NaNMonitorCallback(terminate_on_nan=terminate_on_nan, patience=patience)