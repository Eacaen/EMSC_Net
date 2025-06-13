"""
动态批次大小调整模块
根据CPU使用率自动调整批次大小以优化资源利用率
"""

import threading
import time
import numpy as np
import tensorflow as tf
import sys
import os
from typing import Optional, Callable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class DynamicBatchSizeCallback(tf.keras.callbacks.Callback):
    """动态批次大小调整回调"""
    
    def __init__(self, 
                 target_cpu_usage=75.0,  # 目标CPU使用率
                 tolerance=10.0,         # 容忍度
                 min_batch_size=16,      # 最小批次大小
                 max_batch_size=512,     # 最大批次大小
                 adjustment_factor=1.5,  # 调整因子（更激进）
                 monitor_interval=15,    # 监控间隔（秒）- 更频繁
                 patience=1,             # 连续多少个epoch才调整 - 更快响应
                 verbose=True):
        """
        Args:
            target_cpu_usage: 目标CPU使用率（百分比）
            tolerance: CPU使用率容忍度
            min_batch_size: 最小批次大小
            max_batch_size: 最大批次大小
            adjustment_factor: 批次大小调整因子
            monitor_interval: CPU监控间隔
            patience: 连续多少个epoch使用率不达标才调整
            verbose: 是否打印调整信息
        """
        super().__init__()
        self.target_cpu_usage = target_cpu_usage
        self.tolerance = tolerance
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.adjustment_factor = adjustment_factor
        self.monitor_interval = monitor_interval
        self.patience = patience
        self.verbose = verbose
        
        # 状态变量
        self.current_batch_size = None
        self.cpu_usage_history = []
        self.low_usage_count = 0
        self.high_usage_count = 0
        self.monitoring = False
        self.monitor_thread = None
        self.latest_cpu_usage = 0.0
        
        # 数据重建回调
        self.dataset_rebuilder = None
        
    def set_dataset_rebuilder(self, rebuilder_func: Callable):
        """设置数据集重建函数"""
        self.dataset_rebuilder = rebuilder_func
    
    def on_train_begin(self, logs=None):
        """训练开始时启动CPU监控"""
        try:
            import psutil
            self.psutil = psutil
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_cpu)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            if self.verbose:
                print(f"启动动态批次大小调整 (目标CPU使用率: {self.target_cpu_usage}%)")
        except ImportError:
            if self.verbose:
                print("警告: 未安装psutil，无法进行动态批次大小调整")
    
    def on_train_end(self, logs=None):
        """训练结束时停止监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        if self.verbose:
            print("停止动态批次大小调整")
    
    def on_epoch_begin(self, epoch, logs=None):
        """记录当前epoch的批次大小"""
        if hasattr(self.model, 'current_batch_size'):
            self.current_batch_size = self.model.current_batch_size
        if self.verbose and epoch == 0:
            print(f"初始批次大小: {self.current_batch_size}")
    
    def on_epoch_end(self, epoch, logs=None):
        """在epoch结束时决定是否调整批次大小"""
        if not self.cpu_usage_history:
            return
        
        # 计算最近几次的平均CPU使用率
        recent_usage = np.mean(self.cpu_usage_history[-3:]) if len(self.cpu_usage_history) >= 3 else self.latest_cpu_usage
        
        # 判断是否需要调整
        if recent_usage < self.target_cpu_usage - self.tolerance:
            self.low_usage_count += 1
            self.high_usage_count = 0
        elif recent_usage > self.target_cpu_usage + self.tolerance:
            self.high_usage_count += 1
            self.low_usage_count = 0
        else:
            self.low_usage_count = 0
            self.high_usage_count = 0
        
        # 计算CPU使用率差距，决定调整幅度
        cpu_gap = abs(recent_usage - self.target_cpu_usage)
        
        # 根据差距决定调整因子（超激进的调整策略）
        if cpu_gap > 30:
            dynamic_factor = 4.0  # 差距很大时，超激进调整
        elif cpu_gap > 20:
            dynamic_factor = 3.0
        elif cpu_gap > 10:
            dynamic_factor = 2.5
        else:
            dynamic_factor = 2.0
        
        # 达到patience才调整
        new_batch_size = self.current_batch_size
        adjustment_made = False
        
        if self.low_usage_count >= self.patience and self.current_batch_size < self.max_batch_size:
            # CPU使用率过低，增加批次大小
            new_batch_size = min(
                int(self.current_batch_size * dynamic_factor),
                self.max_batch_size
            )
            # 确保是8的倍数
            new_batch_size = (new_batch_size // 8) * 8
            adjustment_made = True
            self.low_usage_count = 0
            
        elif self.high_usage_count >= self.patience and self.current_batch_size > self.min_batch_size:
            # CPU使用率过高，减少批次大小
            new_batch_size = max(
                int(self.current_batch_size / dynamic_factor),
                self.min_batch_size
            )
            # 确保是8的倍数
            new_batch_size = (new_batch_size // 8) * 8
            adjustment_made = True
            self.high_usage_count = 0
        
        if adjustment_made and new_batch_size != self.current_batch_size:
            if self.verbose:
                print(f"\n🔄 动态调整批次大小:")
                print(f"   当前CPU使用率: {recent_usage:.1f}%")
                print(f"   目标CPU使用率: {self.target_cpu_usage}%")
                print(f"   使用率差距: {cpu_gap:.1f}%")
                print(f"   动态调整因子: {dynamic_factor:.1f}x")
                print(f"   批次大小: {self.current_batch_size} → {new_batch_size}")
                
                # 给出进一步的诊断建议
                if cpu_gap > 30:
                    print(f"   💡 CPU使用率差距很大，可能存在以下瓶颈:")
                    print(f"      - 数据I/O等待时间过长")
                    print(f"      - 网络存储延迟")
                    print(f"      - 内存带宽限制")
                    print(f"      - TensorFlow线程配置不当")
            
            # 更新批次大小
            self.current_batch_size = new_batch_size
            
            # 如果有数据集重建函数，调用它
            if self.dataset_rebuilder:
                try:
                    self.dataset_rebuilder(new_batch_size)
                    if self.verbose:
                        print(f"   ✅ 数据集已重建为新批次大小")
                except Exception as e:
                    if self.verbose:
                        print(f"   ❌ 数据集重建失败: {e}")
        
        # 增加CPU使用率监控显示
        elif self.verbose and epoch % 5 == 0:  # 每5个epoch显示一次状态
            print(f"\n📊 CPU使用率状态:")
            print(f"   当前: {recent_usage:.1f}% | 目标: {self.target_cpu_usage}% | 批次: {self.current_batch_size}")
            if cpu_gap > 15:
                print(f"   ⚠️  使用率偏差较大 ({cpu_gap:.1f}%)，继续监控...")
    
    def _monitor_cpu(self):
        """CPU监控主循环"""
        while self.monitoring:
            try:
                # 获取CPU使用率
                cpu_percent = self.psutil.cpu_percent(interval=1)
                
                # 记录使用率
                self.latest_cpu_usage = cpu_percent
                self.cpu_usage_history.append(cpu_percent)
                
                # 保持历史记录不超过20条
                if len(self.cpu_usage_history) > 20:
                    self.cpu_usage_history.pop(0)
                
                # 等待下一次检查
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                if self.verbose:
                    print(f"CPU监控出错: {e}")
                time.sleep(self.monitor_interval)


class DynamicBatchTrainer:
    """支持动态批次大小的训练器"""
    
    def __init__(self, model, train_data_info, val_data_info, initial_batch_size=32):
        """
        Args:
            model: Keras模型
            train_data_info: 训练数据信息 (X_train, Y_train, init_states_train)
            val_data_info: 验证数据信息 (X_val, Y_val, init_states_val)
            initial_batch_size: 初始批次大小
        """
        self.model = model
        self.train_data_info = train_data_info
        self.val_data_info = val_data_info
        self.current_batch_size = initial_batch_size
        self.model.current_batch_size = initial_batch_size
        
        # 数据集
        self.train_dataset = None
        self.val_dataset = None
        self._rebuild_datasets()
    
    def _rebuild_datasets(self):
        """重建数据集"""
        from core.EMSC_data import create_tf_dataset
        
        X_train, Y_train, init_states_train = self.train_data_info
        X_val, Y_val, init_states_val = self.val_data_info
        
        # 重建训练数据集
        self.train_dataset = create_tf_dataset(
            X_train, Y_train, init_states_train,
            batch_size=self.current_batch_size,
            shuffle=True,
            num_parallel_calls=tf.data.AUTOTUNE
        ).prefetch(tf.data.AUTOTUNE)
        
        # 重建验证数据集
        self.val_dataset = create_tf_dataset(
            X_val, Y_val, init_states_val,
            batch_size=self.current_batch_size,
            shuffle=False,
            num_parallel_calls=tf.data.AUTOTUNE
        ).prefetch(tf.data.AUTOTUNE)
        
        # 更新模型的批次大小记录
        self.model.current_batch_size = self.current_batch_size
    
    def update_batch_size(self, new_batch_size):
        """更新批次大小并重建数据集"""
        if new_batch_size != self.current_batch_size:
            self.current_batch_size = new_batch_size
            self._rebuild_datasets()
    
    def fit(self, epochs, callbacks=None, **kwargs):
        """训练模型"""
        if callbacks is None:
            callbacks = []
        
        # 添加动态批次大小调整回调
        dynamic_callback = DynamicBatchSizeCallback(verbose=True)
        dynamic_callback.set_dataset_rebuilder(self.update_batch_size)
        callbacks.append(dynamic_callback)
        
        return self.model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            **kwargs
        )


def create_dynamic_batch_callback(target_cpu_usage=75.0, 
                                min_batch_size=16,
                                max_batch_size=512,
                                verbose=True):
    """
    创建动态批次大小调整回调
    
    Args:
        target_cpu_usage: 目标CPU使用率
        min_batch_size: 最小批次大小
        max_batch_size: 最大批次大小
        verbose: 是否打印详细信息
    
    Returns:
        DynamicBatchSizeCallback实例
    """
    return DynamicBatchSizeCallback(
        target_cpu_usage=target_cpu_usage,
        min_batch_size=min_batch_size,
        max_batch_size=max_batch_size,
        verbose=verbose
    )