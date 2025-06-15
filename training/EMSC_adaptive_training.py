"""
EMSC自适应训练脚本
专门用于解决训练停滞问题的高级训练策略
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.EMSC_model import build_msc_model
from core.EMSC_losses import EMSCLoss
from training.EMSC_callbacks import MSCProgressCallback, create_early_stopping_callback


class AdaptiveLossWeight:
    """自适应损失权重调整器"""
    
    def __init__(self, initial_mse_weight=1.0, initial_reg_weight=0.01):
        self.mse_weight = initial_mse_weight
        self.reg_weight = initial_reg_weight
        self.loss_history = []
        self.patience_count = 0
        
    def update_weights(self, epoch, current_loss, val_loss=None):
        """根据训练状况动态调整损失权重"""
        self.loss_history.append(current_loss)
        
        # 如果损失停滞，调整权重
        if len(self.loss_history) >= 10:
            recent_losses = self.loss_history[-10:]
            loss_variance = np.var(recent_losses)
            loss_improvement = (recent_losses[0] - recent_losses[-1]) / recent_losses[0]
            
            # 如果损失改善很小且方差很小（停滞）
            if loss_improvement < 0.01 and loss_variance < 1e-6:
                self.patience_count += 1
                if self.patience_count >= 5:
                    # 降低正则化权重，增强拟合能力
                    self.reg_weight *= 0.5
                    self.mse_weight *= 1.2
                    self.patience_count = 0
                    print(f"Epoch {epoch}: 检测到停滞，调整权重 - MSE权重: {self.mse_weight:.4f}, 正则化权重: {self.reg_weight:.4f}")
            else:
                self.patience_count = 0
                
        return self.mse_weight, self.reg_weight


class CyclicalLearningRate:
    """循环学习率调度器"""
    
    def __init__(self, base_lr=1e-5, max_lr=1e-2, step_size=100):
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        
    def get_lr(self, epoch):
        """计算当前epoch的学习率"""
        cycle = np.floor(1 + epoch / (2 * self.step_size))
        x = np.abs(epoch / self.step_size - 2 * cycle + 1)
        lr = self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x))
        return lr


class WarmRestartCallback(tf.keras.callbacks.Callback):
    """热重启回调 - 检测停滞时重启训练"""
    
    def __init__(self, patience=20, restart_lr_factor=0.5, min_delta=1e-6):
        super().__init__()
        self.patience = patience
        self.restart_lr_factor = restart_lr_factor
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.wait = 0
        self.restart_count = 0
        
    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get('val_loss') or logs.get('loss')
        
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1
            
        if self.wait >= self.patience:
            # 执行热重启
            self.restart_count += 1
            new_lr = self.model.optimizer.learning_rate.numpy() * self.restart_lr_factor
            self.model.optimizer.learning_rate.assign(max(new_lr, 1e-7))
            
            # 添加一些噪声到模型权重
            for layer in self.model.layers:
                if hasattr(layer, 'kernel'):
                    noise = tf.random.normal(layer.kernel.shape, stddev=0.01)
                    layer.kernel.assign_add(noise)
                    
            self.wait = 0
            print(f"Epoch {epoch + 1}: 执行热重启 #{self.restart_count}, 新学习率: {new_lr:.2e}")


def create_adaptive_training_config():
    """创建自适应训练配置"""
    return {
        'use_cyclical_lr': True,
        'use_warm_restart': True,
        'use_adaptive_loss_weight': True,
        'use_gradient_scaling': True,
        'use_noise_injection': True,
        'early_stopping_patience': 50,  # 增加早停耐心
        'lr_reduction_patience': 8      # 更快的学习率调整
    }


def noise_injection_callback():
    """噪声注入回调 - 防止过拟合和局部最优"""
    class NoiseInjection(tf.keras.callbacks.Callback):
        def __init__(self, noise_std=0.001, injection_frequency=10):
            super().__init__()
            self.noise_std = noise_std
            self.injection_frequency = injection_frequency
            
        def on_epoch_end(self, epoch, logs=None):
            if epoch % self.injection_frequency == 0 and epoch > 0:
                # 只对部分层注入噪声
                for layer in self.model.layers[-2:]:  # 只对最后两层注入噪声
                    if hasattr(layer, 'kernel'):
                        noise = tf.random.normal(layer.kernel.shape, stddev=self.noise_std)
                        layer.kernel.assign_add(noise)
                print(f"Epoch {epoch + 1}: 注入噪声到模型权重")
                        
    return NoiseInjection()


def adaptive_train_model(model, train_dataset, val_dataset, epochs=2000, 
                        initial_lr=1e-3, state_dim=8, save_path='./models/'):
    """
    使用自适应策略训练EMSC模型
    
    Args:
        model: EMSC模型
        train_dataset: 训练数据集
        val_dataset: 验证数据集
        epochs: 训练轮数
        initial_lr: 初始学习率
        state_dim: 状态维度
        save_path: 模型保存路径
    """
    
    print("🚀 启动自适应训练模式...")
    
    # 创建自适应组件
    adaptive_loss = AdaptiveLossWeight()
    cyclical_lr = CyclicalLearningRate(base_lr=initial_lr/10, max_lr=initial_lr*2)
    
    # 创建优化器 - 使用更激进的设置
    optimizer = Adam(
        learning_rate=initial_lr,
        clipnorm=0.5,    # 更严格的梯度裁剪
        clipvalue=0.1,   # 更严格的值裁剪
        amsgrad=True     # 使用AMSGrad变体
    )
    
    # 创建损失函数
    custom_loss = EMSCLoss(state_dim=state_dim)
    
    # 编译模型
    model.compile(
        optimizer=optimizer,
        loss=custom_loss,
        jit_compile=False
    )
    
    # 创建回调列表
    callbacks = []
    
    # 1. 进度保存回调
    progress_callback = MSCProgressCallback(
        save_path=save_path,
        model_name='adaptive_msc_model',
        best_model_name='best_adaptive_msc_model',
        save_frequency=5  # 更频繁保存
    )
    callbacks.append(progress_callback)
    
    # 2. 循环学习率回调
    class CyclicalLRCallback(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            lr = cyclical_lr.get_lr(epoch)
            self.model.optimizer.learning_rate.assign(lr)
            if epoch % 50 == 0:
                print(f"Epoch {epoch}: 循环学习率 = {lr:.2e}")
    
    callbacks.append(CyclicalLRCallback())
    
    # 3. 热重启回调
    callbacks.append(WarmRestartCallback(patience=15, restart_lr_factor=0.3))
    
    # 4. 噪声注入回调
    callbacks.append(noise_injection_callback())
    
    # 5. 改进的早停回调
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=80,        # 更大的耐心
        min_delta=1e-7,     # 更小的最小改善
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stopping)
    
    # 6. 学习率监控
    lr_monitor = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: cyclical_lr.get_lr(epoch), verbose=0
    )
    callbacks.append(lr_monitor)
    
    print("📊 自适应训练配置:")
    print(f"- 初始学习率: {initial_lr}")
    print(f"- 循环学习率范围: {cyclical_lr.base_lr:.2e} - {cyclical_lr.max_lr:.2e}")
    print(f"- 梯度裁剪: clipnorm=0.5, clipvalue=0.1")
    print(f"- 优化器: Adam with AMSGrad")
    print(f"- 热重启: 15 epoch patience")
    print(f"- 噪声注入: 每10 epoch")
    print(f"- 早停: 80 epoch patience")
    
    # 开始训练
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    print("✅ 自适应训练完成!")
    
    return model, history


if __name__ == "__main__":
    print("EMSC自适应训练模块")
    print("请从主训练脚本调用 adaptive_train_model 函数") 