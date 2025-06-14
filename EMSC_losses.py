"""
EMSC模型的自定义损失函数模块

包含：
1. EMSCLoss: 增强型最小状态单元的自定义损失函数，包含门控参数正则化
2. MaskedMSELoss: 带掩码的MSE损失函数
"""

import tensorflow as tf

class EMSCLoss(tf.keras.losses.Loss):
    """
    增强型最小状态单元的自定义损失函数
    
    参数:
    state_dim: 状态向量维度
    name: 损失函数名称
    """
    def __init__(self, state_dim=8, name='emsc_loss', **kwargs):
        # 兼容不同TensorFlow版本的构造函数
        try:
            # 尝试使用新版本构造函数
            super().__init__(name=name)
        except TypeError:
            # 兼容旧版本
            super().__init__()
        self.state_dim = state_dim
        self.epoch = tf.Variable(0, trainable=False, dtype=tf.int32)
        
        # 获取当前策略的dtype（使用完全自定义的属性名避免冲突）
        mixed_precision_policy = tf.keras.mixed_precision.global_policy()
        self.emsc_compute_dtype = mixed_precision_policy.compute_dtype
        self.emsc_variable_dtype = mixed_precision_policy.variable_dtype
    
    def call(self, y_true, y_pred, gate_params=None):
        """
        计算损失值
        
        参数:
        y_true: 真实值 (batch_size, seq_len, 1)
        y_pred: 预测值 (batch_size, seq_len, 1)
        gate_params: 门控参数字典，包含 'alpha', 'beta', 'gamma' 序列
        """
        # 确保输入使用正确的dtype
        y_true = tf.cast(y_true, self.emsc_compute_dtype)
        y_pred = tf.cast(y_pred, self.emsc_compute_dtype)
        
        # 数值稳定性检查 - 裁剪输入到合理范围
        if self.emsc_compute_dtype == tf.float16:
            # float16 的安全范围
            y_true = tf.clip_by_value(y_true, -1000.0, 1000.0)
            y_pred = tf.clip_by_value(y_pred, -1000.0, 1000.0)
        
        # 检查 NaN 和 Inf
        y_true = tf.where(tf.math.is_finite(y_true), y_true, tf.zeros_like(y_true))
        y_pred = tf.where(tf.math.is_finite(y_pred), y_pred, tf.zeros_like(y_pred))
        
        # 1. 计算MSE损失
        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        
        # 确保MSE损失是有限的
        mse_loss = tf.where(tf.math.is_finite(mse_loss), mse_loss, tf.cast(1.0, self.emsc_compute_dtype))
        
        if gate_params is None:
            return mse_loss
        
        # 2. 获取门控参数序列并确保正确的dtype
        alpha_seq = tf.cast(gate_params['alpha'], self.emsc_compute_dtype)  # (batch_size, seq_len, state_dim)
        beta_seq = tf.cast(gate_params['beta'], self.emsc_compute_dtype)    # (batch_size, seq_len, state_dim)
        gamma_seq = tf.cast(gate_params['gamma'], self.emsc_compute_dtype)  # (batch_size, seq_len, state_dim)
        
        # 3. 计算门控参数在序列上的平均值
        mean_alpha = tf.reduce_mean(alpha_seq, axis=1)  # (batch_size, state_dim)
        mean_beta = tf.reduce_mean(beta_seq, axis=1)    # (batch_size, state_dim)
        mean_gamma = tf.reduce_mean(gamma_seq, axis=1)  # (batch_size, state_dim)
        
        # 4. 计算正则化损失
        reg_alpha_beta = tf.reduce_mean(tf.square(mean_alpha - mean_beta))
        reg_alpha_gamma = tf.reduce_mean(tf.square(mean_alpha - mean_gamma))
        reg_loss = (reg_alpha_beta + reg_alpha_gamma) / self.state_dim
        
        # 5. 计算动态正则化权重
        # 1000个epoch内从1e-3降到1e-6，之后恒为1e-6
        omega = tf.maximum(
            tf.cast(1e-6, self.emsc_compute_dtype), 
            tf.cast(1e-3, self.emsc_compute_dtype) - 
            tf.cast(9.99e-7, self.emsc_compute_dtype) * tf.cast(self.epoch, self.emsc_compute_dtype)
        )
        
        # 6. 计算总损失
        total_loss = mse_loss + omega * reg_loss
        
        # 7. 更新epoch计数
        self.epoch.assign_add(1)
        
        return total_loss
    
    def get_config(self):
        """获取配置，用于序列化"""
        try:
            config = super().get_config()
        except AttributeError:
            # 兼容旧版本TensorFlow
            config = {}
        config.update({
            'state_dim': self.state_dim
        })
        return config


class MaskedMSELoss(tf.keras.losses.Loss):
    """
    自定义掩码MSE损失类
    
    用于处理带掩码的序列数据，可以忽略特定位置的损失计算
    """
    def __init__(self, name='masked_mse_loss', **kwargs):
        # 兼容不同TensorFlow版本的构造函数
        try:
            # 尝试使用reduction参数（新版本）
            super().__init__(reduction=tf.keras.losses.Reduction.AUTO, name=name)
        except TypeError:
            # 如果不支持reduction参数，则使用旧版本的构造函数
            super().__init__(name=name)
        
        # 获取当前策略的dtype（使用完全自定义的属性名避免冲突）
        mixed_precision_policy = tf.keras.mixed_precision.global_policy()
        self.emsc_compute_dtype = mixed_precision_policy.compute_dtype
        self.emsc_variable_dtype = mixed_precision_policy.variable_dtype
    
    def call(self, y_true, y_pred, sample_weight=None):
        """
        计算带掩码的MSE损失
        
        参数:
        y_true: 真实值
        y_pred: 预测值
        sample_weight: 样本权重（掩码）
        """
        # 确保输入使用正确的dtype
        y_true = tf.cast(y_true, self.emsc_compute_dtype)
        y_pred = tf.cast(y_pred, self.emsc_compute_dtype)
        
        # 计算MSE
        mse = tf.keras.losses.mean_squared_error(y_true, y_pred)
        
        # 如果提供了sample_weight（掩码），应用它
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.emsc_compute_dtype)
            # 确保掩码形状与预测值匹配
            sample_weight = tf.reshape(sample_weight, tf.shape(mse))
            mse = mse * sample_weight
            # 计算有效样本的平均损失
            return tf.reduce_sum(mse) / tf.reduce_sum(sample_weight)
        
        return tf.reduce_mean(mse)
    
    def get_config(self):
        """获取配置，用于序列化"""
        try:
            base_config = super().get_config()
        except AttributeError:
            # 兼容旧版本TensorFlow
            base_config = {}
        return base_config 