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
    
    def call(self, y_true, y_pred, gate_params=None):
        """
        计算损失值
        
        参数:
        y_true: 真实值 (batch_size, seq_len, 1)
        y_pred: 预测值 (batch_size, seq_len, 1)
        gate_params: 门控参数字典，包含 'alpha', 'beta', 'gamma' 序列
        """
        # 确保输入数据类型一致且数值稳定
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # 检查输入是否包含NaN或Inf
        y_true = tf.where(tf.math.is_finite(y_true), y_true, tf.zeros_like(y_true))
        y_pred = tf.where(tf.math.is_finite(y_pred), y_pred, tf.zeros_like(y_pred))
        
        # 1. 计算MSE损失，添加数值稳定性保护
        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        mse_loss = tf.where(tf.math.is_finite(mse_loss), mse_loss, tf.constant(0.0, dtype=tf.float32))
        
        if gate_params is None:
            return mse_loss
        
        # 2. 获取门控参数序列，确保数值稳定
        alpha_seq = tf.cast(gate_params['alpha'], tf.float32)  # (batch_size, seq_len, state_dim)
        beta_seq = tf.cast(gate_params['beta'], tf.float32)    # (batch_size, seq_len, state_dim)
        gamma_seq = tf.cast(gate_params['gamma'], tf.float32)  # (batch_size, seq_len, state_dim)
        
        # 检查门控参数是否有效
        alpha_seq = tf.where(tf.math.is_finite(alpha_seq), alpha_seq, tf.zeros_like(alpha_seq))
        beta_seq = tf.where(tf.math.is_finite(beta_seq), beta_seq, tf.zeros_like(beta_seq))
        gamma_seq = tf.where(tf.math.is_finite(gamma_seq), gamma_seq, tf.zeros_like(gamma_seq))
        
        # 3. 计算门控参数在序列上的平均值
        mean_alpha = tf.reduce_mean(alpha_seq, axis=1)  # (batch_size, state_dim)
        mean_beta = tf.reduce_mean(beta_seq, axis=1)    # (batch_size, state_dim)
        mean_gamma = tf.reduce_mean(gamma_seq, axis=1)
        
        # 4. 改进的正则化损失计算 - 使用多种正则化策略
        # a) 门控参数差异正则化（降低权重）
        reg_alpha_beta = tf.reduce_mean(tf.square(mean_alpha - mean_beta))
        reg_alpha_gamma = tf.reduce_mean(tf.square(mean_alpha - mean_gamma))
        reg_beta_gamma = tf.reduce_mean(tf.square(mean_beta - mean_gamma))
        gate_diversity_loss = (reg_alpha_beta + reg_alpha_gamma + reg_beta_gamma) / 3.0
        
        # b) 门控参数范围正则化（防止过度激活）
        alpha_range_loss = tf.reduce_mean(tf.maximum(0.0, mean_alpha - 5.0))  # 限制alpha不要太大
        beta_range_loss = tf.reduce_mean(tf.maximum(0.0, mean_beta - 5.0))
        gamma_range_loss = tf.reduce_mean(tf.maximum(0.0, mean_gamma - 5.0))
        range_loss = alpha_range_loss + beta_range_loss + gamma_range_loss
        
        # c) 门控参数平滑性正则化
        alpha_smooth = tf.reduce_mean(tf.square(alpha_seq[:, 1:, :] - alpha_seq[:, :-1, :]))
        beta_smooth = tf.reduce_mean(tf.square(beta_seq[:, 1:, :] - beta_seq[:, :-1, :]))
        gamma_smooth = tf.reduce_mean(tf.square(gamma_seq[:, 1:, :] - gamma_seq[:, :-1, :]))
        smooth_loss = (alpha_smooth + beta_smooth + gamma_smooth) / 3.0
        
        # 安全除法，避免除零
        state_dim_safe = tf.maximum(tf.cast(self.state_dim, tf.float32), 1e-7)
        gate_diversity_loss = gate_diversity_loss / state_dim_safe
        range_loss = range_loss / state_dim_safe
        smooth_loss = smooth_loss / state_dim_safe
        
        # 确保正则化损失数值稳定
        gate_diversity_loss = tf.where(tf.math.is_finite(gate_diversity_loss), gate_diversity_loss, tf.constant(0.0, dtype=tf.float32))
        range_loss = tf.where(tf.math.is_finite(range_loss), range_loss, tf.constant(0.0, dtype=tf.float32))
        smooth_loss = tf.where(tf.math.is_finite(smooth_loss), smooth_loss, tf.constant(0.0, dtype=tf.float32))
        
        # 5. 改进的动态正则化权重 - 更平缓的衰减
        epoch_float = tf.cast(self.epoch, tf.float32)
        # 限制epoch值，避免数值溢出
        epoch_float = tf.clip_by_value(epoch_float, 0.0, 1e6)
        
        # 改进的权重计算：使用分段函数，前期权重较高，后期逐渐衰减但不会过快归零
        if epoch_float < 100:
            # 前100个epoch保持较高的正则化权重
            omega_diversity = 1e-2
            omega_range = 5e-3
            omega_smooth = 1e-3
        elif epoch_float < 500:
            # 100-500 epoch 逐渐降低
            progress = (epoch_float - 100) / 400
            omega_diversity = 1e-2 * (1 - 0.7 * progress)  # 从1e-2衰减到3e-3
            omega_range = 5e-3 * (1 - 0.6 * progress)      # 从5e-3衰减到2e-3
            omega_smooth = 1e-3 * (1 - 0.5 * progress)     # 从1e-3衰减到5e-4
        else:
            # 500+ epoch 保持最小正则化权重，避免完全归零
            omega_diversity = 3e-3
            omega_range = 2e-3
            omega_smooth = 5e-4
        
        # 额外保护
        omega_diversity = tf.clip_by_value(omega_diversity, 1e-6, 1e-1)
        omega_range = tf.clip_by_value(omega_range, 1e-6, 1e-1)
        omega_smooth = tf.clip_by_value(omega_smooth, 1e-6, 1e-1)
        
        # 6. 计算总损失，添加最终数值检查
        total_reg_loss = (omega_diversity * gate_diversity_loss + 
                         omega_range * range_loss + 
                         omega_smooth * smooth_loss)
        
        total_loss = mse_loss + total_reg_loss
        total_loss = tf.where(tf.math.is_finite(total_loss), total_loss, mse_loss)
        
        # 确保损失值在合理范围内
        total_loss = tf.clip_by_value(total_loss, 1e-8, 1e8)
        
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
    
    def call(self, y_true, y_pred, sample_weight=None):
        """
        计算带掩码的MSE损失
        
        参数:
        y_true: 真实值
        y_pred: 预测值
        sample_weight: 样本权重（掩码）
        """
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
        try:
            base_config = super().get_config()
        except AttributeError:
            # 兼容旧版本TensorFlow
            base_config = {}
        return base_config 