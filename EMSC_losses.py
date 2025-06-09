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
    def __init__(self, state_dim=8, name='emsc_loss'):
        super().__init__(name=name)
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
        # 1. 计算MSE损失
        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        
        if gate_params is None:
            return mse_loss
        
        # 2. 获取门控参数序列
        alpha_seq = gate_params['alpha']  # (batch_size, seq_len, state_dim)
        beta_seq = gate_params['beta']    # (batch_size, seq_len, state_dim)
        gamma_seq = gate_params['gamma']  # (batch_size, seq_len, state_dim)
        
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
        omega = tf.maximum(1e-6, 1e-3 - 9.99e-7 * tf.cast(self.epoch, tf.float32))
        
        # 6. 计算总损失
        total_loss = mse_loss + omega * reg_loss
        
        # 7. 更新epoch计数
        self.epoch.assign_add(1)
        
        return total_loss
    
    def get_config(self):
        """获取配置，用于序列化"""
        config = super().get_config()
        config.update({
            'state_dim': self.state_dim
        })
        return config


class MaskedMSELoss(tf.keras.losses.Loss):
    """
    自定义掩码MSE损失类
    
    用于处理带掩码的序列数据，可以忽略特定位置的损失计算
    """
    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO, name='masked_mse_loss'):
        super().__init__(reduction=reduction, name=name)
    
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
        base_config = super().get_config()
        return base_config 