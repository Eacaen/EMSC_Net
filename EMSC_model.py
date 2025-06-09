"""
EMSC模型定义模块
包含所有模型相关的类定义和构建函数
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

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
        """计算增量方向向量"""
        delta_norm = tf.sqrt(tf.reduce_sum(tf.square(delta_features), axis=-1, keepdims=True))
        delta_norm = tf.maximum(delta_norm, 1e-7)  # 避免除零
        direction = delta_features / delta_norm
        return direction
    
    def process_internal_layers(self, l_0):
        """处理内部层 (tanh∘tanh 逐层)"""
        l = l_0
        for W_a, W_b in self.internal_layers:
            l_a = W_a(l)
            l_b = W_b(l)
            l = l_a * l_b
        return l
    
    def calc_gate_params(self, l_d):
        """计算门控参数 (exp激活确保非负)"""
        alpha = tf.exp(self.W_alpha(l_d))  # 应变增量门控
        beta = tf.exp(self.W_beta(l_d))    # 时间增量门控
        gamma = tf.exp(self.W_gamma(l_d))  # 温度门控
        return alpha, beta, gamma
    
    def calc_update_gate(self, alpha, beta, gamma, delta_features):
        """计算更新门"""
        delta_strain = delta_features[..., 0:1]
        delta_time = delta_features[..., 1:2]
        delta_temperature = delta_features[..., 2:3]
        
        z = 1 - tf.exp(-alpha * tf.abs(delta_strain) - 
                       beta * delta_time - 
                       gamma * tf.abs(delta_temperature))
        return z
    
    def reconstruct_temp_seq(self, delta_x):
        """重建温度序列"""
        delta_features = delta_x[..., :3]
        init_features = delta_x[..., 3:]
        
        init_temp = init_features[..., 0, 2:3]
        delta_temp = delta_features[..., 2:3]
        
        temp_seq = tf.cumsum(delta_temp, axis=1)
        temp_seq = temp_seq + init_temp
        
        return temp_seq

    def call(self, inputs):
        """前向传播"""
        h_prev, delta_x = inputs
        
        delta_features = delta_x[..., :3]
        temp_seq = self.reconstruct_temp_seq(delta_x)
        direction = self.calc_direction_vec(delta_features)
        
        l_0 = tf.concat([h_prev, temp_seq, direction], axis=-1)
        l_d = self.process_internal_layers(l_0)
        
        alpha, beta, gamma = self.calc_gate_params(l_d)
        c_n = tf.tanh(self.W_c(l_d))
        z = self.calc_update_gate(alpha, beta, gamma, delta_features)
        
        h_n = (1 - z) * h_prev + z * c_n
        sigma_n = self.W_out(h_n)
        
        gate_params = {
            'alpha': alpha,
            'beta': beta,
            'gamma': gamma
        }
        
        return h_n, sigma_n, gate_params

class MSC_Sequence(tf.keras.layers.Layer):
    """
    EMSC 序列处理层
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
        """处理输入序列"""
        delta_seq, state_0 = inputs
        
        def step_fn(t, state, outputs):
            delta_x_t = delta_seq[:, t, :]
            new_state, output, gate_params = self.msc_cell([state, delta_x_t])
            outputs = outputs.write(t, output)
            return t + 1, new_state, outputs
        
        outputs = tf.TensorArray(dtype=tf.float32, size=tf.shape(delta_seq)[1])
        
        _, final_state, outputs = tf.while_loop(
            lambda t, *_: t < tf.shape(delta_seq)[1],
            step_fn,
            [tf.constant(0), state_0, outputs]
        )
        
        return tf.transpose(outputs.stack(), [1, 0, 2])

def build_msc_model(state_dim=8, input_dim=6, output_dim=1,
                   hidden_dim=32, num_internal_layers=2):
    """
    构建 EMSC 模型
    
    参数:
    state_dim: 状态向量维度
    input_dim: 输入特征维度
    output_dim: 输出维度
    hidden_dim: 内部层维度
    num_internal_layers: 内部层数量
    """
    delta_input = Input(shape=(None, input_dim), name='delta_input')
    init_state = Input(shape=(state_dim,), name='init_state')
    
    state_seq = MSC_Sequence(
        state_dim=state_dim,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_internal_layers=num_internal_layers
    )([delta_input, init_state])
    
    stress_out = Dense(output_dim, name='stress_out')(state_seq)
    
    return Model(inputs=[delta_input, init_state], outputs=stress_out)