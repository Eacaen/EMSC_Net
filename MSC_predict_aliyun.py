import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib
import os
from MSC_train_aliyun import MSC_Sequence
from EMSC_losses import EMSCLoss, MaskedMSELoss  # 从新模块导入损失函数

# 定义列名映射
column_mapping = {
    'time': 'time',
    'true_strain': 'true_strain',
    'true_stress': 'true_stress',
    'temperature': 'temperature',
    'delta_strain': 'delta_strain',  # 原 Δε
    'delta_time': 'delta_time',      # 原 Δt
    'delta_temperature': 'delta_temperature',  # 原 ΔT
    'init_strain': 'init_strain',    # 初始应变
    'init_time': 'init_time',        # 初始时间
    'init_temp': 'init_temp'         # 初始温度
}

# 加载模型和数据标准化器
def load_trained_model(model_dir='msc_model', 
                      x_scaler_path='x_scaler.save',
                      y_scaler_path='y_scaler.save',
                      state_dim=8,
                      input_dim=6):
    """
    加载SavedModel格式的模型和数据标准化器
    
    参数:
    model_dir: SavedModel格式模型目录
    x_scaler_path: 输入数据标准化器路径
    y_scaler_path: 输出数据标准化器路径
    state_dim: 状态向量维度
    input_dim: 输入特征维度
    
    返回:
    model: 加载的模型
    x_scaler: 输入数据标准化器
    y_scaler: 输出数据标准化器
    """
    try:
        print(f"Loading model from directory: {model_dir}")
        with tf.keras.utils.custom_object_scope({
            'MSC_Sequence': MSC_Sequence,
            'EMSCLoss': EMSCLoss,
            'MaskedMSELoss': MaskedMSELoss
        }):
            model = tf.keras.models.load_model(
                model_dir,
                custom_objects={
                    'MSC_Sequence': MSC_Sequence,
                    'MaskedMSELoss': MaskedMSELoss
                }
            )
        print("Model loaded successfully")
        
        print(f"Loading scalers from: {x_scaler_path} and {y_scaler_path}")
        x_scaler = joblib.load(x_scaler_path)
        y_scaler = joblib.load(y_scaler_path)
        print("Scalers loaded successfully")
        
        return model, x_scaler, y_scaler
    except Exception as e:
        print(f"Error loading model or scalers: {e}")
        raise

def prepare_input_data(strain_sequence, temperature, time_sequence=None, state_dim=8, input_dim=6, x_scaler=None):
    """
    准备模型输入数据
    
    参数:
    strain_sequence: 应变序列 (numpy array 或 list)
    temperature: 温度值 (标量)
    time_sequence: 时间序列 (可选，如果为None则自动生成均匀时间步)
    state_dim: 状态向量维度
    input_dim: 输入特征维度
    x_scaler: 输入数据标准化器
    
    返回:
    X_seq: 预处理后的输入序列
    init_state: 初始状态向量
    mask: 全1掩码，表示所有数据点有效
    """
    if time_sequence is None:
        time_sequence = np.linspace(0, len(strain_sequence)-1, len(strain_sequence))
    
    # 计算应变增量
    delta_strain = np.diff(strain_sequence, prepend=strain_sequence[0])
    # 计算时间增量
    delta_time = np.diff(time_sequence, prepend=time_sequence[0])
    # 确保时间增量不为0
    delta_time = np.maximum(delta_time, 1e-5)
    
    # 获取初始值
    init_strain = strain_sequence[0]
    init_time = time_sequence[0]
    init_temp = temperature
    
    # 构建输入特征矩阵 [Δε, Δt, ΔT, ε0, t0, T0]
    X = np.column_stack([
        delta_strain, 
        delta_time, 
        np.zeros_like(delta_strain),  # ΔT = 0 (温度恒定)
        np.full_like(delta_strain, init_strain),
        np.full_like(delta_strain, init_time),
        np.full_like(delta_strain, init_temp)
    ])
    
    # 使用保存的标准化器进行数据标准化
    X_norm = x_scaler.transform(X)
    
    # 准备序列输入
    X_seq = X_norm.reshape(1, -1, input_dim)  # 添加batch维度和特征维度
    init_state = np.zeros((1, state_dim))  # 添加batch维度
    mask = np.ones((1, X_norm.shape[0]), dtype=np.float32)  # 全1掩码，表示所有数据点有效
    
    return X_seq, init_state, mask

def extract_strain_rate_from_filename(filename):
    """从文件名中提取应变率信息"""
    try:
        strain_rate = os.path.splitext(filename)[0].split('_')[2]
        return float(strain_rate)
    except (IndexError, ValueError) as e:
        print(f"警告: 无法从文件名 {filename} 提取应变率信息: {e}")
        return None

def load_experimental_data(file_path):
    """
    加载实验数据
    
    参数:
    file_path: Excel文件路径
    
    返回:
    strain: 应变序列
    stress: 应力序列
    time: 时间序列
    temperature: 温度值
    """
    try:
        df = pd.read_excel(file_path)
        df = df.rename(columns=lambda x: x.strip().lower())
        
        # 检查必要的列是否存在
        required_columns = {'time', 'true_strain', 'true_stress', 'temperature'}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"Excel文件缺少必要的列: {required_columns}")
        
        # 提取数据
        strain = df[column_mapping['true_strain']].values
        stress = df[column_mapping['true_stress']].values
        time = df[column_mapping['time']].values
        temperature = df[column_mapping['temperature']].values[0]  # 取第一个温度值
        
        # 如果文件名包含'com'，将应力和应变取反
        if 'com' in os.path.basename(file_path).lower():
            strain = -strain
            stress = -stress
            print(f"检测到压缩数据文件，已将应力和应变取反")
        
        # 提取应变率信息
        strain_rate = extract_strain_rate_from_filename(file_path)
        if strain_rate is not None:
            print(f"检测到应变率: {strain_rate:.2e} s⁻¹")
        
        return strain, stress, time, temperature
    except Exception as e:
        print(f"加载实验数据时出错: {e}")
        return None, None, None, None

def predict_with_sliding_window(model, x_scaler, y_scaler, strain_sequence, temperature, 
                              time_sequence=None, window_size=1000, state_dim=8, input_dim=6):
    """
    使用滑动窗口进行预测，处理任意长度的输入序列
    
    参数:
    model: 加载的模型
    x_scaler, y_scaler: 数据标准化器
    strain_sequence: 应变序列
    temperature: 温度值
    time_sequence: 时间序列（可选）
    window_size: 窗口大小，默认为训练时使用的 WINDOW_SIZE
    state_dim: 状态向量维度
    input_dim: 输入特征维度
    
    返回:
    predicted_stress: 预测的应力值
    time_sequence: 使用的时间序列
    """
    if time_sequence is None:
        time_sequence = np.linspace(0, len(strain_sequence)-1, len(strain_sequence))
    
    sequence_length = len(strain_sequence)
    
    # 如果序列长度小于窗口大小，直接预测
    if sequence_length <= window_size:
        # 准备输入数据
        X_seq, init_state, mask = prepare_input_data(
            strain_sequence, temperature, time_sequence,
            state_dim=state_dim, input_dim=input_dim, x_scaler=x_scaler
        )
        
        # 进行预测
        y_pred_norm = model.predict({
            'delta_input': X_seq, 
            'init_state': init_state
        })
        
        # 反标准化预测结果
        predicted_stress = y_scaler.inverse_transform(y_pred_norm.reshape(-1, 1))
        predicted_stress = predicted_stress[:sequence_length]  # 确保只返回有效长度
        
    else:
        # 使用滑动窗口进行预测
        predicted_stress = np.zeros(sequence_length)  # 创建结果数组
        counts = np.zeros(sequence_length)  # 用于记录每个位置的预测次数
        
        # 设置滑动窗口参数
        stride = window_size // 4  # 使用75%的重叠
        
        for start_idx in range(0, sequence_length, stride):
            end_idx = min(start_idx + window_size, sequence_length)
            current_window_size = end_idx - start_idx
            
            # 提取当前窗口的数据
            window_strain = strain_sequence[start_idx:end_idx]
            window_time = time_sequence[start_idx:end_idx]
            
            # 如果是最后一个窗口且长度不足，进行填充
            if len(window_strain) < window_size:
                pad_length = window_size - len(window_strain)
                window_strain = np.pad(window_strain, (0, pad_length), mode='edge')
                window_time = np.pad(window_time, (0, pad_length), mode='edge')
            
            # 准备窗口数据
            X_seq, init_state, mask = prepare_input_data(
                window_strain, temperature, window_time,
                state_dim=state_dim, input_dim=input_dim, x_scaler=x_scaler
            )
            
            # 预测当前窗口
            y_pred_norm = model.predict({
                'delta_input': X_seq, 
                'init_state': init_state
            })
            
            # 反标准化预测结果
            window_pred = y_scaler.inverse_transform(y_pred_norm.reshape(-1, 1)).flatten()
            
            # 如果预测结果长度超过当前窗口大小，截断
            if len(window_pred) > current_window_size:
                window_pred = window_pred[:current_window_size]
            
            # 使用三角形窗口权重
            weight = np.ones(len(window_pred))
            if start_idx > 0:  # 不是第一个窗口
                ramp_length = min(stride, len(weight))
                weight[:ramp_length] = np.linspace(0, 1, ramp_length)
            if end_idx < sequence_length:  # 不是最后一个窗口
                ramp_length = min(stride, len(weight))
                weight[-ramp_length:] = np.linspace(1, 0, ramp_length)
            
            # 累加加权预测结果
            predicted_stress[start_idx:start_idx + len(window_pred)] += window_pred * weight
            counts[start_idx:start_idx + len(window_pred)] += weight
        
        # 计算加权平均
        counts[counts == 0] = 1  # 避免除以零
        predicted_stress = predicted_stress / counts
        
        # 对结果进行轻微平滑处理
        from scipy.signal import savgol_filter
        try:
            window_length = min(11, len(predicted_stress) // 10)
            if window_length % 2 == 0:
                window_length += 1
            predicted_stress = savgol_filter(predicted_stress, window_length, 2)
        except Exception as e:
            print(f"Warning: Could not apply smoothing filter: {e}")
    
    return predicted_stress.reshape(-1, 1), time_sequence

def predict_stress(model, x_scaler, y_scaler, strain_sequence, temperature, 
                  time_sequence=None, state_dim=8, input_dim=6, window_size=1000):
    """
    使用模型进行应力预测
    
    参数:
    model: 加载的模型
    x_scaler, y_scaler: 数据标准化器
    strain_sequence: 应变序列
    temperature: 温度值
    time_sequence: 时间序列（可选）
    state_dim: 状态向量维度
    input_dim: 输入特征维度
    window_size: 窗口大小
    
    返回:
    predicted_stress: 预测的应力值
    time_sequence: 使用的时间序列
    """
    # 使用滑动窗口进行预测
    return predict_with_sliding_window(
        model, x_scaler, y_scaler,
        strain_sequence=strain_sequence,
        temperature=temperature,
        time_sequence=time_sequence,
        window_size=window_size,
        state_dim=state_dim,
        input_dim=input_dim
    )

def calculate_error_metrics(y_pred, exp_strain, exp_stress, strain_sequence):
    """
    计算预测误差指标
    
    参数:
    y_pred: 预测的应力值
    exp_strain: 实验应变数据
    exp_stress: 实验应力数据
    strain_sequence: 预测使用的应变序列
    
    返回:
    dict: 包含各种误差指标的字典
    """
    # 确保输入数据的形状正确
    y_pred = y_pred.reshape(-1)  # 将预测值转换为1D数组
    exp_stress = exp_stress.reshape(-1)  # 将实验值转换为1D数组
    strain_sequence = strain_sequence.reshape(-1)  # 将应变序列转换为1D数组
    
    # 对实验数据进行插值，使其与预测数据点对应
    from scipy.interpolate import interp1d
    f = interp1d(exp_strain, exp_stress, bounds_error=False, fill_value=(exp_stress[0], exp_stress[-1]))
    exp_stress_interp = f(strain_sequence)
    
    # 确保长度匹配
    min_len = min(len(y_pred), len(exp_stress_interp))
    y_pred = y_pred[:min_len]
    exp_stress_interp = exp_stress_interp[:min_len]
    
    # 计算误差指标
    mse = np.mean((y_pred - exp_stress_interp) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_pred - exp_stress_interp))
    r2 = 1 - np.sum((y_pred - exp_stress_interp) ** 2) / np.sum((exp_stress_interp - np.mean(exp_stress_interp)) ** 2)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'exp_stress_interp': exp_stress_interp
    }

def plot_results(strain_sequence, predicted_stress, time_sequence, temperature,
                exp_strain=None, exp_stress=None, exp_time=None,
                error_metrics=None):
    """
    绘制预测结果和实验数据对比图
    
    参数:
    strain_sequence: 应变序列
    predicted_stress: 预测的应力值
    time_sequence: 时间序列
    temperature: 温度值
    exp_strain: 实验应变数据（可选）
    exp_stress: 实验应力数据（可选）
    exp_time: 实验时间数据（可选）
    error_metrics: 误差指标字典（可选）
    """
    # 创建图形
    fig, (ax1) = plt.subplots(1, 1, figsize=(6, 4))
    
    # 绘制应力-应变曲线
    ax1.plot(strain_sequence, predicted_stress, 'b-', label='Predicted Stress', linewidth=2)
    if exp_strain is not None and exp_stress is not None:
        ax1.plot(exp_strain, exp_stress, 'r--', label='Experimental Data', linewidth=2)
    ax1.set_xlabel('True Strain')
    ax1.set_ylabel('True Stress (MPa)')
    ax1.set_title(f'Stress-Strain Curve at {temperature}°C')
    ax1.grid(True)
    ax1.legend()
    
    plt.tight_layout()
    plt.show()
    
    # 打印误差指标
    if error_metrics is not None:
        print("\n预测误差分析:")
        print(f"均方误差 (MSE): {error_metrics['MSE']:.2f}")
        print(f"均方根误差 (RMSE): {error_metrics['RMSE']:.2f}")
        print(f"平均绝对误差 (MAE): {error_metrics['MAE']:.2f}")
        print(f"决定系数 (R²): {error_metrics['R2']:.4f}")

if __name__ == '__main__':
    # 模型参数配置
    state_dim = 8
    input_dim = 6  # [delta_strain, delta_time, delta_temperature, init_strain, init_time, init_temp]
    window_size = 1000
    
    # 1. 加载模型和标准化器
    model_dir = '/Users/tianyunhu/Documents/temp/code/Test_app/EMSC_Model/msc_models/best_msc_model'  
    scaler_dir = '/Users/tianyunhu/Documents/temp/code/Test_app/EMSC_Model/msc_models/scalers'
    x_scaler_path = os.path.join(scaler_dir, 'x_scaler.save')
    y_scaler_path = os.path.join(scaler_dir, 'y_scaler.save')
    
    try:
        model, x_scaler, y_scaler = load_trained_model(
            model_dir, x_scaler_path, y_scaler_path,
            state_dim=state_dim,
            input_dim=input_dim
        )
    except Exception as e:
        print(f"Failed to load model or scalers: {e}")
        exit(1)
    
    # 2. 加载实验数据

    file_path = '/Users/tianyunhu/Documents/temp/CTC/PPCC/'
    # 随机选择一个文件
    import glob
    file_list = glob.glob(os.path.join(file_path, "*.xlsx"))
    if not file_list:
        print("未找到任何Excel文件")
        exit(1)
    file_path = np.random.choice(file_list)
    print(f"随机选择的文件: {os.path.basename(file_path)}")
    exp_strain, exp_stress, exp_time, temperature = load_experimental_data(file_path)
    
    if exp_strain is None:
        print("无法加载实验数据，程序退出")
        exit(1)
    
    # 使用实验数据的应变范围进行预测
    strain_sequence = exp_strain
    time_sequence = exp_time
    
    # 预测
    predicted_stress, time_sequence = predict_stress(
        model, x_scaler, y_scaler,
        strain_sequence=strain_sequence,
        time_sequence=time_sequence,
        temperature=temperature,
        state_dim=state_dim,
        input_dim=input_dim,
        window_size=window_size
    )
    
    # 计算误差指标
    error_metrics = calculate_error_metrics(
        predicted_stress, exp_strain, exp_stress, strain_sequence
    )
    
    # 绘图
    plot_results(
        strain_sequence=strain_sequence,
        predicted_stress=predicted_stress,
        time_sequence=time_sequence,
        temperature=temperature,
        exp_strain=exp_strain,
        exp_stress=exp_stress,
        exp_time=exp_time,
        error_metrics=error_metrics
    )
    
    # 打印预测结果示例
    print("\n预测结果示例（前5个点）:")
    print("应变\t实验应力(MPa)\t预测应力(MPa)")
    for i in range(min(5, len(strain_sequence))):
        print(f"{strain_sequence[i]:.4f}\t{exp_stress[i]:.2f}\t{predicted_stress[i][0]:.2f}") 