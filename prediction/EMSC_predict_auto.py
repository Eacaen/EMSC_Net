#!/usr/bin/env python3
"""
EMSC自动参数预测脚本
自动从模型和配置文件中推断必要的参数
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib
import os
import json
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.EMSC_model import MSC_Sequence
from core.EMSC_losses import EMSCLoss, MaskedMSELoss

def auto_load_model_and_config(model_path):
    """
    自动加载模型并推断所有必要参数
    
    Args:
        model_path: 模型路径（可以是文件夹或.h5文件）
    
    Returns:
        dict: 包含模型、标准化器和参数的字典
    """
    result = {
        'model': None,
        'x_scaler': None, 
        'y_scaler': None,
        'state_dim': 8,      # 默认值
        'input_dim': 6,      # 默认值
        'window_size': 5000, # 默认值
        'network_structure': None
    }
    
    print(f"🔍 自动加载模型: {model_path}")
    
    # 1. 确定模型目录和文件
    if os.path.isfile(model_path):
        # 如果是文件（如.h5），则父目录是模型目录
        model_dir = os.path.dirname(model_path)
        model_file = model_path
    else:
        # 如果是目录，则需要判断是SavedModel目录还是包含模型的目录
        if os.path.isdir(model_path):
            # 检查是否是SavedModel目录（包含saved_model.pb）
            saved_model_pb = os.path.join(model_path, 'saved_model.pb')
            if os.path.exists(saved_model_pb):
                # 这是SavedModel目录
                model_dir = os.path.dirname(model_path)
                model_file = model_path
            else:
                # 这是包含模型的目录，查找模型文件/目录
                model_dir = model_path
                possible_models = [
                    os.path.join(model_dir, 'best_msc_model'),
                    os.path.join(model_dir, 'msc_model'),
                    os.path.join(model_dir, 'best_msc_model.h5'),
                    os.path.join(model_dir, 'msc_model.h5')
                ]
                model_file = None
                for candidate in possible_models:
                    if os.path.exists(candidate):
                        model_file = candidate
                        break
                
                if model_file is None:
                    raise FileNotFoundError(f"在 {model_dir} 中未找到模型文件")
        else:
            raise FileNotFoundError(f"路径不存在: {model_path}")
    
    print(f"📁 模型目录: {model_dir}")
    print(f"📄 模型文件: {os.path.basename(model_file)}")
    
    # 2. 从目录结构推断网络参数
    parent_dir = os.path.basename(model_dir)
    if parent_dir.startswith('network_'):
        # 解析网络结构 例如: network_6-32-32-8-1
        structure = parent_dir.replace('network_', '')
        parts = structure.split('-')
        if len(parts) == 5:
            input_layer, hidden1, hidden2, state_dim, output_layer = map(int, parts)
            result['state_dim'] = state_dim
            result['input_dim'] = input_layer
            result['network_structure'] = structure
            print(f"🏗️  从文件夹名推断网络结构: {structure}")
            print(f"   state_dim: {state_dim}, input_dim: {input_layer}")
    
    # 3. 尝试加载配置文件
    config_file = os.path.join(model_dir, 'training_config.json')
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # 从配置文件更新参数
            if 'STATE_DIM' in config:
                result['state_dim'] = config['STATE_DIM']
            if 'INPUT_DIM' in config:
                result['input_dim'] = config['INPUT_DIM']
                
            
            print(f"📋 从配置文件加载参数:")
            print(f"   state_dim: {result['state_dim']}")
            print(f"   input_dim: {result['input_dim']}")
            print(f"   window_size: {result['window_size']} (使用默认值)")
                
        except Exception as e:
            print(f"⚠️  配置文件读取失败: {e}")
    
    # 4. 加载模型
    try:
        with tf.keras.utils.custom_object_scope({
            'MSC_Sequence': MSC_Sequence,
            'EMSCLoss': EMSCLoss,
            'MaskedMSELoss': MaskedMSELoss
        }):
            result['model'] = tf.keras.models.load_model(
                model_file,
                custom_objects={
                    'MSC_Sequence': MSC_Sequence,
                    'EMSCLoss': EMSCLoss,
                    'MaskedMSELoss': MaskedMSELoss
                }
            )
        print("✅ 模型加载成功")
        
        # 5. 从模型结构验证/推断参数
        try:
            # 获取模型输入形状
            input_shapes = [layer.input_shape for layer in result['model'].layers if hasattr(layer, 'input_shape')]
            
            # 查找init_state输入来推断state_dim
            for layer in result['model'].layers:
                if hasattr(layer, 'input_shape') and isinstance(layer.input_shape, list):
                    for shape in layer.input_shape:
                        if shape and len(shape) == 2 and shape[1] is not None:
                            # 这可能是init_state的形状 [batch_size, state_dim]
                            inferred_state_dim = shape[1]
                            if inferred_state_dim != result['state_dim']:
                                print(f"🔄 从模型结构修正state_dim: {result['state_dim']} -> {inferred_state_dim}")
                                result['state_dim'] = inferred_state_dim
                            break
            
            print(f"🏗️  最终模型参数:")
            print(f"   state_dim: {result['state_dim']}")
            print(f"   input_dim: {result['input_dim']}")
            
        except Exception as e:
            print(f"⚠️  模型结构分析失败: {e}")
            
    except Exception as e:
        raise RuntimeError(f"模型加载失败: {e}")
    
    # 6. 加载标准化器
    scaler_locations = [
        model_dir,  # 模型同目录
        os.path.join(os.path.dirname(model_dir), 'scalers'),  # scalers子目录
        os.path.join(model_dir, 'scalers'),  # 模型目录下的scalers
    ]
    
    for scaler_dir in scaler_locations:
        x_scaler_path = os.path.join(scaler_dir, 'x_scaler.save')
        y_scaler_path = os.path.join(scaler_dir, 'y_scaler.save')
        
        if os.path.exists(x_scaler_path) and os.path.exists(y_scaler_path):
            try:
                result['x_scaler'] = joblib.load(x_scaler_path)
                result['y_scaler'] = joblib.load(y_scaler_path)
                print(f"✅ 标准化器加载成功: {scaler_dir}")
                break
            except Exception as e:
                print(f"⚠️  标准化器加载失败 {scaler_dir}: {e}")
                continue
    
    if result['x_scaler'] is None or result['y_scaler'] is None:
        print("❌ 未找到标准化器文件，预测可能不准确")
        print("💡 建议检查以下位置的标准化器文件:")
        for loc in scaler_locations:
            print(f"   - {loc}/x_scaler.save")
            print(f"   - {loc}/y_scaler.save")
    
    return result

def smart_predict(model_info, strain_sequence, temperature, time_sequence=None):
    """
    智能预测函数，自动使用推断的参数
    
    Args:
        model_info: auto_load_model_and_config返回的信息字典
        strain_sequence: 应变序列
        temperature: 温度
        time_sequence: 时间序列（可选）
    
    Returns:
        predicted_stress: 预测应力
        time_sequence: 时间序列
    """
    print(f"🚀 开始预测 (序列长度: {len(strain_sequence)})")
    print(f"🌡️  温度: {temperature}°C")
    print(f"📏 使用参数: state_dim={model_info['state_dim']}, input_dim={model_info['input_dim']}, window_size={model_info['window_size']}")
    
    if model_info['x_scaler'] is None or model_info['y_scaler'] is None:
        raise ValueError("缺少标准化器，无法进行预测")
    
    # 使用已有的预测函数
    from prediction.EMSC_predict import predict_stress
    
    return predict_stress(
        model=model_info['model'],
        x_scaler=model_info['x_scaler'],
        y_scaler=model_info['y_scaler'],
        strain_sequence=strain_sequence,
        temperature=temperature,
        time_sequence=time_sequence,
        state_dim=model_info['state_dim'],
        input_dim=model_info['input_dim'],
        window_size=model_info['window_size']
    )

def find_best_model():
    """
    自动寻找最佳训练模型
    
    Returns:
        str: 最佳模型的路径
    """
    base_paths = [
        '/Users/tianyunhu/Documents/temp/code/Test_app/EMSC_Model/msc_models/dataset_EMSC_big',
        'models/dataset_EMSC_big',
        '../models/dataset_EMSC_big'
    ]
    
    best_models = []
    
    for base_path in base_paths:
        if not os.path.exists(base_path):
            continue
            
        print(f"🔍 搜索模型: {base_path}")
        
        # 查找所有网络文件夹
        for item in os.listdir(base_path):
            if item.startswith('network_'):
                model_dir = os.path.join(base_path, item)
                if os.path.isdir(model_dir):
                    # 检查是否有best模型
                    best_model_path = os.path.join(model_dir, 'best_msc_model')
                    if os.path.exists(best_model_path):
                        best_models.append({
                            'path': best_model_path,
                            'structure': item.replace('network_', ''),
                            'dir': model_dir
                        })
                        print(f"  ✅ 找到: {item}")
    
    if not best_models:
        raise FileNotFoundError("未找到任何训练好的模型")
    
    # 按网络结构排序，选择标准配置或最大的网络
    best_models.sort(key=lambda x: x['structure'])
    
    # 优先选择标准配置 6-32-32-8-1
    for model in best_models:
        if model['structure'] == '6-8-8-8-1':
            print(f"🎯 选择标准配置: {model['structure']}")
            return model['path']
    
    # 否则选择第一个
    selected = best_models[0]
    print(f"🎯 选择模型: {selected['structure']}")
    return selected['path']

def main():
    """主函数 - 简化的使用接口"""
    
    # 1. 自动找到最佳模型
    try:
        model_path = find_best_model()
    except Exception as e:
        print(f"❌ {e}")
        print("💡 请手动指定模型路径")
        return
    
    # 2. 自动加载模型和配置
    try:
        model_info = auto_load_model_and_config(model_path)
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    print(f"\n{'='*50}")
    print(f"🎉 模型加载完成!")
    if model_info['network_structure']:
        print(f"📊 网络结构: {model_info['network_structure']}")
    print(f"📏 模型参数: state_dim={model_info['state_dim']}, input_dim={model_info['input_dim']}")
    print(f"🪟 窗口大小: {model_info['window_size']}")
    print(f"{'='*50}\n")
    
    # 3. 加载实验数据
    file_path = '/Users/tianyunhu/Documents/temp/CTC/PPCC/'
    import glob
    file_list = glob.glob(os.path.join(file_path, "*.xlsx"))
    
    if not file_list:
        print("❌ 未找到实验数据文件")
        return
    
    # 随机选择一个文件进行测试
    selected_file = np.random.choice(file_list)
    print(f"📁 测试文件: {os.path.basename(selected_file)}")
    
    from prediction.EMSC_predict import load_experimental_data, calculate_error_metrics, plot_results
    
    exp_strain, exp_stress, exp_time, temperature = load_experimental_data(selected_file)
    
    if exp_strain is None:
        print("❌ 实验数据加载失败")
        return
    
    # 4. 进行预测（现在不需要手动指定参数了！）
    try:
        predicted_stress, time_sequence = smart_predict(
            model_info=model_info,
            strain_sequence=exp_strain,
            temperature=temperature,
            time_sequence=exp_time
        )
        
        print("✅ 预测完成!")
        
    except Exception as e:
        print(f"❌ 预测失败: {e}")
        return
    
    # 5. 计算误差和绘图
    error_metrics = calculate_error_metrics(
        predicted_stress, exp_strain, exp_stress, exp_strain
    )
    
    plot_results(
        strain_sequence=exp_strain,
        predicted_stress=predicted_stress,
        time_sequence=time_sequence,
        temperature=temperature,
        exp_strain=exp_strain,
        exp_stress=exp_stress,
        exp_time=exp_time,
        error_metrics=error_metrics
    )
    
    print(f"\n🎯 预测结果总结:")
    print(f"   R²: {error_metrics['R2']:.4f}")
    print(f"   RMSE: {error_metrics['RMSE']:.2f} MPa")
    print(f"   MAE: {error_metrics['MAE']:.2f} MPa")

if __name__ == '__main__':
    main() 