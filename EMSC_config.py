"""
EMSC配置模块
包含所有配置相关的函数和参数
"""

import os
import json
import argparse

def create_training_config(state_dim=8, input_dim=6, hidden_dim=32, learning_rate=1e-3, 
                         target_sequence_length=5000, window_size=None, stride=None, 
                         max_subsequences=200, train_test_split_ratio=0.8, random_seed=42,
                         epochs=500, batch_size=8, save_frequency=1):
    """
    创建训练配置字典
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

def save_training_config(config, save_path):
    """保存训练配置"""
    try:
        config_file = os.path.join(save_path, 'training_config.json')
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Training configuration saved to: {config_file}")
    except Exception as e:
        print(f"Error saving training config: {e}")

def load_training_config(save_path):
    """加载训练配置"""
    try:
        config_file = os.path.join(save_path, 'training_config.json')
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

def parse_training_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='EMSC模型训练参数')
    parser.add_argument('--epochs', type=int, default=2000, 
                       help='训练轮数 (默认: 2000)')
    parser.add_argument('--save_frequency', type=int, default=10, 
                       help='模型保存频率，每N个epoch保存一次 (默认: 10)')
    parser.add_argument('--dataset', type=str, default='dataset', 
                       help='数据集名称 (默认: dataset)')
    parser.add_argument('--resume', action='store_true', default=True,
                       help='是否从检查点恢复训练 (默认: True)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='批次大小 (默认: None)')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='学习率 (默认: 1e-3)')
    parser.add_argument('--state_dim', type=int, default=8,
                       help='状态向量维度 (默认: 8)')
    parser.add_argument('--hidden_dim', type=int, default=32,
                       help='隐藏层维度 (默认: 32)')
    parser.add_argument('--mixed_precision', type=str, choices=['true', 'false', 'auto'], default='auto',
                       help='是否启用混合精度训练 (选项: true, false, auto，默认: auto)')
    
    return parser.parse_args()

def get_dataset_paths(dataset_name='dataset'):
    """获取数据集路径"""
    dataset_dir_aliyun = "/mnt/data/msc_models/"
    dataset_dir_local = "/Users/tianyunhu/Documents/temp/code/Test_app/EMSC_Model/msc_models"
    
    dataset_dir_aliyun = os.path.join(dataset_dir_aliyun, dataset_name)
    dataset_dir_local = os.path.join(dataset_dir_local, dataset_name)
    
    if os.path.exists(dataset_dir_aliyun):
        dataset_dir = dataset_dir_aliyun
    else:
        dataset_dir = dataset_dir_local
    
    model_name = 'msc_model'
    best_model_name = 'best_msc_model'
    dataset_path = os.path.join(dataset_dir, f'{dataset_name}.npz')
    
    return {
        'dataset_dir': dataset_dir,
        'model_name': model_name,
        'best_model_name': best_model_name,
        'dataset_path': dataset_path
    }