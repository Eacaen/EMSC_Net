"""
EMSC分阶段训练框架
实现从短序列到长序列的渐进式训练策略

训练阶段：
1. 阶段一：短序列（201点，200epochs）
2. 阶段二：中序列（1001点，400epochs）  
3. 阶段三：长序列（完整序列，2900epochs）
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence
import matplotlib.pyplot as plt
from datetime import datetime
import json
from typing import List, Tuple, Optional, Dict

# 导入现有模块
from EMSC_model import build_msc_model
from EMSC_data import load_dataset_from_npz
from EMSC_callbacks import MSCProgressCallback, create_early_stopping_callback
from EMSC_config import get_dataset_paths
from EMSC_losses import EMSCLoss
from EMSC_utils import print_training_summary
from EMSC_window_sampler import EMSCWindowSampler


class EMSCStagedTrainer:
    """
    EMSC分阶段训练器
    """
    
    def __init__(self, 
                 dataset_name: str = 'dataset',
                 state_dim: int = 8,
                 hidden_dim: int = 32,
                 learning_rate: float = 1e-3,
                 base_save_dir: str = './',
                 normalize: bool = True):
        """
        初始化分阶段训练器
        
        Args:
            dataset_name: 数据集名称
            state_dim: 状态向量维度
            hidden_dim: 隐藏层维度
            learning_rate: 学习率
            base_save_dir: 基础保存目录
            normalize: 是否进行数据归一化
        """
        self.dataset_name = dataset_name
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.base_save_dir = base_save_dir
        self.normalize = normalize
        
        # 获取数据集路径
        self.dataset_paths = get_dataset_paths(dataset_name)
        self.train_path = self.dataset_paths['train']
        self.val_path = self.dataset_paths['val']
        
        # 分阶段配置
        self.stage_configs = {
            'stage1': {
                'window_size': 201,
                'stride': 10,
                'epochs': 200,
                'description': '短序列基础训练',
                'augment_factor': 5  # 每个完整序列生成5个窗口
            },
            'stage2': {
                'window_size': 1001,
                'stride': 50,
                'epochs': 400,
                'description': '中序列扩展训练',
                'augment_factor': 3  # 每个完整序列生成3个窗口
            },
            'stage3': {
                'window_size': None,  # 使用完整序列
                'stride': None,
                'epochs': 2900,
                'description': '完整序列精细训练',
                'augment_factor': 1   # 使用原始完整序列
            }
        }
        
        # 创建阶段保存目录
        self.stage_dirs = {}
        for stage_name in self.stage_configs.keys():
            stage_dir = os.path.join(base_save_dir, f"{dataset_name}_{stage_name}")
            os.makedirs(stage_dir, exist_ok=True)
            self.stage_dirs[stage_name] = stage_dir
        
        # 初始化数据采样器
        self.train_sampler = None
        self.val_sampler = None
        
        print(f"EMSC分阶段训练器初始化完成")
        print(f"数据集: {dataset_name}")
        print(f"保存目录: {base_save_dir}")
        print(f"数据归一化: {'启用' if normalize else '禁用'}")
    
    def load_data(self):
        """加载训练和验证数据"""
        print("\n加载数据集...")
        
        # 加载训练数据
        self.train_sampler = EMSCWindowSampler(
            dataset_path=self.train_path,
            scalers_dir=os.path.join(os.path.dirname(self.train_path), 'scalers'),
            normalize=self.normalize,
            state_dim=self.state_dim
        )
        
        # 加载验证数据
        self.val_sampler = EMSCWindowSampler(
            dataset_path=self.val_path,
            scalers_dir=os.path.join(os.path.dirname(self.val_path), 'scalers'),
            normalize=self.normalize,
            state_dim=self.state_dim
        )
        
        print("数据集加载完成")
    
    def prepare_stage_data(self, stage_name: str) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray],
                                                         List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        准备指定阶段的训练数据
        
        Args:
            stage_name: 阶段名称 ('stage1', 'stage2', 'stage3')
            
        Returns:
            Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray],
                 List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
                - 训练输入窗口列表
                - 训练输出窗口列表
                - 训练初始状态列表
                - 验证输入窗口列表
                - 验证输出窗口列表
                - 验证初始状态列表
        """
        config = self.stage_configs[stage_name]
        window_size = config['window_size']
        stride = config['stride']
        augment_factor = config['augment_factor']
        
        print(f"\n准备 {stage_name} 阶段数据:")
        print(f"- 窗口大小: {window_size if window_size else '完整序列'}")
        print(f"- 采样步长: {stride if stride else 'N/A'}")
        print(f"- 增强因子: {augment_factor}")
        
        # 恢复物理量序列
        train_physical = self.train_sampler.recover_physical_quantities()
        val_physical = self.val_sampler.recover_physical_quantities()
        
        # 采样窗口
        if window_size is None:
            # 阶段3：使用完整序列
            train_windows_X = self.train_sampler.X_sequences
            train_windows_Y = self.train_sampler.Y_sequences
            train_init_states = [np.zeros(self.state_dim) for _ in range(len(train_windows_X))]
            
            val_windows_X = self.val_sampler.X_sequences
            val_windows_Y = self.val_sampler.Y_sequences
            val_init_states = [np.zeros(self.state_dim) for _ in range(len(val_windows_X))]
        else:
            # 阶段1和2：使用窗口采样
            train_windows_X, train_windows_Y, train_init_states = self.train_sampler.sample_windows(
                train_physical,
                self.train_sampler.Y_sequences,
                window_size=window_size,
                stride=stride,
                augment_factor=augment_factor,
                shuffle=True
            )
            
            val_windows_X, val_windows_Y, val_init_states = self.val_sampler.sample_windows(
                val_physical,
                self.val_sampler.Y_sequences,
                window_size=window_size,
                stride=stride,
                augment_factor=augment_factor,
                shuffle=False
            )
        
        print(f"\n数据准备完成:")
        print(f"- 训练窗口数: {len(train_windows_X)}")
        print(f"- 验证窗口数: {len(val_windows_X)}")
        if len(train_windows_X) > 0:
            print(f"- 窗口形状: {train_windows_X[0].shape}")
        
        return (train_windows_X, train_windows_Y, train_init_states,
                val_windows_X, val_windows_Y, val_init_states)
    
    def train_stage(self, stage_name: str):
        """
        训练指定阶段
        
        Args:
            stage_name: 阶段名称 ('stage1', 'stage2', 'stage3')
        """
        config = self.stage_configs[stage_name]
        stage_dir = self.stage_dirs[stage_name]
        
        print(f"\n开始 {stage_name} 训练:")
        print(f"- 描述: {config['description']}")
        print(f"- 训练轮数: {config['epochs']}")
        
        # 准备数据
        (train_windows_X, train_windows_Y, train_init_states,
         val_windows_X, val_windows_Y, val_init_states) = self.prepare_stage_data(stage_name)
        
        # 构建模型
        model = build_msc_model(
            state_dim=self.state_dim,
            hidden_dim=self.hidden_dim,
            learning_rate=self.learning_rate
        )
        
        # 设置回调
        callbacks = [
            MSCProgressCallback(stage_name),
            create_early_stopping_callback(patience=50),
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(stage_dir, 'best_model.h5'),
                save_best_only=True,
                monitor='val_loss'
            )
        ]
        
        # 训练模型
        history = model.fit(
            x=[train_windows_X, train_init_states],
            y=train_windows_Y,
            validation_data=([val_windows_X, val_init_states], val_windows_Y),
            epochs=config['epochs'],
            callbacks=callbacks,
            verbose=1
        )
        
        # 保存训练历史
        history_path = os.path.join(stage_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history.history, f, indent=4)
        
        # 保存阶段配置
        self.save_stage_config(stage_name)
        
        print(f"\n{stage_name} 训练完成")
        print(f"模型和训练历史已保存到: {stage_dir}")
        
        return model, history
    
    def train_all_stages(self):
        """训练所有阶段"""
        models = {}
        histories = {}
        
        for stage_name in ['stage1', 'stage2', 'stage3']:
            model, history = self.train_stage(stage_name)
            models[stage_name] = model
            histories[stage_name] = history
        
        return models, histories

    def analyze_stage_requirements(self, stage_name):
        """
        分析阶段训练的数据要求
        
        Args:
            stage_name: 阶段名称 ('stage1', 'stage2', 'stage3')
        """
        config = self.stage_configs[stage_name]
        window_size = config['window_size']
        
        print(f"\n=== {stage_name.upper()} 数据需求分析 ===")
        print(f"描述: {config['description']}")
        print(f"窗口大小: {window_size if window_size else '完整序列'}")
        print(f"训练轮数: {config['epochs']}")
        print(f"增强因子: {config['augment_factor']}")
        
        if window_size:
            # 分析可用于该窗口大小的序列
            valid_train_seqs = [seq for seq in self.X_train if len(seq) >= window_size]
            valid_val_seqs = [seq for seq in self.X_val if len(seq) >= window_size]
            
            total_train_windows = sum(min(len(seq) - window_size + 1, config['augment_factor']) 
                                    for seq in valid_train_seqs)
            total_val_windows = sum(min(len(seq) - window_size + 1, config['augment_factor']) 
                                  for seq in valid_val_seqs)
            
            print(f"\n数据可用性:")
            print(f"- 可用训练序列: {len(valid_train_seqs)}/{len(self.X_train)}")
            print(f"- 可用验证序列: {len(valid_val_seqs)}/{len(self.X_val)}")
            print(f"- 总训练窗口数: {total_train_windows}")
            print(f"- 总验证窗口数: {total_val_windows}")
            
            # 计算数据增强效果
            print(f"\n数据增强效果:")
            print(f"- 原始训练序列: {len(valid_train_seqs)}")
            print(f"- 增强后窗口: {total_train_windows}")
            print(f"- 增强倍数: {total_train_windows / len(valid_train_seqs):.1f}x")
        else:
            print(f"\n使用完整序列:")
            print(f"- 训练序列数: {len(self.X_train)}")
            print(f"- 验证序列数: {len(self.X_val)}")
    
    def save_stage_config(self, stage_name):
        """保存阶段配置"""
        config = self.stage_configs[stage_name].copy()
        config.update({
            'dataset_name': self.dataset_name,
            'state_dim': self.state_dim,
            'hidden_dim': self.hidden_dim,
            'learning_rate': self.learning_rate,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        config_path = os.path.join(self.stage_dirs[stage_name], 'stage_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        print(f"阶段配置已保存: {config_path}")





def main():
    """主函数 - 演示分阶段训练框架的使用"""
    
    # 初始化训练器
    trainer = EMSCStagedTrainer(
        dataset_name='dataset_EMSC_tt',
        state_dim=8,
        hidden_dim=32,
        learning_rate=1e-3,
        normalize=True  # 启用数据归一化
    )
    
    # 加载数据
    trainer.load_data()
    
    # 训练所有阶段
    models, histories = trainer.train_all_stages()
    
    print("\n分阶段训练完成！")
    print("模型已保存到各阶段目录")


if __name__ == '__main__':
    main() 