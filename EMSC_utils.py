"""
EMSC工具模块
包含通用工具函数
"""

import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from EMSC_model import MSC_Sequence, build_msc_model
from EMSC_losses import EMSCLoss

def load_or_create_model_with_history(model_path='./msc_models/', model_name='msc_model', 
                                     best_model_name='best_msc_model',
                                     state_dim=8, input_dim=6, output_dim=1,
                                     hidden_dim=32, num_internal_layers=2):
    """
    加载已存在的SavedModel格式模型或创建新模型
    """
    model_file_path = os.path.join(model_path, best_model_name)
    fallback_path = os.path.join(model_path, model_name)
    
    if os.path.exists(model_file_path):
        print(f"Found existing best model: {model_file_path}")
        try:
            model = tf.keras.models.load_model(
                model_file_path,
                custom_objects={'MSC_Sequence': MSC_Sequence}
            )
            print("Successfully loaded existing best model")
            return model, False
        except Exception as e:
            print(f"Error loading best model: {e}")
    
    if os.path.exists(fallback_path):
        print(f"Found existing current model: {fallback_path}")
        try:
            model = tf.keras.models.load_model(
                fallback_path,
                custom_objects={'MSC_Sequence': MSC_Sequence}
            )
            print("Successfully loaded existing current model")
            return model, False
        except Exception as e:
            print(f"Error loading current model: {e}")
    
    print("Creating new model")
    model = build_msc_model(
        state_dim=state_dim,
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        num_internal_layers=num_internal_layers
    )
    return model, True

def resume_training_from_checkpoint(model_path='./msc_models/', model_name='msc_model', 
                                   best_model_name='best_msc_model', resume_from_best=True,
                                   state_dim=8, input_dim=6, output_dim=1,
                                   hidden_dim=32, num_internal_layers=2):
    """从SavedModel格式的检查点恢复训练"""
    if resume_from_best:
        model_dir = os.path.join(model_path, best_model_name)
        fallback_dir = os.path.join(model_path, model_name)
    else:
        model_dir = os.path.join(model_path, model_name)
        fallback_dir = os.path.join(model_path, best_model_name)
    
    model = None
    if os.path.exists(model_dir):
        try:
            print(f"恢复训练，加载模型: {model_dir}")
            with tf.keras.utils.custom_object_scope({
                'MSC_Sequence': MSC_Sequence,
                'EMSCLoss': EMSCLoss
            }):
                model = tf.keras.models.load_model(model_dir)
            print("模型加载成功")
        except Exception as e:
            print(f"加载主模型失败: {e}")
    
    if model is None and os.path.exists(fallback_dir):
        try:
            print(f"尝试加载后备模型: {fallback_dir}")
            with tf.keras.utils.custom_object_scope({
                'MSC_Sequence': MSC_Sequence,
                'EMSCLoss': EMSCLoss
            }):
                model = tf.keras.models.load_model(fallback_dir)
            print("后备模型加载成功")
        except Exception as e:
            print(f"加载后备模型也失败: {e}")
    
    if model is None:
        print("未找到可用的检查点模型")
        return None, 0
    
    epoch_offset = 0
    try:
        import pandas as pd
        history_files = [
            ('training_history.csv', pd.read_csv),
            ('training_history.xlsx', pd.read_excel),
            ('training_history.excel', pd.read_excel)
        ]
        
        for filename, read_func in history_files:
            history_file = os.path.join(model_path, filename)
            if os.path.exists(history_file):
                try:
                    df = read_func(history_file)
                    epoch_offset = len(df)
                    print(f"从训练历史中检测到已完成 {epoch_offset} 个epochs (来源: {filename})")
                    break
                except Exception as e:
                    print(f"读取 {filename} 时出错: {e}")
                    continue
        else:
            print("未找到训练历史文件，从epoch 0开始")
    except Exception as e:
        print(f"读取训练历史时出错: {e}")
        epoch_offset = 0
    
    return model, epoch_offset

def plot_final_training_summary(history, initial_epoch, epochs, progress_callback, dataset_dir):
    """绘制最终的训练总结图"""
    if len(history.history['loss']) > 0:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(range(initial_epoch + 1, epochs + 1), history.history['loss'], 
                label='Training Loss', color='blue')
        plt.plot(range(initial_epoch + 1, epochs + 1), history.history['val_loss'], 
                label='Validation Loss', color='red')
        plt.title('Model Loss During Training (Final)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        if progress_callback.training_history['epoch_times']:
            plt.subplot(1, 2, 2)
            recent_times = progress_callback.training_history['epoch_times'][-epochs:]
            plt.plot(range(initial_epoch + 1, initial_epoch + 1 + len(recent_times)), 
                    recent_times, label='Epoch Duration', color='green')
            plt.title('Training Time per Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('Time (seconds)')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(dataset_dir, 'final_training_summary.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"最终训练总结图保存至: {os.path.join(dataset_dir, 'final_training_summary.png')}")

def print_training_summary(progress_callback, dataset_dir, best_model_name, model_name):
    """打印训练总结"""
    print("="*60)
    print("训练完成总结:")
    print(f"最佳验证损失: {progress_callback.best_val_loss:.6f}")
    print(f"总训练时间: {sum(progress_callback.training_history['epoch_times']):.2f} 秒")
    print(f"平均每epoch时间: {np.mean(progress_callback.training_history['epoch_times']):.2f} 秒")
    print(f"模型保存路径: {dataset_dir}")
    print(f"最佳模型: {os.path.join(dataset_dir, best_model_name)}")
    print(f"当前模型: {os.path.join(dataset_dir, model_name)}")
    print(f"训练历史: {os.path.join(dataset_dir, 'training_history.csv')}")
    print(f"训练图表: {os.path.join(dataset_dir, 'training_history.png')}")
    print(f"训练配置: {os.path.join(dataset_dir, 'training_config.json')}")
    print("="*60)