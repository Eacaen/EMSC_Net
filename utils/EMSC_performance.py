"""
EMSC模型性能优化模块
提供性能优化相关的配置和工具函数
"""

import os
import tensorflow as tf
import numpy as np
from tensorflow.keras import mixed_precision
from tensorflow.keras.backend import clear_session

def get_optimal_batch_size(num_samples, num_gpus=1, min_batch=8, max_batch=32):
    """
    计算最优批处理大小
    
    Args:
        num_samples: 训练样本数量
        num_gpus: GPU数量
        min_batch: 最小批处理大小
        max_batch: 最大批处理大小
    
    Returns:
        int: 最优批处理大小
    """
    # 基础批处理大小
    base_batch = min(max_batch, num_samples // 100)
    base_batch = max(min_batch, base_batch)
    
    # 根据GPU数量调整
    optimal_batch = base_batch * num_gpus
    
    # 确保是8的倍数（对GPU内存对齐有利）
    optimal_batch = (optimal_batch // 8) * 8
    
    return optimal_batch

def get_optimal_workers():
    """
    计算最优工作线程数
    
    Returns:
        int: 最优工作线程数
    """
    cpu_count = os.cpu_count()
    if cpu_count is None:
        return 4
    
    # 保留一半CPU核心给系统和其他进程
    return max(1, cpu_count // 2)

def setup_mixed_precision():
    """
    设置混合精度训练
    
    Returns:
        bool: 是否成功启用混合精度
    """
    try:
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_policy(policy)
        return True
    except Exception as e:
        print(f"混合精度训练设置失败: {e}")
        return False

def setup_gpu_memory():
    """
    设置GPU内存增长
    
    Returns:
        int: 可用的GPU数量
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"已启用 {len(gpus)} 个GPU的动态内存分配")
        except RuntimeError as e:
            print(f"GPU设置错误: {e}")
    return len(gpus)

def setup_tf_performance():
    """
    设置TensorFlow性能优化
    
    Returns:
        dict: 性能配置信息
    """
    # 设置线程数
    num_workers = get_optimal_workers()
    tf.config.threading.set_inter_op_parallelism_threads(num_workers)
    tf.config.threading.set_intra_op_parallelism_threads(num_workers)
    
    # 设置GPU内存
    num_gpus = setup_gpu_memory()
    
    # 设置混合精度
    mixed_precision_enabled = setup_mixed_precision()
    
    # 启用XLA JIT编译
    try:
        tf.config.optimizer.set_jit(True)
        xla_enabled = True
    except Exception as e:
        print(f"XLA JIT编译设置失败: {e}")
        xla_enabled = False
    
    return {
        'num_workers': num_workers,
        'num_gpus': num_gpus,
        'mixed_precision': mixed_precision_enabled,
        'xla_enabled': xla_enabled
    }

def create_performance_optimized_dataset(dataset, batch_size, is_training=True):
    """
    创建性能优化的数据集
    
    Args:
        dataset: 输入数据集
        batch_size: 批处理大小
        is_training: 是否为训练集
    
    Returns:
        tf.data.Dataset: 优化后的数据集
    """
    # 设置预取和缓存
    dataset = dataset.cache()
    
    if is_training:
        # 训练集优化
        dataset = dataset.shuffle(
            buffer_size=min(10000, batch_size * 10),
            reshuffle_each_iteration=True
        )
    
    # 批处理
    dataset = dataset.batch(batch_size)
    
    # 预取
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    # 并行处理
    dataset = dataset.map(
        lambda x, y: (x, y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    return dataset

def get_distributed_strategy():
    """
    获取分布式训练策略
    
    Returns:
        tf.distribute.Strategy: 分布式训练策略
    """
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if len(gpus) > 1:
            # 使用多GPU策略
            strategy = tf.distribute.MirroredStrategy()
            print(f"使用多GPU训练策略，检测到 {len(gpus)} 个GPU")
        else:
            # 使用默认策略
            strategy = tf.distribute.get_strategy()
            print("使用默认训练策略")
        return strategy
    except Exception as e:
        print(f"创建分布式策略失败: {e}")
        return tf.distribute.get_strategy()

def clear_tf_memory():
    """
    清理TensorFlow内存
    """
    clear_session()
    tf.keras.backend.clear_session()
    
    # 强制垃圾回收
    import gc
    gc.collect()

def get_performance_config():
    """
    获取完整的性能配置
    
    Returns:
        dict: 性能配置信息
    """
    # 清理现有会话
    clear_tf_memory()
    
    # 设置基本性能优化
    perf_config = setup_tf_performance()
    
    # 获取分布式策略
    strategy = get_distributed_strategy()
    perf_config['strategy'] = strategy
    
    # 添加其他性能相关配置
    perf_config.update({
        'tf_data_optimization': True,
        'memory_growth': True,
        'num_parallel_calls': tf.data.AUTOTUNE,
        'prefetch_buffer_size': tf.data.AUTOTUNE
    })
    
    return perf_config

def print_performance_config(config):
    """
    打印性能配置信息
    
    Args:
        config: 性能配置字典
    """
    print("\n性能优化配置:")
    print(f"- GPU数量: {config['num_gpus']}")
    print(f"- 工作线程数: {config['num_workers']}")
    print(f"- 混合精度训练: {'启用' if config['mixed_precision'] else '禁用'}")
    print(f"- XLA JIT编译: {'启用' if config['xla_enabled'] else '禁用'}")
    print(f"- 分布式策略: {type(config['strategy']).__name__}")
    print(f"- 数据优化: {'启用' if config['tf_data_optimization'] else '禁用'}")
    print(f"- 内存增长: {'启用' if config['memory_growth'] else '禁用'}")
    print(f"- 并行调用: {config['num_parallel_calls']}")
    print(f"- 预取缓冲区: {config['prefetch_buffer_size']}")
    print()