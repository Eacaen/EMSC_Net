"""
阿里云环境数据I/O优化模块
专门解决云环境CPU使用率低的问题
"""

import os
import time
import threading
import multiprocessing as mp
import numpy as np
import tensorflow as tf
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed


class CloudIOOptimizer:
    """
    阿里云环境I/O优化器
    主要解决：
    1. 云盘IOPS限制导致的I/O瓶颈
    2. 数据加载管道效率低下
    3. CPU等待I/O导致使用率低
    """
    
    def __init__(self, 
                 io_buffer_size=64,      # I/O缓冲区大小 
                 prefetch_factor=8,      # 预取因子（云环境需要更激进）
                 io_threads=16,          # I/O线程数（云环境增加）
                 memory_cache_size=512,  # 内存缓存大小(MB)
                 use_ramdisk=False):     # 是否使用内存盘
        
        self.io_buffer_size = io_buffer_size
        self.prefetch_factor = prefetch_factor
        self.io_threads = io_threads
        self.memory_cache_size = memory_cache_size
        self.use_ramdisk = use_ramdisk
        
        # 创建I/O线程池
        self.io_executor = ThreadPoolExecutor(max_workers=io_threads)
        
        # 内存缓存
        self.memory_cache = {}
        self.cache_lock = threading.Lock()
        
        print(f"🔧 阿里云I/O优化器初始化:")
        print(f"- I/O缓冲区: {io_buffer_size}")
        print(f"- 预取因子: {prefetch_factor}x")
        print(f"- I/O线程数: {io_threads}")
        print(f"- 内存缓存: {memory_cache_size}MB")
    
    def optimize_tensorflow_io(self):
        """优化TensorFlow的I/O配置"""
        
        # 1. 设置TensorFlow I/O优化
        os.environ['TF_DATA_EXPERIMENTAL_IO_MEMORY_BUFFER_SIZE'] = str(self.memory_cache_size * 1024 * 1024)
        os.environ['TF_DATA_EXPERIMENTAL_SLACK'] = 'True'
        
        # 2. 云环境专用配置
        os.environ['TF_DATA_AUTOTUNE_MEMORY_BUDGET'] = str(self.memory_cache_size * 1024 * 1024)
        
        # 3. 激进的并行配置
        tf.config.threading.set_inter_op_parallelism_threads(self.io_threads)
        
        print(f"✅ TensorFlow I/O优化配置完成")
    
    def create_optimized_dataset(self, X_paths, Y_paths, init_states, batch_size):
        """
        创建针对云环境优化的数据集
        """
        print(f"🚀 创建云环境优化数据集...")
        
        # 创建数据集
        dataset = tf.data.Dataset.from_tensor_slices((
            {'delta_input': X_paths, 'init_state': init_states},
            Y_paths
        ))
        
        # 1. 激进的缓存策略
        dataset = dataset.cache()  # 全量缓存到内存
        
        # 2. 更大的shuffle buffer（云环境网络延迟高）
        dataset = dataset.shuffle(
            buffer_size=min(len(X_paths), 10000),  # 更大的shuffle buffer
            reshuffle_each_iteration=True
        )
        
        # 3. 批处理
        dataset = dataset.batch(batch_size, drop_remainder=False)
        
        # 4. 超激进的预取（云环境关键优化）
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        
        # 5. 并行数据处理
        dataset = dataset.map(
            lambda x, y: (x, y),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False  # 允许非确定性以提高性能
        )
        
        # 6. 再次预取（双重预取策略）
        dataset = dataset.prefetch(buffer_size=self.prefetch_factor)
        
        print(f"✅ 云环境数据集优化完成")
        return dataset
    
    def diagnose_io_bottleneck(self, data_path):
        """
        诊断I/O瓶颈
        """
        print(f"🔍 开始I/O瓶颈诊断...")
        
        # 1. 磁盘读取性能测试
        start_time = time.time()
        test_size = 100 * 1024 * 1024  # 100MB测试
        
        try:
            # 创建测试文件
            test_file = os.path.join(data_path, 'io_test.tmp')
            test_data = np.random.bytes(test_size)
            
            # 写入测试
            write_start = time.time()
            with open(test_file, 'wb') as f:
                f.write(test_data)
            write_time = time.time() - write_start
            
            # 读取测试
            read_start = time.time()
            with open(test_file, 'rb') as f:
                _ = f.read()
            read_time = time.time() - read_start
            
            # 清理测试文件
            os.remove(test_file)
            
            write_speed = test_size / write_time / (1024*1024)  # MB/s
            read_speed = test_size / read_time / (1024*1024)   # MB/s
            
            print(f"📊 磁盘性能测试结果:")
            print(f"- 写入速度: {write_speed:.1f} MB/s")
            print(f"- 读取速度: {read_speed:.1f} MB/s")
            
            # 性能建议
            if read_speed < 50:
                print(f"⚠️  读取速度较慢，建议:")
                print(f"   - 升级到SSD云盘")
                print(f"   - 增加IOPS配置")
                print(f"   - 使用本地SSD")
            
            if write_speed < 30:
                print(f"⚠️  写入速度较慢，可能影响模型保存")
            
        except Exception as e:
            print(f"❌ I/O测试失败: {e}")
    
    def optimize_cloud_environment(self):
        """
        云环境综合优化
        """
        print(f"🔧 开始云环境综合优化...")
        
        # 1. 系统级优化
        self._optimize_system_io()
        
        # 2. TensorFlow优化
        self.optimize_tensorflow_io()
        
        # 3. 内存优化
        self._optimize_memory()
        
        print(f"✅ 云环境优化完成")
    
    def _optimize_system_io(self):
        """系统级I/O优化"""
        try:
            # 设置I/O调度器优化参数
            io_optimizations = {
                'TF_DATA_EXPERIMENTAL_ENABLE_NUMA_AWARE_DATASETS': '1',
                'TF_DATA_EXPERIMENTAL_IO_THREAD_POOL_SIZE': str(self.io_threads),
                'TF_ENABLE_EAGER_CLIENT_STREAMING_ENQUEUE': '1',
            }
            
            for key, value in io_optimizations.items():
                os.environ[key] = value
                
            print(f"✅ 系统I/O优化完成")
            
        except Exception as e:
            print(f"⚠️  系统I/O优化部分失败: {e}")
    
    def _optimize_memory(self):
        """内存优化"""
        try:
            # Python内存优化
            import gc
            gc.collect()
            
            # 设置TensorFlow内存增长
            gpus = tf.config.list_physical_devices('GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            print(f"✅ 内存优化完成")
            
        except Exception as e:
            print(f"⚠️  内存优化部分失败: {e}")


def create_cloud_optimized_training_config(num_samples, cpu_count):
    """
    为阿里云环境创建优化的训练配置
    
    Args:
        num_samples: 训练样本数
        cpu_count: CPU核心数
    
    Returns:
        dict: 优化配置
    """
    
    # 云环境特殊配置
    config = {
        # 批次大小：云环境优先考虑I/O效率
        'batch_size': max(64, min(256, num_samples // 20)),
        
        # I/O配置：激进的并行和缓存
        'io_threads': min(cpu_count * 2, 32),
        'prefetch_factor': 16,  # 云环境更激进的预取
        'cache_size': 1024,     # 1GB内存缓存
        
        # 数据加载配置
        'num_parallel_calls': tf.data.AUTOTUNE,
        'shuffle_buffer': min(num_samples, 8192),
        
        # 训练配置
        'workers': min(cpu_count, 16),  # 限制进程数
        'max_queue_size': cpu_count * 4,  # 大队列
        'use_multiprocessing': True,
    }
    
    print(f"🌥️  阿里云训练配置:")
    for key, value in config.items():
        print(f"- {key}: {value}")
    
    return config


def monitor_cloud_performance():
    """
    监控云环境性能指标
    """
    try:
        import psutil
        
        print(f"📊 云环境性能监控:")
        
        # CPU信息
        cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
        avg_cpu = sum(cpu_percent) / len(cpu_percent)
        print(f"- 平均CPU使用率: {avg_cpu:.1f}%")
        print(f"- CPU核心使用率分布: {[f'{x:.1f}%' for x in cpu_percent]}")
        
        # 内存信息
        memory = psutil.virtual_memory()
        print(f"- 内存使用率: {memory.percent:.1f}%")
        print(f"- 可用内存: {memory.available / (1024**3):.1f}GB")
        
        # 磁盘I/O
        disk_io = psutil.disk_io_counters()
        print(f"- 磁盘读取: {disk_io.read_bytes / (1024**3):.2f}GB")
        print(f"- 磁盘写入: {disk_io.write_bytes / (1024**3):.2f}GB")
        
        # 网络I/O
        net_io = psutil.net_io_counters()
        print(f"- 网络接收: {net_io.bytes_recv / (1024**3):.2f}GB")
        print(f"- 网络发送: {net_io.bytes_sent / (1024**3):.2f}GB")
        
        # 诊断建议
        if avg_cpu < 30:
            print(f"💡 CPU使用率偏低，可能原因:")
            print(f"   - I/O瓶颈（磁盘/网络）")
            print(f"   - 数据加载管道效率低")
            print(f"   - 批次大小过小")
            print(f"   - 线程配置不当")
        
        if memory.percent > 85:
            print(f"⚠️  内存使用率过高，可能影响性能")
        
    except ImportError:
        print(f"❌ 需要安装psutil进行性能监控")


if __name__ == "__main__":
    # 创建优化器
    optimizer = CloudIOOptimizer()
    
    # 运行优化
    optimizer.optimize_cloud_environment()
    
    # 性能监控
    monitor_cloud_performance()