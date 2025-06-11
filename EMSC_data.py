"""
EMSC数据加载模块
包含所有数据加载和处理相关的类和函数
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import threading
from queue import Queue
from datetime import datetime
import joblib

class EMSCDataGenerator(Sequence):
    """
    自定义数据生成器，用于高效加载大型数据集
    针对15000条数据的数据集优化参数
    """
    def __init__(self, X_paths, Y_paths, init_states, batch_size=8, shuffle=True,
                 num_workers=4, prefetch_factor=2):
        self.X_paths = X_paths
        self.Y_paths = Y_paths
        self.init_states = init_states
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(X_paths))
        self.on_epoch_end()
        
        # 优化缓存参数
        total_samples = len(X_paths)
        self.cache_size = min(int(total_samples * 0.05), 1000)
        self.preload_queue_size = min(self.cache_size * prefetch_factor, 2000)
        
        # 创建数据缓存
        self.cache = {}
        self.cache_lock = threading.Lock()
        
        # 创建预加载线程池
        self.preload_queue = Queue(maxsize=self.preload_queue_size)
        self.stop_preload = threading.Event()
        self.num_preload_threads = min(num_workers, 8)  # 限制最大线程数
        self.preload_threads = []
        for _ in range(self.num_preload_threads):
            thread = threading.Thread(target=self._preload_data)
            thread.daemon = True
            thread.start()
            self.preload_threads.append(thread)
        
        print(f"数据生成器初始化完成:")
        print(f"- 总样本数: {total_samples}")
        print(f"- 缓存大小: {self.cache_size}")
        print(f"- 预加载队列大小: {self.preload_queue_size}")
        print(f"- 预加载线程数: {self.num_preload_threads}")
        print(f"- 预取因子: {prefetch_factor}")
    
    def __len__(self):
        """返回批次数量"""
        return int(np.ceil(len(self.X_paths) / self.batch_size))
    
    def on_epoch_end(self):
        """每个epoch结束时调用"""
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def _preload_data(self):
        """预加载数据的线程函数"""
        while not self.stop_preload.is_set():
            try:
                idx = self.preload_queue.get(timeout=1)
                if idx not in self.cache:
                    with self.cache_lock:
                        if len(self.cache) >= self.cache_size:
                            num_to_remove = int(self.cache_size * 0.2)
                            for _ in range(num_to_remove):
                                if self.cache:
                                    oldest_key = next(iter(self.cache))
                                    del self.cache[oldest_key]
                        self.cache[idx] = {
                            'X': np.array(self.X_paths[idx], dtype=np.float32),
                            'Y': np.array(self.Y_paths[idx], dtype=np.float32)
                        }
                self.preload_queue.task_done()
            except:
                continue
    
    def __getitem__(self, idx):
        """获取一个批次的数据"""
        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        batch_X = []
        batch_Y = []
        batch_init_states = []
        
        next_batch_start = ((idx + 1) * self.batch_size) % len(self.X_paths)
        next_batch_indices = range(next_batch_start, 
                                 min(next_batch_start + self.batch_size, len(self.X_paths)))
        
        for next_idx in next_batch_indices:
            if next_idx not in self.cache:
                try:
                    self.preload_queue.put(next_idx, block=False)
                except:
                    pass
        
        for i in batch_indexes:
            with self.cache_lock:
                if i in self.cache:
                    data = self.cache[i]
                    X = data['X']
                    Y = data['Y']
                else:
                    X = np.array(self.X_paths[i], dtype=np.float32)
                    Y = np.array(self.Y_paths[i], dtype=np.float32)
                    if len(self.cache) < self.cache_size:
                        self.cache[i] = {'X': X, 'Y': Y}
            
            batch_X.append(X)
            batch_Y.append(Y)
            batch_init_states.append(self.init_states[i])
        
        batch_X = np.array(batch_X)
        batch_Y = np.array(batch_Y)
        batch_init_states = np.array(batch_init_states)
        
        return {
            'delta_input': batch_X,
            'init_state': batch_init_states
        }, batch_Y
    
    def __del__(self):
        """清理资源"""
        self.stop_preload.set()
        for thread in self.preload_threads:
            thread.join(timeout=1)

def create_tf_dataset(X_paths, Y_paths, init_states, batch_size=8, shuffle=True,
                     num_parallel_calls=tf.data.AUTOTUNE, prefetch_buffer_size=None):
    """
    创建TensorFlow数据集，针对15000条数据优化参数
    
    Args:
        X_paths: 输入数据路径列表
        Y_paths: 输出数据路径列表
        init_states: 初始状态数组
        batch_size: 批处理大小
        shuffle: 是否打乱数据
        num_parallel_calls: 并行调用数
        prefetch_buffer_size: 预取缓冲区大小，如果为None则使用AUTOTUNE
    """
    total_samples = len(X_paths)
    shuffle_buffer_size = min(total_samples, 5000)
    
    dataset = tf.data.Dataset.from_tensor_slices((
        {
            'delta_input': X_paths,
            'init_state': init_states
        },
        Y_paths
    ))
    
    dataset = dataset.cache()
    
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=shuffle_buffer_size,
            reshuffle_each_iteration=True
        )
    
    dataset = dataset.batch(batch_size)
    
    # 使用指定的预取缓冲区大小或AUTOTUNE
    prefetch_size = prefetch_buffer_size if prefetch_buffer_size is not None else tf.data.AUTOTUNE
    dataset = dataset.prefetch(buffer_size=prefetch_size)
    
    # 并行处理优化
    dataset = dataset.map(
        lambda x, y: (x, y),
        num_parallel_calls=num_parallel_calls
    )
    
    return dataset

def save_dataset_to_npz(X_paths, Y_paths, save_path='./msc_models/dataset.npz'):
    """
    将预处理后的数据集保存为npz格式
    """
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        X_array = np.array(X_paths, dtype=object)
        Y_array = np.array(Y_paths, dtype=object)
        
        np.savez_compressed(
            save_path,
            X_paths=X_array,
            Y_paths=Y_array,
            save_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        print(f"数据集已保存至: {save_path}")
        return save_path
    except Exception as e:
        print(f"保存数据集时出错: {e}")
        return None

def load_dataset_from_npz(npz_path='./msc_models/dataset.npz'):
    """
    从npz文件加载数据集
    """
    try:
        if not os.path.exists(npz_path):
            print(f"数据集文件不存在: {npz_path}")
            return None, None
        
        data = np.load(npz_path, allow_pickle=True)
        X_paths = data['X_paths'].tolist()
        Y_paths = data['Y_paths'].tolist()
        
        print(f"从 {npz_path} 加载数据集")
        print(f"保存时间: {data['save_time']}")
        print(f"序列数量: {len(X_paths)}")
        
        lengths = [len(x) for x in X_paths]
        print(f"序列长度统计:")
        print(f"最短: {min(lengths)}")
        print(f"最长: {max(lengths)}")
        print(f"平均: {np.mean(lengths):.2f}")
        print(f"中位数: {np.median(lengths)}")
        
        return X_paths, Y_paths
    except Exception as e:
        print(f"加载数据集时出错: {e}")
        return None, None