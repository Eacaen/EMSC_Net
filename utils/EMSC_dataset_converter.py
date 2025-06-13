#!/usr/bin/env python3
"""
将npz数据集转换为TFRecord格式，提供高效的数据加载
"""

import tensorflow as tf
import numpy as np
import os
import json
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
from functools import partial
import time
import shutil
import glob
try:
    import psutil
except ImportError:
    print("⚠️ 警告: psutil未安装，无法监控内存使用率")
    psutil = None

def _bytes_feature(value):
    """将numpy数组转换为bytes特征"""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.tobytes()]))

def _float_feature(value):
    """将float值转换为float特征"""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def check_dataset_exists(tfrecord_path):
    """
    检查TFRecord数据集是否存在且完整
    
    Args:
        tfrecord_path: TFRecord文件路径
    
    Returns:
        bool: 数据集是否存在且完整
    """
    info_path = tfrecord_path + '.info.json'
    return os.path.exists(tfrecord_path) and os.path.exists(info_path)

def _process_data_chunk(data_chunk, keys, chunk_id):
    """
    处理单个数据块 - 用于多进程
    
    Args:
        data_chunk: 数据块 {key: chunk_data}
        keys: 字段列表
        chunk_id: 块ID
    
    Returns:
        list: 序列化的Example列表
    """
    examples = []
    chunk_size = len(data_chunk[keys[0]])
    
    for i in range(chunk_size):
        feature = {}
        
        for key in keys:
            item_data = data_chunk[key][i]
            
            if isinstance(item_data, np.ndarray):
                if item_data.dtype == np.float32 or item_data.dtype == np.float64:
                    if item_data.size == 1:
                        feature[key] = _float_feature(float(item_data))
                    else:
                        feature[key] = _bytes_feature(item_data.astype(np.float32))
                elif item_data.dtype == object:
                    if isinstance(item_data, np.ndarray) and len(item_data) > 0:
                        combined_array = item_data.flatten().astype(np.float32)
                        feature[key] = _bytes_feature(combined_array)
                    else:
                        str_data = np.array([str(x) for x in item_data], dtype=np.string_)
                        feature[key] = _bytes_feature(str_data)
                else:
                    str_data = np.array([str(x) for x in item_data], dtype=np.string_)
                    feature[key] = _bytes_feature(str_data)
            else:
                str_data = np.array([str(item_data)], dtype=np.string_)
                feature[key] = _bytes_feature(str_data)
        
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        examples.append(example.SerializeToString())
    
    return examples

def convert_npz_to_tfrecord(npz_path, tfrecord_path, batch_size=1000, force=False):
    """
    将npz数据集转换为TFRecord格式
    
    Args:
        npz_path: npz文件路径
        tfrecord_path: 输出TFRecord文件路径
        batch_size: 每批处理的数据量
        force: 是否强制重新转换（即使已存在）
    
    Returns:
        bool: 转换是否成功
    """
    # 检查输入文件是否存在
    if not os.path.exists(npz_path):
        print(f"❌ 输入文件不存在: {npz_path}")
        return False
    
    # 检查是否已经存在转换后的文件
    if check_dataset_exists(tfrecord_path) and not force:
        print(f"✅ TFRecord数据集已存在: {tfrecord_path}")
        print("💡 如需重新转换，请使用 --force 参数")
        return True
    
    try:
        print(f"📥 加载npz数据: {npz_path}")
        data = np.load(npz_path, mmap_mode='r', allow_pickle=True)  # 允许加载pickle数据
        
        # 获取所有键名
        keys = list(data.keys())
        print(f"📋 数据集包含以下字段: {keys}")
        
        # 只保留需要转换的字段
        target_keys = ['X_paths', 'Y_paths']
        keys = [key for key in keys if key in target_keys]
        if not keys:
            print("❌ 错误：未找到需要转换的字段 'X_paths' 或 'Y_paths'")
            return False
        print(f"🎯 将转换以下字段: {keys}")
        
        # 获取数据信息
        total_samples = len(data[keys[0]])  # 使用第一个键的长度作为样本数
        print(f"📊 总样本数: {total_samples}")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(tfrecord_path), exist_ok=True)
        
        # 创建TFRecord写入器
        with tf.io.TFRecordWriter(tfrecord_path) as writer:
            # 分批处理数据
            for i in tqdm(range(0, total_samples, batch_size), desc="转换数据"):
                # 创建特征字典
                feature = {}
                
                # 处理每个字段
                for key in keys:
                    batch_data = data[key][i:i+batch_size]
                    if isinstance(batch_data, np.ndarray):
                        if batch_data.dtype == np.float32 or batch_data.dtype == np.float64:
                            if batch_data.size == 1:  # 标量值（如temperature）
                                feature[key] = _float_feature(float(batch_data))
                            else:  # 数组
                                feature[key] = _bytes_feature(batch_data)
                        elif batch_data.dtype == object:  # 处理对象数组（嵌套numpy数组）
                            # 检查是否是嵌套的numpy数组
                            if len(batch_data) > 0 and isinstance(batch_data[0], np.ndarray):
                                # 将嵌套的numpy数组展平并合并
                                flattened_arrays = []
                                for item in batch_data:
                                    flattened_arrays.append(item.flatten())
                                # 合并所有数组
                                combined_array = np.concatenate(flattened_arrays).astype(np.float32)
                                feature[key] = _bytes_feature(combined_array)
                            else:
                                # 将对象数组转换为字符串列表
                                str_data = np.array([str(x) for x in batch_data], dtype=np.string_)
                                feature[key] = _bytes_feature(str_data)
                        else:
                            print(f"⚠️ 警告: 字段 {key} 的数据类型 {batch_data.dtype} 将被转换为字符串")
                            str_data = np.array([str(x) for x in batch_data], dtype=np.string_)
                            feature[key] = _bytes_feature(str_data)
                    else:
                        # 处理非数组数据
                        str_data = np.array([str(batch_data)], dtype=np.string_)
                        feature[key] = _bytes_feature(str_data)
                
                # 创建Example
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                
                # 写入TFRecord
                writer.write(example.SerializeToString())
        
        # 保存数据集信息
        dataset_info = {
            'keys': keys,  # 只保存转换的字段
            'total_samples': total_samples,
            'shapes': {},
            'dtypes': {},
            'created_at': str(np.datetime64('now')),
            'source_file': npz_path,
            'note': '仅包含 X_paths 和 Y_paths 字段'
        }
        
        # 为每个字段计算正确的形状和类型信息
        for key in keys:
            original_data = data[key]
            if original_data.dtype == object and len(original_data) > 0 and isinstance(original_data[0], np.ndarray):
                # 嵌套numpy数组：记录展平后的总大小
                sample_item = original_data[0]
                flattened_size = sample_item.size  # 每个子数组展平后的大小
                dataset_info['shapes'][key] = (flattened_size * batch_size,)  # 每批的展平大小
                dataset_info['dtypes'][key] = 'float32'
                dataset_info['original_shapes'] = dataset_info.get('original_shapes', {})
                dataset_info['original_shapes'][key] = {
                    'outer_shape': original_data.shape,
                    'inner_shape': sample_item.shape,
                    'note': f'嵌套数组: 外层{original_data.shape}, 内层{sample_item.shape}'
                }
            else:
                dataset_info['shapes'][key] = original_data.shape
                dataset_info['dtypes'][key] = str(original_data.dtype)
        
        info_path = tfrecord_path + '.info.json'
        with open(info_path, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        print(f"✅ 转换完成: {tfrecord_path}")
        print(f"📋 数据集信息已保存: {info_path}")
        
        # 清理内存
        del data
        import gc
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        # 清理可能存在的部分文件
        if os.path.exists(tfrecord_path):
            os.remove(tfrecord_path)
        if os.path.exists(tfrecord_path + '.info.json'):
            os.remove(tfrecord_path + '.info.json')
        return False

def convert_npz_to_tfrecord_fast(npz_path, tfrecord_path, batch_size=1000, force=False, 
                                num_workers=None, use_multiprocessing=True, memory_efficient=True):
    """
    优化版本：使用多线程/多进程加速转换
    
    Args:
        npz_path: npz文件路径
        tfrecord_path: 输出TFRecord文件路径
        batch_size: 每批处理的数据量
        force: 是否强制重新转换
        num_workers: 工作线程/进程数，None为自动检测
        use_multiprocessing: 是否使用多进程（否则使用多线程）
        memory_efficient: 是否启用内存优化模式
    
    Returns:
        bool: 转换是否成功
    """
    start_time = time.time()
    
    # 检查输入文件
    if not os.path.exists(npz_path):
        print(f"❌ 输入文件不存在: {npz_path}")
        return False
    
    # 检查是否已存在
    if check_dataset_exists(tfrecord_path) and not force:
        print(f"✅ TFRecord数据集已存在: {tfrecord_path}")
        return True
    
    # 自动检测工作进程数
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 8)  # 限制最大进程数避免内存爆炸
    
    print(f"🚀 快速转换模式:")
    print(f"   - 工作进程数: {num_workers}")
    print(f"   - 多进程模式: {'是' if use_multiprocessing else '否（多线程）'}")
    print(f"   - 内存优化: {'是' if memory_efficient else '否'}")
    print(f"   - 批次大小: {batch_size}")
    
    try:
        print(f"📥 加载npz数据: {npz_path}")
        
        if memory_efficient:
            # 内存优化：使用mmap模式
            data = np.load(npz_path, mmap_mode='r', allow_pickle=True)
        else:
            data = np.load(npz_path, allow_pickle=True)
        
        # 获取字段信息
        keys = list(data.keys())
        target_keys = ['X_paths', 'Y_paths']
        keys = [key for key in keys if key in target_keys]
        
        if not keys:
            print("❌ 错误：未找到需要转换的字段")
            return False
        
        print(f"🎯 转换字段: {keys}")
        
        total_samples = len(data[keys[0]])
        print(f"📊 总样本数: {total_samples}")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(tfrecord_path), exist_ok=True)
        
        # 计算数据块大小
        if memory_efficient and total_samples > 10000:
            # 大数据集：使用小块处理
            chunk_size = min(batch_size, 100)
        else:
            chunk_size = min(batch_size, total_samples // num_workers + 1)
        
        print(f"🔄 开始并行转换 (块大小: {chunk_size})")
        
        # 创建TFRecord写入器
        with tf.io.TFRecordWriter(tfrecord_path) as writer:
            
            if use_multiprocessing and total_samples > 1000:
                # 多进程模式 - 适合CPU密集型任务
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    futures = []
                    
                    # 分块提交任务
                    for i in range(0, total_samples, chunk_size):
                        end_idx = min(i + chunk_size, total_samples)
                        
                        # 准备数据块
                        chunk_data = {}
                        for key in keys:
                            chunk_data[key] = data[key][i:end_idx]
                        
                        # 提交处理任务
                        future = executor.submit(_process_data_chunk, chunk_data, keys, i // chunk_size)
                        futures.append(future)
                    
                    # 收集结果并写入
                    with tqdm(total=len(futures), desc="处理数据块") as pbar:
                        for future in futures:
                            examples = future.result()
                            for example_bytes in examples:
                                writer.write(example_bytes)
                            pbar.update(1)
            
            else:
                # 多线程模式 - 适合I/O密集型任务或小数据集
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    
                    # 创建线程安全的写入锁
                    write_lock = threading.Lock()
                    processed_count = [0]  # 使用列表避免闭包问题
                    
                    def process_and_write(start_idx):
                        end_idx = min(start_idx + chunk_size, total_samples)
                        
                        # 处理数据块
                        chunk_data = {}
                        for key in keys:
                            chunk_data[key] = data[key][start_idx:end_idx]
                        
                        examples = _process_data_chunk(chunk_data, keys, start_idx // chunk_size)
                        
                        # 线程安全写入
                        with write_lock:
                            for example_bytes in examples:
                                writer.write(example_bytes)
                            processed_count[0] += len(examples)
                    
                    # 提交所有任务
                    futures = []
                    for i in range(0, total_samples, chunk_size):
                        future = executor.submit(process_and_write, i)
                        futures.append(future)
                    
                    # 等待完成并显示进度
                    with tqdm(total=total_samples, desc="转换数据") as pbar:
                        last_count = 0
                        for future in futures:
                            future.result()
                            current_count = processed_count[0]
                            pbar.update(current_count - last_count)
                            last_count = current_count
        
        # 保存数据集信息
        dataset_info = {
            'keys': keys,
            'total_samples': total_samples,
            'shapes': {},
            'dtypes': {},
            'created_at': str(np.datetime64('now')),
            'source_file': npz_path,
            'conversion_time': time.time() - start_time,
            'conversion_method': 'multiprocess' if use_multiprocessing else 'multithread',
            'num_workers': num_workers,
            'note': '快速转换版本'
        }
        
        # 计算形状信息
        for key in keys:
            original_data = data[key]
            if hasattr(original_data, 'shape'):
                if len(original_data.shape) > 1:
                    # 多维数组
                    sample_item = original_data[0]
                    if hasattr(sample_item, 'shape'):
                        flattened_size = sample_item.size
                        dataset_info['shapes'][key] = [flattened_size]
                        dataset_info['dtypes'][key] = 'float32'
                        dataset_info['original_shapes'] = dataset_info.get('original_shapes', {})
                        dataset_info['original_shapes'][key] = {
                            'outer_shape': list(original_data.shape),
                            'inner_shape': list(sample_item.shape),
                            'note': f'多维数组: 外层{original_data.shape}, 内层{sample_item.shape}'
                        }
                    else:
                        dataset_info['shapes'][key] = list(original_data.shape)
                        dataset_info['dtypes'][key] = str(original_data.dtype)
                else:
                    dataset_info['shapes'][key] = list(original_data.shape)
                    dataset_info['dtypes'][key] = str(original_data.dtype)
        
        # 保存信息文件
        info_path = tfrecord_path + '.info.json'
        with open(info_path, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        elapsed_time = time.time() - start_time
        print(f"✅ 快速转换完成: {tfrecord_path}")
        print(f"⏱️ 转换耗时: {elapsed_time:.2f}秒")
        print(f"📈 转换速度: {total_samples/elapsed_time:.1f} 样本/秒")
        print(f"📋 数据集信息: {info_path}")
        
        # 清理内存
        del data
        import gc
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"❌ 快速转换失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 清理失败的文件
        for file_path in [tfrecord_path, tfrecord_path + '.info.json']:
            if os.path.exists(file_path):
                os.remove(file_path)
        return False

def convert_npz_to_tfrecord_streaming(npz_path, tfrecord_path, chunk_size=100, force=False, 
                                     num_workers=None, buffer_size=10, progress_callback=None):
    """
    流式处理版本：逐块加载和转换，适合超大数据集
    
    Args:
        npz_path: npz文件路径
        tfrecord_path: 输出TFRecord文件路径
        chunk_size: 每次流式处理的块大小
        force: 是否强制重新转换
        num_workers: 工作线程数
        buffer_size: 内存缓冲区大小（块数）
        progress_callback: 进度回调函数
    
    Returns:
        bool: 转换是否成功
    """
    start_time = time.time()
    
    # 检查输入文件
    if not os.path.exists(npz_path):
        print(f"❌ 输入文件不存在: {npz_path}")
        return False
    
    # 检查是否已存在
    if check_dataset_exists(tfrecord_path) and not force:
        print(f"✅ TFRecord数据集已存在: {tfrecord_path}")
        return True
    
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 4)  # 流式处理使用较少的工作线程
    
    print(f"🌊 流式转换模式:")
    print(f"   - 流式块大小: {chunk_size}")
    print(f"   - 工作线程数: {num_workers}")
    print(f"   - 内存缓冲区: {buffer_size} 块")
    
    try:
        # 第一次打开获取基本信息
        print(f"📊 分析数据集结构: {npz_path}")
        with np.load(npz_path, allow_pickle=True) as data:
            keys = list(data.keys())
            target_keys = ['X_paths', 'Y_paths']
            keys = [key for key in keys if key in target_keys]
            
            if not keys:
                print("❌ 错误：未找到需要转换的字段")
                return False
            
            total_samples = len(data[keys[0]])
            
            # 获取数据类型和形状信息
            sample_shapes = {}
            sample_dtypes = {}
            for key in keys:
                sample_data = data[key][0]
                if hasattr(sample_data, 'shape'):
                    sample_shapes[key] = sample_data.shape
                    sample_dtypes[key] = sample_data.dtype
                else:
                    sample_shapes[key] = ()
                    sample_dtypes[key] = type(sample_data)
        
        print(f"🎯 流式转换字段: {keys}")
        print(f"📊 总样本数: {total_samples}")
        print(f"💾 预计处理块数: {(total_samples + chunk_size - 1) // chunk_size}")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(tfrecord_path), exist_ok=True)
        
        # 流式处理状态
        processed_samples = 0
        memory_usage = []
        
        # 创建TFRecord写入器
        with tf.io.TFRecordWriter(tfrecord_path) as writer:
            
            # 使用队列进行生产者-消费者模式的流式处理
            from queue import Queue
            
            # 数据块队列
            data_queue = Queue(maxsize=buffer_size)
            result_queue = Queue()
            
            def data_producer():
                """数据生产者：流式读取数据块"""
                try:
                    # 使用memory mapping避免加载整个数据集
                    data = np.load(npz_path, mmap_mode='r', allow_pickle=True)
                    
                    for i in range(0, total_samples, chunk_size):
                        end_idx = min(i + chunk_size, total_samples)
                        
                        # 只读取当前块的数据
                        chunk_data = {}
                        for key in keys:
                            chunk_data[key] = data[key][i:end_idx].copy()  # copy避免mmap引用
                        
                        data_queue.put((i // chunk_size, chunk_data))
                        
                        # 监控内存使用
                        if psutil and len(memory_usage) < 10:  # 只记录前10个块的内存使用
                            memory_usage.append(psutil.virtual_memory().percent)
                    
                    # 发送结束信号
                    data_queue.put(None)
                    
                except Exception as e:
                    print(f"❌ 数据生产者错误: {e}")
                    data_queue.put(None)
            
            def data_consumer():
                """数据消费者：处理数据块并生成TFRecord样本"""
                try:
                    while True:
                        item = data_queue.get()
                        if item is None:  # 结束信号
                            result_queue.put(None)
                            break
                        
                        chunk_id, chunk_data = item
                        
                        # 处理当前块
                        examples = _process_data_chunk(chunk_data, keys, chunk_id)
                        result_queue.put((chunk_id, examples))
                        
                        # 释放内存
                        del chunk_data
                        
                        data_queue.task_done()
                        
                except Exception as e:
                    print(f"❌ 数据消费者错误: {e}")
                    result_queue.put(None)
            
            # 启动生产者和消费者线程
            from threading import Thread
            
            producer_thread = Thread(target=data_producer)
            consumer_thread = Thread(target=data_consumer)
            
            producer_thread.start()
            consumer_thread.start()
            
            # 主线程：收集结果并写入TFRecord
            last_report_time = time.time()
            chunk_times = []
            
            with tqdm(total=total_samples, desc="流式转换", 
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
                while True:
                    result = result_queue.get()
                    if result is None:  # 结束信号
                        break
                    
                    chunk_start_time = time.time()
                    chunk_id, examples = result
                    
                    # 写入TFRecord
                    for example_bytes in examples:
                        writer.write(example_bytes)
                    
                    processed_samples += len(examples)
                    pbar.update(len(examples))
                    
                    # 性能监控
                    chunk_time = time.time() - chunk_start_time
                    chunk_times.append(chunk_time)
                    
                    # 每10个chunk或每30秒报告一次详细进度
                    current_time = time.time()
                    if chunk_id % 10 == 0 or current_time - last_report_time > 30:
                        avg_chunk_time = np.mean(chunk_times[-10:]) if chunk_times else 0
                        samples_per_sec = len(examples) / avg_chunk_time if avg_chunk_time > 0 else 0
                        remaining_chunks = (total_samples - processed_samples) // chunk_size
                        eta_minutes = (remaining_chunks * avg_chunk_time) / 60 if avg_chunk_time > 0 else 0
                        
                        print(f"\n📊 进度报告 - 块 {chunk_id+1}/{(total_samples-1)//chunk_size+1}:")
                        print(f"   ✅ 已处理: {processed_samples:,}/{total_samples:,} 样本 ({processed_samples/total_samples*100:.1f}%)")
                        print(f"   ⚡ 处理速度: {samples_per_sec:.1f} 样本/秒")
                        print(f"   ⏱️ 预计剩余: {eta_minutes:.1f} 分钟")
                        if psutil:
                            memory_percent = psutil.virtual_memory().percent
                            print(f"   🧠 内存使用: {memory_percent:.1f}%")
                        last_report_time = current_time
                    
                    # 调用进度回调
                    if progress_callback:
                        progress_callback(processed_samples, total_samples, chunk_id)
                    
                    result_queue.task_done()
            
            # 等待线程结束
            producer_thread.join()
            consumer_thread.join()
        
        # 保存数据集信息
        dataset_info = {
            'keys': keys,
            'total_samples': total_samples,
            'shapes': {},
            'dtypes': {},
            'created_at': str(np.datetime64('now')),
            'source_file': npz_path,
            'conversion_time': time.time() - start_time,
            'conversion_method': 'streaming',
            'chunk_size': chunk_size,
            'num_workers': num_workers,
            'buffer_size': buffer_size,
            'memory_usage_samples': memory_usage,
            'note': '流式转换版本 - 内存友好'
        }
        
        # 计算形状信息
        for key in keys:
            if key in sample_shapes:
                if len(sample_shapes[key]) > 0:
                    flattened_size = np.prod(sample_shapes[key])
                    dataset_info['shapes'][key] = [flattened_size]
                    dataset_info['dtypes'][key] = 'float32'
                    dataset_info['original_shapes'] = dataset_info.get('original_shapes', {})
                    dataset_info['original_shapes'][key] = {
                        'outer_shape': [total_samples] + list(sample_shapes[key]),
                        'inner_shape': list(sample_shapes[key]),
                        'note': f'流式处理: 形状{sample_shapes[key]}'
                    }
                else:
                    dataset_info['shapes'][key] = []
                    dataset_info['dtypes'][key] = str(sample_dtypes[key])
        
        # 保存信息文件
        info_path = tfrecord_path + '.info.json'
        with open(info_path, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        elapsed_time = time.time() - start_time
        avg_memory = np.mean(memory_usage) if memory_usage else 0
        
        print(f"✅ 流式转换完成: {tfrecord_path}")
        print(f"⏱️ 转换耗时: {elapsed_time:.2f}秒")
        print(f"📈 转换速度: {total_samples/elapsed_time:.1f} 样本/秒")
        print(f"🧠 平均内存使用率: {avg_memory:.1f}%")
        print(f"📋 数据集信息: {info_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ 流式转换失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 清理失败的文件
        for file_path in [tfrecord_path, tfrecord_path + '.info.json']:
            if os.path.exists(file_path):
                os.remove(file_path)
        return False

def convert_npz_to_tfrecord_adaptive(npz_path, tfrecord_path, force=False, 
                                    max_memory_gb=8, auto_optimize=True):
    """
    自适应转换：根据数据集大小和系统资源自动选择最佳转换方法
    
    Args:
        npz_path: npz文件路径
        tfrecord_path: 输出TFRecord文件路径
        force: 是否强制重新转换
        max_memory_gb: 最大内存使用限制（GB）
        auto_optimize: 是否自动优化参数
    
    Returns:
        bool: 转换是否成功
    """
    if not psutil:
        print("❌ 错误: 自适应转换需要psutil库，请安装: pip install psutil")
        return False
    
    # 获取系统信息
    total_memory_gb = psutil.virtual_memory().total / (1024**3)
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    cpu_count = mp.cpu_count()
    
    # 获取数据集大小
    file_size_gb = os.path.getsize(npz_path) / (1024**3)
    
    print(f"🤖 自适应转换分析:")
    print(f"   - 数据集大小: {file_size_gb:.2f} GB")
    print(f"   - 系统总内存: {total_memory_gb:.2f} GB")
    print(f"   - 可用内存: {available_memory_gb:.2f} GB")
    print(f"   - CPU核心数: {cpu_count}")
    print(f"   - 内存限制: {max_memory_gb:.2f} GB")
    
    # 先快速检查样本数量以做更精确的决策
    try:
        with np.load(npz_path, allow_pickle=True) as data:
            sample_count = len(data['X_paths']) if 'X_paths' in data else 0
        print(f"   - 样本数量: {sample_count}")
    except:
        sample_count = 0
    
    # 优化的决策逻辑 - 针对大文件优化
    if sample_count < 1000:  # 小数据集
        print("🚀 选择：原始转换（小数据集）")
        return convert_npz_to_tfrecord(npz_path, tfrecord_path, batch_size=1000, force=force)
    
    elif sample_count < 5000 and available_memory_gb > 4:  # 中等数据集且内存充足
        print("🚀 选择：多进程快速转换（优化大文件）")
        num_workers = min(cpu_count, 6)  # 适中的进程数
        batch_size = max(50, min(500, sample_count // 20))  # 动态调整batch_size
        print(f"   - 优化参数: workers={num_workers}, batch_size={batch_size}")
        return convert_npz_to_tfrecord_fast(
            npz_path, tfrecord_path, 
            batch_size=batch_size, force=force,
            num_workers=num_workers, use_multiprocessing=True, 
            memory_efficient=True
        )
    
    elif sample_count < 20000 and available_memory_gb > 2:  # 大数据集但内存可用
        print("🚀 选择：多线程快速转换（内存优化）")
        num_workers = min(cpu_count // 2, 4)
        batch_size = max(100, min(1000, sample_count // 50))  # 更大的batch_size提高效率
        print(f"   - 优化参数: workers={num_workers}, batch_size={batch_size}")
        return convert_npz_to_tfrecord_fast(
            npz_path, tfrecord_path,
            batch_size=batch_size, force=force,
            num_workers=num_workers, use_multiprocessing=False,
            memory_efficient=True
        )
    
    else:  # 超大数据集或内存严重不足
        print("🌊 选择：流式转换（超大数据集/内存受限）")
        # 针对大文件优化chunk_size
        if auto_optimize and sample_count > 0:
            # 目标：每个chunk处理时间约1-2秒，内存使用<500MB
            target_chunk_size = max(200, min(2000, sample_count // 100))
            chunk_size = target_chunk_size
        else:
            chunk_size = 500  # 增大默认chunk_size提高效率
        
        buffer_size = max(3, min(8, int(available_memory_gb)))  # 根据可用内存调整
        num_workers = 2  # 流式处理使用少量线程避免竞争
        
        print(f"   - 优化参数: chunk_size={chunk_size}, buffer_size={buffer_size}")
        print(f"   - 预计转换时间: {sample_count/chunk_size/10:.1f}-{sample_count/chunk_size/5:.1f} 分钟")
        
        return convert_npz_to_tfrecord_streaming(
            npz_path, tfrecord_path,
            chunk_size=chunk_size, force=force,
            num_workers=num_workers, buffer_size=buffer_size
        )

def load_tfrecord_dataset(tfrecord_path, batch_size=32, shuffle_buffer=1000, auto_convert=True, state_dim=None):
    """
    加载TFRecord数据集
    
    Args:
        tfrecord_path: TFRecord文件路径
        batch_size: 批次大小
        shuffle_buffer: 随机打乱缓冲区大小
        auto_convert: 如果数据集不存在，是否自动从npz转换
        state_dim: 状态向量维度，如果为None则使用默认值8
    
    Returns:
        tf.data.Dataset: TensorFlow数据集
    """
    # 检查数据集是否存在
    if not check_dataset_exists(tfrecord_path):
        if auto_convert:
            # 尝试从npz转换
            npz_path = tfrecord_path.replace('.tfrecord', '.npz')
            if os.path.exists(npz_path):
                print(f"🔄 自动转换数据集: {npz_path}")
                if not convert_npz_to_tfrecord(npz_path, tfrecord_path):
                    raise FileNotFoundError(f"数据集转换失败: {npz_path}")
            else:
                raise FileNotFoundError(f"找不到数据集文件: {tfrecord_path} 或 {npz_path}")
        else:
            raise FileNotFoundError(f"找不到数据集文件: {tfrecord_path}")
    
    try:
        # 加载数据集信息
        info_path = tfrecord_path + '.info.json'
        with open(info_path, 'r') as f:
            dataset_info = json.load(f)
        
        # 动态创建特征描述
        feature_description = {}
        for key in dataset_info['keys']:
            if dataset_info['shapes'][key] == ():  # 标量值
                feature_description[key] = tf.io.FixedLenFeature([], tf.float32)
            else:  # 数组
                feature_description[key] = tf.io.FixedLenFeature([], tf.string)
        
        def _parse_function(example_proto):
            # 解析TFRecord
            parsed_features = tf.io.parse_single_example(example_proto, feature_description)
            
            # 将bytes转换回numpy数组
            decoded_data = {}
            for key in dataset_info['keys']:
                if dataset_info['dtypes'][key] == 'float32':  # 嵌套numpy数组已转换为float32
                    # 解码为float32数组
                    decoded = tf.io.decode_raw(parsed_features[key], tf.float32)
                    decoded_data[key] = decoded
                elif dataset_info['shapes'][key] == ():  # 标量值
                    decoded_data[key] = parsed_features[key]
                else:  # 其他数组类型
                    decoded_data[key] = tf.io.decode_raw(parsed_features[key], tf.float32)
            
            # 根据数据集信息重构数据
            X_data = decoded_data['X_paths']  # (6000,) -> (1000, 6)
            Y_data = decoded_data['Y_paths']  # (1000,) -> (1000, 1)
            
            # 重构为正确的形状
            if 'original_shapes' in dataset_info:
                X_inner_shape = dataset_info['original_shapes']['X_paths']['inner_shape']  # [1000, 6]
                Y_inner_shape = dataset_info['original_shapes']['Y_paths']['inner_shape']  # [1000, 1]
                
                X_data_reshaped = tf.reshape(X_data, X_inner_shape)  # (1000, 6)
                Y_data_reshaped = tf.reshape(Y_data, Y_inner_shape)  # (1000, 1)
            else:
                # 后备方案：根据记录的形状推断
                # X_paths: 6000 -> (1000, 6), Y_paths: 1000 -> (1000, 1)
                X_data_reshaped = tf.reshape(X_data, [1000, 6])
                Y_data_reshaped = tf.reshape(Y_data, [1000, 1])
            
            # 创建初始状态（零状态）
            # 使用传入的 state_dim 参数，如果没有则使用默认值
            actual_state_dim = state_dim if state_dim is not None else 8
            init_state = tf.zeros((actual_state_dim,), dtype=tf.float32)
            
            # 返回模型期望的格式
            inputs = {
                'delta_input': X_data_reshaped,  # (1000, 6)
                'init_state': init_state         # (8,)
            }
            
            return inputs, Y_data_reshaped  # (1000, 1)
        
        # 创建数据集
        dataset = tf.data.TFRecordDataset(tfrecord_path)
        dataset = dataset.map(_parse_function)
        
        # 设置数据集参数
        dataset = dataset.shuffle(shuffle_buffer)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
        
    except Exception as e:
        print(f"❌ 加载数据集失败: {e}")
        raise

def split_and_convert_npz(npz_path, output_dir, samples_per_chunk=1000, 
                         target_keys=None, force=False, cleanup_splits=True):
    """
    将大的NPZ文件分割成小块并分别转换为TFRecord
    
    Args:
        npz_path: 原始NPZ文件路径
        output_dir: 输出目录
        samples_per_chunk: 每个分块的样本数
        target_keys: 要转换的字段列表，None表示转换所有字段
        force: 是否强制重新转换
        cleanup_splits: 转换完成后是否删除临时分割文件
    
    Returns:
        bool: 转换是否成功
    """
    start_time = time.time()
    
    # 检查输入文件
    if not os.path.exists(npz_path):
        print(f"❌ 输入文件不存在: {npz_path}")
        return False
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 最终合并的TFRecord文件路径
    final_tfrecord = os.path.join(output_dir, 'dataset.tfrecord')
    final_info = final_tfrecord + '.info.json'
    
    # 检查是否已存在最终文件
    if os.path.exists(final_tfrecord) and os.path.exists(final_info) and not force:
        print(f"✅ 转换后的数据集已存在: {final_tfrecord}")
        return True
    
    try:
        print(f"📊 分析NPZ文件: {npz_path}")
        # 获取文件大小
        file_size_mb = os.path.getsize(npz_path) / (1024 * 1024)
        print(f"📁 文件大小: {file_size_mb:.1f} MB")
        
        # 第一步：分析数据结构
        with np.load(npz_path, allow_pickle=True) as data:
            all_keys = list(data.keys())
            
            # 确定要转换的字段
            if target_keys is None:
                target_keys = ['X_paths', 'Y_paths']
            
            keys = [key for key in all_keys if key in target_keys]
            if not keys:
                print(f"❌ 错误：未找到目标字段 {target_keys}")
                print(f"📋 可用字段: {all_keys}")
                return False
            
            print(f"🎯 将转换字段: {keys}")
            
            # 获取总样本数
            total_samples = len(data[keys[0]])
            print(f"📊 总样本数: {total_samples}")
            
            # 计算分块信息
            num_chunks = (total_samples + samples_per_chunk - 1) // samples_per_chunk
            print(f"🧩 将分割为 {num_chunks} 个块，每块约 {samples_per_chunk} 样本")
            
            # 获取数据类型信息
            sample_info = {}
            for key in keys:
                sample_data = data[key][0]
                if hasattr(sample_data, 'shape') and hasattr(sample_data, 'dtype'):
                    sample_info[key] = {
                        'shape': sample_data.shape,
                        'dtype': sample_data.dtype,
                        'is_nested': sample_data.dtype == object
                    }
                    print(f"   {key}: shape={sample_data.shape}, dtype={sample_data.dtype}")
                else:
                    sample_info[key] = {
                        'shape': (),
                        'dtype': type(sample_data),
                        'is_nested': False
                    }
        
        # 第二步：创建分割目录
        splits_dir = os.path.join(output_dir, 'temp_splits')
        os.makedirs(splits_dir, exist_ok=True)
        
        print(f"🔪 开始分割NPZ文件...")
        
        # 分割文件列表
        chunk_files = []
        
        # 使用内存映射模式分割文件
        data = np.load(npz_path, mmap_mode='r', allow_pickle=True)
        
        for chunk_id in tqdm(range(num_chunks), desc="分割文件"):
            start_idx = chunk_id * samples_per_chunk
            end_idx = min(start_idx + samples_per_chunk, total_samples)
            
            # 创建分块文件路径
            chunk_file = os.path.join(splits_dir, f'chunk_{chunk_id:04d}.npz')
            chunk_files.append(chunk_file)
            
            # 如果分块文件已存在且不强制重建，跳过
            if os.path.exists(chunk_file) and not force:
                continue
            
            # 提取当前块的数据
            chunk_data = {}
            for key in keys:
                chunk_data[key] = data[key][start_idx:end_idx]
                
                # 对于嵌套数组，需要特殊处理以避免引用问题
                if sample_info[key]['is_nested']:
                    # 创建独立的副本
                    chunk_data[key] = np.array([item.copy() if hasattr(item, 'copy') else item 
                                              for item in chunk_data[key]], dtype=object)
            
            # 保存分块
            np.savez_compressed(chunk_file, **chunk_data)
        
        # 清理原始数据引用
        del data
        import gc
        gc.collect()
        
        print(f"✅ 文件分割完成，共 {len(chunk_files)} 个分块")
        
        # 第三步：并行转换分块
        print(f"🚀 开始并行转换分块...")
        
        def convert_chunk(chunk_info):
            chunk_id, chunk_file = chunk_info
            tfrecord_file = chunk_file.replace('.npz', '.tfrecord')
            
            # 使用简单的转换方法
            success = convert_npz_to_tfrecord(
                chunk_file, tfrecord_file, 
                batch_size=min(100, samples_per_chunk), 
                force=force
            )
            
            if success:
                return tfrecord_file
            else:
                print(f"❌ 分块 {chunk_id} 转换失败")
                return None
        
        # 并行转换所有分块
        max_workers = min(mp.cpu_count(), 4)  # 限制并发数避免资源竞争
        chunk_infos = [(i, chunk_file) for i, chunk_file in enumerate(chunk_files)]
        
        tfrecord_files = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(
                executor.map(convert_chunk, chunk_infos),
                total=len(chunk_infos),
                desc="转换分块"
            ))
            
            tfrecord_files = [f for f in results if f is not None]
        
        if len(tfrecord_files) != len(chunk_files):
            print(f"❌ 部分分块转换失败: {len(tfrecord_files)}/{len(chunk_files)}")
            return False
        
        print(f"✅ 所有分块转换完成")
        
        # 第四步：合并TFRecord文件
        print(f"🔗 合并TFRecord文件...")
        
        with tf.io.TFRecordWriter(final_tfrecord) as final_writer:
            total_records = 0
            
            for tfrecord_file in tqdm(tfrecord_files, desc="合并文件"):
                # 读取并写入每个分块的记录
                for record in tf.data.TFRecordDataset(tfrecord_file):
                    final_writer.write(record.numpy())
                    total_records += 1
        
        print(f"✅ TFRecord合并完成，共 {total_records} 条记录")
        
        # 第五步：创建合并后的信息文件
        final_dataset_info = {
            'keys': keys,
            'total_samples': total_samples,
            'total_records': total_records,
            'shapes': {},
            'dtypes': {},
            'created_at': str(np.datetime64('now')),
            'source_file': npz_path,
            'conversion_time': time.time() - start_time,
            'conversion_method': 'split_and_convert',
            'num_chunks': num_chunks,
            'samples_per_chunk': samples_per_chunk,
            'note': '分割转换版本 - 内存友好且高效'
        }
        
        # 从第一个分块的信息文件中获取详细信息
        if tfrecord_files:
            first_info_file = tfrecord_files[0] + '.info.json'
            if os.path.exists(first_info_file):
                with open(first_info_file, 'r') as f:
                    first_info = json.load(f)
                    final_dataset_info['shapes'] = first_info.get('shapes', {})
                    final_dataset_info['dtypes'] = first_info.get('dtypes', {})
                    final_dataset_info['original_shapes'] = first_info.get('original_shapes', {})
        
        # 保存最终信息文件
        with open(final_info, 'w') as f:
            json.dump(final_dataset_info, f, indent=2)
        
        # 第六步：清理临时文件
        if cleanup_splits:
            print(f"🧹 清理临时文件...")
            try:
                shutil.rmtree(splits_dir)
                print(f"✅ 临时文件清理完成")
            except Exception as e:
                print(f"⚠️ 清理临时文件时出错: {e}")
        else:
            print(f"💾 临时文件保留在: {splits_dir}")
        
        # 性能统计
        elapsed_time = time.time() - start_time
        conversion_rate = total_samples / elapsed_time
        
        print(f"\n🎉 分割转换完成！")
        print(f"📁 输出文件: {final_tfrecord}")
        print(f"📊 转换统计:")
        print(f"   ⏱️ 总耗时: {elapsed_time:.2f}秒")
        print(f"   📈 转换速度: {conversion_rate:.1f} 样本/秒")
        print(f"   🧩 分块数量: {num_chunks}")
        print(f"   📦 最终文件大小: {os.path.getsize(final_tfrecord)/(1024*1024):.1f} MB")
        print(f"   💾 压缩比: {file_size_mb/(os.path.getsize(final_tfrecord)/(1024*1024)):.2f}:1")
        
        return True
        
    except Exception as e:
        print(f"❌ 分割转换失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 清理失败的文件
        if os.path.exists(final_tfrecord):
            os.remove(final_tfrecord)
        if os.path.exists(final_info):
            os.remove(final_info)
        
        return False

def smart_convert_npz_to_tfrecord(npz_path, output_path, method='auto', **kwargs):
    """
    智能转换：根据文件大小自动选择最优转换方法
    
    Args:
        npz_path: NPZ文件路径
        output_path: 输出路径（文件或目录）
        method: 转换方法 ('auto', 'simple', 'fast', 'streaming', 'adaptive', 'split')
        **kwargs: 其他参数
    
    Returns:
        bool: 转换是否成功
    """
    # 获取文件大小
    file_size_mb = os.path.getsize(npz_path) / (1024 * 1024)
    
    # 获取系统内存
    if psutil:
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
    else:
        available_memory_gb = 4  # 默认假设4GB可用内存
    
    print(f"🤖 智能转换分析:")
    print(f"   📁 文件大小: {file_size_mb:.1f} MB")
    print(f"   🧠 可用内存: {available_memory_gb:.1f} GB")
    
    # 自动选择转换方法
    if method == 'auto':
        if file_size_mb < 50:  # 小文件 (<50MB)
            method = 'simple'
        elif file_size_mb < 500 and available_memory_gb > 2:  # 中等文件且内存充足
            method = 'fast'
        elif file_size_mb < 2000 and available_memory_gb > 1:  # 大文件但内存够用
            method = 'adaptive'
        else:  # 超大文件或内存不足
            method = 'split'
    
    print(f"🚀 选择转换方法: {method}")
    
    # 根据方法执行转换
    if method == 'simple':
        return convert_npz_to_tfrecord(npz_path, output_path, **kwargs)
    
    elif method == 'fast':
        return convert_npz_to_tfrecord_fast(npz_path, output_path, **kwargs)
    
    elif method == 'streaming':
        return convert_npz_to_tfrecord_streaming(npz_path, output_path, **kwargs)
    
    elif method == 'adaptive':
        return convert_npz_to_tfrecord_adaptive(npz_path, output_path, **kwargs)
    
    elif method == 'split':
        # 对于分割方法，output_path应该是目录
        if output_path.endswith('.tfrecord'):
            output_dir = os.path.dirname(output_path)
        else:
            output_dir = output_path
        
        return split_and_convert_npz(
            npz_path, output_dir, 
            samples_per_chunk=kwargs.get('samples_per_chunk', 1000),
            target_keys=kwargs.get('target_keys', ['X_paths', 'Y_paths']),
            force=kwargs.get('force', False),
            cleanup_splits=kwargs.get('cleanup_splits', True)
        )
    
    else:
        print(f"❌ 未知的转换方法: {method}")
        return False

def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='转换npz数据集为TFRecord格式')
    parser.add_argument('--npz_path', type=str, required=True,
                      help='输入npz文件路径')
    parser.add_argument('--tfrecord_path', type=str, required=True,
                      help='输出TFRecord文件路径')
    parser.add_argument('--batch_size', type=int, default=1000,
                      help='转换时的批次大小')
    parser.add_argument('--force', action='store_true',
                      help='强制重新转换（即使已存在）')
    args = parser.parse_args()
    
    convert_npz_to_tfrecord(args.npz_path, args.tfrecord_path, args.batch_size, args.force)

if __name__ == '__main__':
    # main() 

    name = 'dataset_EMSC_tt'
    npz_path = f'/Users/tianyunhu/Documents/temp/code/Test_app/EMSC_Model/msc_models/{name}/{name}.npz'
    tfrecord_path = f'/Users/tianyunhu/Documents/temp/code/Test_app/EMSC_Model/msc_models/{name}/{name}.tfrecord'
    
    print("🚀 测试所有转换方法...")
    
    def progress_callback(processed, total, chunk_id):
        """进度回调函数"""
        if chunk_id % 5 == 0:  # 每5个块报告一次
            print(f"   📊 已处理: {processed}/{total} 样本 ({processed/total*100:.1f}%)")
    
    # 测试不同的转换方法
    # print("\n=== 1. 测试流式转换 ===")
    # convert_npz_to_tfrecord_streaming(
    #     npz_path, tfrecord_path + '_streaming', 
    #     chunk_size=20, force=True, 
    #     num_workers=2, buffer_size=5,
    #     progress_callback=progress_callback
    # )
    
    # print("\n=== 2. 测试自适应转换 ===")
    convert_npz_to_tfrecord_adaptive(
        npz_path, tfrecord_path,
        force=True, max_memory_gb=4, auto_optimize=True
    )
    
    # print("\n=== 测试分割转换（推荐用于大文件）===")
    # output_dir = os.path.dirname(tfrecord_path)
    # success = split_and_convert_npz(
    #     npz_path, output_dir,
    #     samples_per_chunk=20,  # 每块20个样本，适合测试
    #     target_keys=['X_paths', 'Y_paths'],
    #     force=True,
    #     cleanup_splits=True  # 转换完成后清理临时文件
    # )
    
    # if success:
    #     print("\n=== 测试智能转换 ===")
    #     # 测试智能转换（自动选择最优方法）
    #     smart_convert_npz_to_tfrecord(
    #         npz_path, 
    #         os.path.join(output_dir, 'smart_dataset.tfrecord'),
    #         method='auto',  # 自动选择方法
    #         force=True
    #     )

