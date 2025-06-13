"""
CPU压力测试模块
用于诊断CPU使用率低的问题，找出真正的瓶颈
"""

import os
import time
import threading
import numpy as np
import tensorflow as tf
from typing import Dict, List


def cpu_intensive_test(duration=30, num_threads=None):
    """
    CPU密集型计算测试
    
    Args:
        duration: 测试持续时间（秒）
        num_threads: 线程数，None为自动检测
    """
    if num_threads is None:
        num_threads = os.cpu_count()
    
    print(f"🔥 启动CPU压力测试 ({duration}秒, {num_threads}线程)")
    
    def cpu_worker():
        """CPU密集型计算工作线程"""
        end_time = time.time() + duration
        while time.time() < end_time:
            # 矩阵乘法计算
            a = np.random.random((100, 100))
            b = np.random.random((100, 100))
            np.dot(a, b)
    
    # 启动工作线程
    threads = []
    start_time = time.time()
    
    for i in range(num_threads):
        t = threading.Thread(target=cpu_worker)
        t.start()
        threads.append(t)
    
    # 监控CPU使用率
    try:
        import psutil
        while time.time() - start_time < duration:
            cpu_percent = psutil.cpu_percent(interval=1)
            print(f"CPU使用率: {cpu_percent:.1f}%")
    except ImportError:
        print("未安装psutil，无法监控CPU使用率")
        time.sleep(duration)
    
    # 等待所有线程完成
    for t in threads:
        t.join()
    
    print("✅ CPU压力测试完成")


def tensorflow_compute_test(batch_sizes=[32, 64, 128, 256], duration=60):
    """
    TensorFlow计算测试
    
    Args:
        batch_sizes: 要测试的批次大小列表
        duration: 每个批次大小的测试时间
    """
    print(f"🧪 TensorFlow计算性能测试")
    
    results = {}
    
    for batch_size in batch_sizes:
        print(f"\n测试批次大小: {batch_size}")
        
        # 创建测试数据
        x = tf.random.normal((batch_size, 100))
        
        # 创建简单的神经网络
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        
        model.build(input_shape=(None, 100))
        
        # 测试前向传播
        start_time = time.time()
        iterations = 0
        
        while time.time() - start_time < duration:
            _ = model(x)
            iterations += 1
        
        elapsed = time.time() - start_time
        ops_per_sec = iterations / elapsed
        
        results[batch_size] = {
            'iterations': iterations,
            'ops_per_sec': ops_per_sec,
            'elapsed': elapsed
        }
        
        print(f"   迭代次数: {iterations}")
        print(f"   每秒操作数: {ops_per_sec:.2f}")
    
    print(f"\n📊 TensorFlow性能测试结果:")
    for batch_size, result in results.items():
        print(f"批次大小 {batch_size}: {result['ops_per_sec']:.2f} ops/sec")
    
    return results


def diagnose_cpu_bottleneck():
    """
    诊断CPU瓶颈
    """
    print(f"🔍 开始CPU瓶颈诊断...")
    
    try:
        import psutil
        
        # 系统信息
        print(f"\n💻 系统信息:")
        print(f"CPU核心数: {psutil.cpu_count(logical=False)} 物理, {psutil.cpu_count(logical=True)} 逻辑")
        
        memory = psutil.virtual_memory()
        print(f"内存: {memory.total / (1024**3):.1f}GB 总量, {memory.available / (1024**3):.1f}GB 可用")
        
        # CPU使用率分布
        print(f"\n📈 当前CPU状态:")
        cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
        for i, usage in enumerate(cpu_percent):
            print(f"CPU{i}: {usage:.1f}%")
        
        avg_usage = sum(cpu_percent) / len(cpu_percent)
        print(f"平均使用率: {avg_usage:.1f}%")
        
        # 进程信息
        print(f"\n⚡ 进程信息:")
        current_process = psutil.Process()
        print(f"当前进程CPU使用率: {current_process.cpu_percent()}%")
        print(f"当前进程内存使用: {current_process.memory_info().rss / (1024**2):.1f}MB")
        print(f"当前进程线程数: {current_process.num_threads()}")
        
        # I/O统计
        io_counters = current_process.io_counters()
        print(f"读取字节数: {io_counters.read_bytes / (1024**2):.1f}MB")
        print(f"写入字节数: {io_counters.write_bytes / (1024**2):.1f}MB")
        
        # 建议
        print(f"\n💡 优化建议:")
        if avg_usage < 50:
            print("   - CPU使用率较低，可能的原因：")
            print("     * 数据I/O成为瓶颈")
            print("     * 内存带宽限制")
            print("     * TensorFlow配置不当")
            print("     * 批次大小过小")
        
        if memory.percent > 80:
            print("   - 内存使用率过高，可能影响性能")
        
        # 环境变量检查
        print(f"\n🔧 TensorFlow环境变量:")
        tf_vars = [
            'TF_NUM_INTEROP_THREADS',
            'TF_NUM_INTRAOP_THREADS', 
            'OMP_NUM_THREADS',
            'MKL_NUM_THREADS',
            'TF_ENABLE_ONEDNN_OPTS'
        ]
        
        for var in tf_vars:
            value = os.environ.get(var, 'Not set')
            print(f"   {var}: {value}")
            
    except ImportError:
        print("未安装psutil，无法进行详细诊断")


def comprehensive_performance_test():
    """
    综合性能测试
    """
    print("🚀 开始综合性能测试...")
    
    # 1. 基础CPU测试
    print("\n1️⃣ 基础CPU压力测试")
    cpu_intensive_test(duration=15)
    
    # 2. TensorFlow计算测试
    print("\n2️⃣ TensorFlow计算测试")
    tensorflow_compute_test(batch_sizes=[32, 128, 256], duration=30)
    
    # 3. 系统诊断
    print("\n3️⃣ 系统瓶颈诊断")
    diagnose_cpu_bottleneck()
    
    print("\n✅ 综合性能测试完成")


if __name__ == "__main__":
    comprehensive_performance_test()