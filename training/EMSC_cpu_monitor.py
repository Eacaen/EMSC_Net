"""
CPU使用率监控模块
用于监控阿里云等CPU训练环境的资源使用情况
"""

import threading
import time
import tensorflow as tf

class CPUMonitorCallback(tf.keras.callbacks.Callback):
    """CPU使用率监控回调"""
    
    def __init__(self, monitor_interval=30, verbose=True):
        """
        Args:
            monitor_interval: 监控间隔（秒）
            verbose: 是否打印详细信息
        """
        super().__init__()
        self.monitor_interval = monitor_interval
        self.verbose = verbose
        self.monitoring = False
        self.monitor_thread = None
        
    def on_train_begin(self, logs=None):
        """训练开始时启动监控"""
        try:
            import psutil
            self.psutil = psutil
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_cpu)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            if self.verbose:
                print("启动CPU使用率监控...")
        except ImportError:
            if self.verbose:
                print("警告: 未安装psutil，无法监控CPU使用率")
    
    def on_train_end(self, logs=None):
        """训练结束时停止监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        if self.verbose:
            print("停止CPU使用率监控")
    
    def _monitor_cpu(self):
        """CPU监控主循环"""
        while self.monitoring:
            try:
                # 获取CPU使用率（所有核心的平均值）
                cpu_percent = self.psutil.cpu_percent(interval=1)
                
                # 获取每个CPU核心的使用率
                cpu_per_core = self.psutil.cpu_percent(interval=None, percpu=True)
                
                # 获取内存使用情况
                memory = self.psutil.virtual_memory()
                
                # 获取负载平均值（Unix系统）
                try:
                    load_avg = self.psutil.getloadavg()
                    load_info = f"负载: {load_avg[0]:.2f}, {load_avg[1]:.2f}, {load_avg[2]:.2f}"
                except (AttributeError, OSError):
                    load_info = "负载: N/A"
                
                if self.verbose:
                    print(f"\n=== CPU监控 ===")
                    print(f"整体CPU使用率: {cpu_percent:.1f}%")
                    print(f"内存使用率: {memory.percent:.1f}% ({memory.used / (1024**3):.1f}GB / {memory.total / (1024**3):.1f}GB)")
                    print(f"{load_info}")
                    
                    # 显示CPU核心使用率（如果核心数不太多）
                    if len(cpu_per_core) <= 16:
                        core_usage = ", ".join([f"C{i}:{usage:.0f}%" for i, usage in enumerate(cpu_per_core)])
                        print(f"各核心使用率: {core_usage}")
                    else:
                        # 核心太多时显示统计信息
                        active_cores = sum(1 for usage in cpu_per_core if usage > 10)
                        avg_usage = sum(cpu_per_core) / len(cpu_per_core)
                        print(f"核心统计: 活跃核心={active_cores}/{len(cpu_per_core)}, 平均使用率={avg_usage:.1f}%")
                    
                    # 检查是否存在性能瓶颈
                    if cpu_percent < 50:
                        print("⚠️  CPU使用率较低，可能存在以下问题:")
                        print("   - 批次大小过小")
                        print("   - 数据加载成为瓶颈")
                        print("   - TensorFlow线程配置不当")
                        print("   - I/O等待时间过长")
                    elif cpu_percent > 95:
                        print("⚠️  CPU使用率过高，可能导致系统不稳定")
                
                # 等待下一次检查
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                if self.verbose:
                    print(f"CPU监控出错: {e}")
                time.sleep(self.monitor_interval)

def create_cpu_monitor_callback(monitor_interval=30, verbose=True):
    """
    创建CPU监控回调
    
    Args:
        monitor_interval: 监控间隔（秒）
        verbose: 是否打印详细信息
    
    Returns:
        CPUMonitorCallback实例
    """
    return CPUMonitorCallback(monitor_interval=monitor_interval, verbose=verbose)