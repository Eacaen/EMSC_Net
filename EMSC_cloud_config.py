"""
EMSC云计算环境配置模块
支持在阿里云DLC等云平台上自适应调整训练参数
"""

import os
import json
import psutil
import tensorflow as tf
from typing import Dict, Optional, Tuple, Union
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CloudEnvironmentConfig:
    """云计算环境配置类"""
    
    # GPU型号对应的推荐配置
    GPU_CONFIGS = {
        'NVIDIA A100': {
            'batch_size': 256,
            'mixed_precision': True,
            'num_parallel_calls': tf.data.AUTOTUNE,
            'prefetch_buffer_size': tf.data.AUTOTUNE,
            'cache_size': 'auto'  # 自动根据显存大小调整
        },
        'NVIDIA V100': {
            'batch_size': 128,
            'mixed_precision': True,
            'num_parallel_calls': tf.data.AUTOTUNE,
            'prefetch_buffer_size': tf.data.AUTOTUNE,
            'cache_size': 'auto'
        },
        'NVIDIA T4': {
            'batch_size': 64,
            'mixed_precision': True,
            'num_parallel_calls': tf.data.AUTOTUNE,
            'prefetch_buffer_size': tf.data.AUTOTUNE,
            'cache_size': 'auto'
        },
        'default': {  # 其他GPU型号的默认配置
            'batch_size': 32,
            'mixed_precision': True,
            'num_parallel_calls': tf.data.AUTOTUNE,
            'prefetch_buffer_size': tf.data.AUTOTUNE,
            'cache_size': 'auto'
        }
    }
    
    # CPU配置
    CPU_CONFIGS = {
        'high_memory': {  # 高内存配置
            'batch_size': 32,
            'mixed_precision': False,
            'num_parallel_calls': 8,
            'prefetch_buffer_size': 2,
            'cache_size': 'auto'
        },
        'medium_memory': {  # 中等内存配置
            'batch_size': 16,
            'mixed_precision': False,
            'num_parallel_calls': 4,
            'prefetch_buffer_size': 2,
            'cache_size': 'auto'
        },
        'low_memory': {  # 低内存配置
            'batch_size': 8,
            'mixed_precision': False,
            'num_parallel_calls': 2,
            'prefetch_buffer_size': 1,
            'cache_size': 'auto'
        }
    }
    
    def __init__(self):
        """初始化云环境配置"""
        self.is_cloud = self._detect_cloud_environment()
        self.gpu_info = self._get_gpu_info()
        self.cpu_info = self._get_cpu_info()
        self.memory_info = self._get_memory_info()
        
        # 根据环境自动选择配置
        self.config = self._select_optimal_config()
        
        # 应用配置
        self._apply_config()
        
        # 记录配置信息
        self._log_environment_info()
    
    def _detect_cloud_environment(self) -> bool:
        """检测是否在云环境中运行"""
        # 检查常见的云环境标识
        cloud_indicators = [
            'ALIYUN_DLC',  # 阿里云DLC
            'ALIYUN_ECS',  # 阿里云ECS
            'AWS_EC2',     # AWS EC2
            'GCP_GCE',     # Google Cloud
            'AZURE_VM'     # Azure VM
        ]
        
        # 检查环境变量
        for indicator in cloud_indicators:
            if os.environ.get(indicator):
                return True
        
        # 检查主机名（通常云实例有特定的命名模式）
        hostname = os.uname().nodename
        cloud_hostname_patterns = [
            'aliyun', 'alibaba', 'aws', 'amazon', 'gcp', 'google', 'azure'
        ]
        if any(pattern in hostname.lower() for pattern in cloud_hostname_patterns):
            return True
        
        return False
    
    def _get_gpu_info(self) -> Dict:
        """获取GPU信息"""
        gpu_info = {
            'available': False,
            'count': 0,
            'devices': [],
            'memory_info': {}
        }
        
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                gpu_info['available'] = True
                gpu_info['count'] = len(gpus)
                gpu_info['devices'] = [gpu.name for gpu in gpus]
                
                # 获取每个GPU的详细信息
                for gpu in gpus:
                    try:
                        # 使用nvidia-smi获取详细信息
                        import subprocess
                        result = subprocess.check_output(
                            ['nvidia-smi', '--query-gpu=name,memory.total,memory.free',
                             '--format=csv,noheader,nounits']
                        ).decode()
                        
                        for line in result.strip().split('\n'):
                            name, total_mem, free_mem = line.split(', ')
                            gpu_info['memory_info'][gpu.name] = {
                                'name': name.strip(),
                                'total_memory': int(total_mem),
                                'free_memory': int(free_mem)
                            }
                    except Exception as e:
                        logger.warning(f"无法获取GPU {gpu.name} 的详细信息: {e}")
        
        except Exception as e:
            logger.warning(f"获取GPU信息时出错: {e}")
        
        return gpu_info
    
    def _get_cpu_info(self) -> Dict:
        """获取CPU信息"""
        return {
            'count': os.cpu_count() or 1,
            'physical_cores': psutil.cpu_count(logical=False) or 1,
            'logical_cores': psutil.cpu_count(logical=True) or 1
        }
    
    def _get_memory_info(self) -> Dict:
        """获取内存信息"""
        memory = psutil.virtual_memory()
        return {
            'total': memory.total,
            'available': memory.available,
            'used': memory.used,
            'percent': memory.percent
        }
    
    def _select_optimal_config(self) -> Dict:
        """根据环境选择最优配置"""
        if self.gpu_info['available']:
            # GPU模式
            if self.gpu_info['devices']:
                # 获取第一个GPU的型号
                gpu_name = self.gpu_info['memory_info'].get(
                    self.gpu_info['devices'][0], {}
                ).get('name', 'default')
                
                # 查找匹配的GPU配置
                for key in self.GPU_CONFIGS:
                    if key in gpu_name:
                        config = self.GPU_CONFIGS[key].copy()
                        break
                else:
                    config = self.GPU_CONFIGS['default'].copy()
                
                # 根据显存大小调整batch_size
                if config['cache_size'] == 'auto':
                    gpu_memory = self.gpu_info['memory_info'].get(
                        self.gpu_info['devices'][0], {}
                    ).get('total_memory', 0)
                    
                    # 根据显存大小调整batch_size
                    if gpu_memory >= 40000:  # 40GB
                        config['batch_size'] *= 2
                    elif gpu_memory <= 8000:  # 8GB
                        config['batch_size'] //= 2
                
                return config
        
        # CPU模式
        memory_gb = self.memory_info['total'] / (1024**3)
        if memory_gb >= 32:  # 32GB以上
            config = self.CPU_CONFIGS['high_memory'].copy()
        elif memory_gb >= 16:  # 16GB以上
            config = self.CPU_CONFIGS['medium_memory'].copy()
        else:
            config = self.CPU_CONFIGS['low_memory'].copy()
        
        # 根据CPU核心数调整并行度
        config['num_parallel_calls'] = min(
            self.cpu_info['physical_cores'],
            config['num_parallel_calls']
        )
        
        return config
    
    def _apply_config(self):
        """应用配置到TensorFlow环境"""
        # 设置混合精度
        if self.config['mixed_precision']:
            try:
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                logger.info("已启用混合精度训练")
            except Exception as e:
                logger.warning(f"启用混合精度训练失败: {e}")
        
        # 配置GPU内存增长
        if self.gpu_info['available']:
            try:
                for gpu in tf.config.list_physical_devices('GPU'):
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info("已配置GPU内存动态增长")
            except Exception as e:
                logger.warning(f"配置GPU内存增长失败: {e}")
    
    def _log_environment_info(self):
        """记录环境信息"""
        logger.info("=== 云环境配置信息 ===")
        logger.info(f"运行环境: {'云环境' if self.is_cloud else '本地环境'}")
        
        if self.gpu_info['available']:
            logger.info("GPU信息:")
            logger.info(f"- 可用GPU数量: {self.gpu_info['count']}")
            for device in self.gpu_info['devices']:
                mem_info = self.gpu_info['memory_info'].get(device, {})
                logger.info(f"- {device}:")
                logger.info(f"  型号: {mem_info.get('name', 'Unknown')}")
                logger.info(f"  总显存: {mem_info.get('total_memory', 0)}MB")
                logger.info(f"  可用显存: {mem_info.get('free_memory', 0)}MB")
        else:
            logger.info("CPU信息:")
            logger.info(f"- 物理核心数: {self.cpu_info['physical_cores']}")
            logger.info(f"- 逻辑核心数: {self.cpu_info['logical_cores']}")
        
        logger.info("内存信息:")
        logger.info(f"- 总内存: {self.memory_info['total'] / (1024**3):.1f}GB")
        logger.info(f"- 可用内存: {self.memory_info['available'] / (1024**3):.1f}GB")
        logger.info(f"- 内存使用率: {self.memory_info['percent']}%")
        
        logger.info("训练配置:")
        for key, value in self.config.items():
            logger.info(f"- {key}: {value}")
    
    def get_config(self) -> Dict:
        """获取当前配置"""
        return self.config.copy()
    
    def get_batch_size(self) -> int:
        """获取推荐的batch size"""
        return self.config['batch_size']
    
    def get_data_config(self) -> Dict:
        """获取数据加载相关配置"""
        return {
            'num_parallel_calls': self.config['num_parallel_calls'],
            'prefetch_buffer_size': self.config['prefetch_buffer_size'],
            'cache_size': self.config['cache_size']
        }
    
    def is_mixed_precision_enabled(self) -> bool:
        """检查是否启用了混合精度"""
        return self.config['mixed_precision']


# 创建全局配置实例
cloud_config = CloudEnvironmentConfig()

def get_cloud_config() -> CloudEnvironmentConfig:
    """获取云环境配置实例"""
    return cloud_config 