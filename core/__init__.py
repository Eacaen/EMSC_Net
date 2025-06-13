"""
EMSC 核心模块
包含模型定义、数据处理和损失函数
"""

import sys
import os

# 添加导入路径兼容性
def _setup_imports():
    """设置导入路径以支持相对导入和绝对导入"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

_setup_imports()

# 导入核心组件
from .EMSC_model import build_msc_model, MSC_Cell, MSC_Sequence
from .EMSC_data import EMSCDataGenerator, create_tf_dataset, load_dataset_from_npz
from .EMSC_losses import EMSCLoss

__all__ = [
    'build_msc_model',
    'MSC_Cell', 
    'MSC_Sequence',
    'EMSCDataGenerator',
    'create_tf_dataset',
    'load_dataset_from_npz',
    'EMSCLoss'
]
