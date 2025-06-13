"""
EMSC (Enhanced Minimal State Cell) 训练系统
一个用于时间序列预测的深度学习框架

主要模块:
- core: 核心组件（模型、数据、损失函数）
- training: 训练相关功能
- prediction: 预测相关功能  
- cloud: 云服务集成
- utils: 工具和辅助功能
- tests: 测试和调试工具
- scripts: 独立脚本和实验工具
"""

__version__ = "1.0.0"
__author__ = "EMSC Team"

# 核心组件导入
try:
    from .core.EMSC_model import build_msc_model
    from .core.EMSC_data import EMSCDataGenerator, create_tf_dataset, load_dataset_from_npz
    from .core.EMSC_losses import EMSCLoss
    
    # 训练组件导入
    from .training.EMSC_train import main as train_main
    from .training.EMSC_callbacks import MSCProgressCallback, create_early_stopping_callback
    from .training.EMSC_dynamic_batch import DynamicBatchTrainer
    
    # 预测组件导入
    from .prediction.EMSC_predict import predict_sequence, load_model_and_predict
    from .prediction.EMSC_predict_auto import auto_predict
    
    # 配置和工具导入
    from .utils.EMSC_config import create_training_config, parse_training_args
    from .utils.EMSC_utils import load_or_create_model_with_history
    
    # 云服务导入（可选）
    try:
        from .cloud.EMSC_oss_uploader import EMSCOSSUploader
        from .cloud.EMSC_oss_downloader import download_dataset
        OSS_AVAILABLE = True
    except ImportError:
        OSS_AVAILABLE = False
        
except ImportError:
    # 如果相对导入失败，尝试绝对导入（用于直接运行脚本的情况）
    import sys
    import os
    
    # 添加当前目录到Python路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    try:
        from core.EMSC_model import build_msc_model
        from core.EMSC_data import EMSCDataGenerator, create_tf_dataset, load_dataset_from_npz
        from core.EMSC_losses import EMSCLoss
        
        # 训练组件导入
        from training.EMSC_train import main as train_main
        from training.EMSC_callbacks import MSCProgressCallback, create_early_stopping_callback
        from training.EMSC_dynamic_batch import DynamicBatchTrainer
        
        # 预测组件导入
        from prediction.EMSC_predict import predict_sequence, load_model_and_predict
        from prediction.EMSC_predict_auto import auto_predict
        
        # 配置和工具导入
        from utils.EMSC_config import create_training_config, parse_training_args
        from utils.EMSC_utils import load_or_create_model_with_history
        
        # 云服务导入（可选）
        try:
            from cloud.EMSC_oss_uploader import EMSCOSSUploader
            from cloud.EMSC_oss_downloader import download_dataset
            OSS_AVAILABLE = True
        except ImportError:
            OSS_AVAILABLE = False
            
    except ImportError as e:
        print(f"警告: 无法导入EMSC模块: {e}")
        OSS_AVAILABLE = False

__all__ = [
    # 核心组件
    'build_msc_model',
    'EMSCDataGenerator', 
    'create_tf_dataset',
    'load_dataset_from_npz',
    'EMSCLoss',
    
    # 训练组件
    'train_main',
    'MSCProgressCallback',
    'create_early_stopping_callback', 
    'DynamicBatchTrainer',
    
    # 预测组件
    'predict_sequence',
    'load_model_and_predict',
    'auto_predict',
    
    # 配置和工具
    'create_training_config',
    'parse_training_args',
    'load_or_create_model_with_history',
    
    # 版本信息
    '__version__',
    'OSS_AVAILABLE'
]

# 云服务组件（如果可用）
if OSS_AVAILABLE:
    __all__.extend(['EMSCOSSUploader', 'download_dataset'])