"""
EMSC模型训练入口脚本
提供简化的命令行接口来启动训练
"""

import os
import sys
import tensorflow as tf
from EMSC_config import parse_training_args, get_dataset_paths

def check_environment():
    """检查训练环境"""
    print("检查训练环境...")
    print(f"TensorFlow版本: {tf.__version__}")
    print(f"当前工作目录: {os.getcwd()}")
    print(f"Python版本: {sys.version}")
    
    # 检查GPU可用性
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"发现 {len(gpus)} 个GPU设备:")
        for gpu in gpus:
            print(f"- {gpu}")
    else:
        print("未发现GPU设备，将使用CPU训练")
    
    # 检查内存使用情况
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"\n系统内存信息:")
        print(f"总内存: {memory.total / (1024**3):.1f} GB")
        print(f"可用内存: {memory.available / (1024**3):.1f} GB")
        print(f"内存使用率: {memory.percent}%")
    except ImportError:
        print("未安装psutil，跳过内存检查")

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_training_args()
    
    # 检查环境
    check_environment()
    
    # 获取数据集路径
    paths = get_dataset_paths(args.dataset)
    dataset_path = paths['dataset_path']
    
    # 检查数据集是否存在
    if not os.path.exists(dataset_path):
        print(f"错误: 数据集不存在: {dataset_path}")
        sys.exit(1)
    
    # 导入并运行主训练脚本
    from EMSC_train import main as train_main
    train_main()

if __name__ == '__main__':
    main()