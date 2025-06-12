"""
EMSC模型训练入口脚本
提供简化的命令行接口来启动训练
"""

import os
import sys
from EMSC_config import parse_training_args, get_dataset_paths
from EMSC_train import check_environment

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_training_args()
    
    # 获取数据集路径
    paths = get_dataset_paths(args.dataset)
    dataset_path = paths['dataset_path']
    
    # 检查数据集是否存在
    if not os.path.exists(dataset_path):
        print(f"错误: 数据集不存在: {dataset_path}")
        sys.exit(1)
    
    # 导入并运行主训练脚本，传递args参数
    from EMSC_train import main as train_main
    train_main(args)

if __name__ == '__main__':
    main()