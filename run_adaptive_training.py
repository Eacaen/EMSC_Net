#!/usr/bin/env python3
"""
EMSC自适应训练启动脚本
专门用于解决损失停滞问题的训练启动器
"""

import os
import sys
import subprocess

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def main():
    """主函数"""
    print("🚀 EMSC自适应训练启动器")
    print("专门解决损失停滞在0.006左右的问题")
    print("=" * 60)
    
    # 用户配置
    config = {
        'dataset': 'dataset_EMSC_big',
        'state_dim': 8,
        'hidden_dim': 8,
        'learning_rate': 1e-2,  # 较大的学习率帮助跳出停滞
        'epochs': 2000,
        'save_frequency': 10,
        'device': 'auto'
    }
    
    print(f"📊 训练配置:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    print()
    
    # 提供多种解决方案选择
    print("🎯 可选的解决方案:")
    print("1. 自适应训练 (推荐) - 循环学习率 + 热重启 + 噪声注入")
    print("2. 激进学习率 - 使用更大的学习率和激进调度")
    print("3. 网络结构调整 - 增大网络容量")
    print("4. 诊断模式 - 分析具体问题原因")
    print("5. 标准训练 - 使用改进的回调")
    print()
    
    try:
        choice = input("请选择解决方案 (1-5) [默认: 1]: ").strip()
        if not choice:
            choice = "1"
            
        print(f"\n🎯 您选择了方案 {choice}")
        print("🚀 准备启动训练...")
        
        if choice == "1":
            # 自适应训练
            run_adaptive_training(config)
        elif choice == "2":
            # 激进学习率训练
            run_aggressive_lr_training(config)
        elif choice == "3":
            # 增大网络结构
            run_larger_network_training(config)
        elif choice == "4":
            # 诊断模式
            run_diagnosis()
        elif choice == "5":
            # 标准训练
            run_standard_training(config)
        else:
            print("❌ 无效的选择，使用默认的自适应训练")
            run_adaptive_training(config)
            
    except KeyboardInterrupt:
        print("\n❌ 用户取消操作")
    except Exception as e:
        print(f"❌ 启动失败: {e}")


def run_adaptive_training(config):
    """运行自适应训练"""
    print("\n🚀 启动自适应训练模式 (解决损失停滞的最佳方案)")
    
    cmd = [
        "python", "-m", "training.EMSC_train",
        "--dataset", config['dataset'],
        "--adaptive_training",
        "--learning_rate", str(config['learning_rate']),
        "--state_dim", str(config['state_dim']),
        "--hidden_dim", str(config['hidden_dim']),
        "--epochs", str(config['epochs']),
        "--save_frequency", str(config['save_frequency']),
        "--device", config['device'],
        "--resume"
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    subprocess.run(cmd)


def run_aggressive_lr_training(config):
    """运行激进学习率训练"""
    print("\n🚀 启动激进学习率训练")
    
    aggressive_lr = 5e-2  # 更大的学习率
    
    cmd = [
        "python", "-m", "training.EMSC_train",
        "--dataset", config['dataset'],
        "--cyclical_lr",
        "--warm_restart",
        "--learning_rate", str(aggressive_lr),
        "--state_dim", str(config['state_dim']),
        "--hidden_dim", str(config['hidden_dim']),
        "--epochs", str(config['epochs']),
        "--save_frequency", str(config['save_frequency']),
        "--device", config['device'],
        "--resume"
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    subprocess.run(cmd)


def run_larger_network_training(config):
    """运行更大网络结构训练"""
    print("\n🚀 启动大网络结构训练")
    
    larger_config = config.copy()
    larger_config['hidden_dim'] = 64  # 增大隐藏层
    larger_config['state_dim'] = 16   # 增大状态维度
    
    cmd = [
        "python", "-m", "training.EMSC_train",
        "--dataset", config['dataset'],
        "--adaptive_training",
        "--learning_rate", str(config['learning_rate']),
        "--state_dim", str(larger_config['state_dim']),
        "--hidden_dim", str(larger_config['hidden_dim']),
        "--epochs", str(config['epochs']),
        "--save_frequency", str(config['save_frequency']),
        "--device", config['device'],
        "--resume"
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    subprocess.run(cmd)


def run_diagnosis():
    """运行诊断"""
    print("\n🔧 启动诊断模式")
    
    cmd = ["python", "run_diagnosis.py"]
    
    print(f"执行命令: {' '.join(cmd)}")
    subprocess.run(cmd)


def run_standard_training(config):
    """运行标准训练（带改进）"""
    print("\n📊 启动标准训练模式 (使用改进的回调)")
    
    cmd = [
        "python", "-m", "training.EMSC_train",
        "--dataset", config['dataset'],
        "--learning_rate", str(config['learning_rate']),
        "--state_dim", str(config['state_dim']),
        "--hidden_dim", str(config['hidden_dim']),
        "--epochs", str(config['epochs']),
        "--save_frequency", str(config['save_frequency']),
        "--device", config['device'],
        "--resume"
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    subprocess.run(cmd)


def show_quick_tips():
    """显示快速提示"""
    print("\n💡 快速解决损失停滞的提示:")
    print("1. 如果仍然停滞在0.006，尝试:")
    print("   - 检查数据标准化范围")
    print("   - 增大学习率到1e-2或更大")
    print("   - 使用自适应训练模式")
    print()
    print("2. 检查训练日志中的:")
    print("   - 正则化损失比例")
    print("   - 梯度范数大小")
    print("   - 学习率变化")
    print()
    print("3. 如果问题持续，考虑:")
    print("   - 数据预处理方式")
    print("   - 损失函数权重")
    print("   - 网络结构设计")


if __name__ == "__main__":
    main()
    show_quick_tips() 