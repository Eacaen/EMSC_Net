#!/usr/bin/env python3
"""
EMSC网络实验管理脚本
支持不同网络结构的批量训练和结果管理
"""

import os
import subprocess
import argparse
from datetime import datetime

def create_experiment_config():
    """定义不同的网络配置实验"""
    experiments = [
        # 标准配置
        {
            "name": "standard",
            "state_dim": 8,
            "hidden_dim": 32,
            "structure": "6-32-32-8-1",
            "description": "标准EMSC配置"
        },
        # 更大的隐藏层
        {
            "name": "large_hidden",
            "state_dim": 8,
            "hidden_dim": 64,
            "structure": "6-64-64-8-1",
            "description": "增大隐藏层维度"
        },
        # 更大的状态维度
        {
            "name": "large_state",
            "state_dim": 16,
            "hidden_dim": 32,
            "structure": "6-32-32-16-1",
            "description": "增大状态维度"
        },
        # 小型网络
        {
            "name": "compact",
            "state_dim": 4,
            "hidden_dim": 16,
            "structure": "6-16-16-4-1",
            "description": "紧凑型网络"
        },
        # 超大网络
        {
            "name": "xlarge",
            "state_dim": 16,
            "hidden_dim": 64,
            "structure": "6-64-64-16-1",
            "description": "超大型网络"
        }
    ]
    return experiments

def run_single_experiment(config, args):
    """运行单个实验配置"""
    print(f"\n{'='*60}")
    print(f"🚀 开始实验: {config['name']}")
    print(f"📊 网络结构: {config['structure']}")
    print(f"📝 描述: {config['description']}")
    print(f"{'='*60}")
    
    # 构建训练命令
    cmd = [
        "python", "train.py",
        "--state_dim", str(config['state_dim']),
        "--hidden_dim", str(config['hidden_dim']),
        "--epochs", str(args.epochs),
        "--learning_rate", str(args.learning_rate),
        "--batch_size", str(args.batch_size) if args.batch_size else "auto"
    ]
    
    # 添加可选参数
    if args.resume:
        cmd.append("--resume")
    if args.dataset:
        cmd.extend(["--dataset", args.dataset])
    if args.save_frequency:
        cmd.extend(["--save_frequency", str(args.save_frequency)])
    if hasattr(args, 'device') and args.device:
        cmd.extend(["--device", args.device])
    
    print(f"🔧 执行命令: {' '.join(cmd)}")
    
    # 记录开始时间
    start_time = datetime.now()
    print(f"⏰ 开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 运行训练
        result = subprocess.run(cmd, capture_output=False, text=True, check=True)
        
        # 记录完成时间
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n✅ 实验 {config['name']} 完成!")
        print(f"⏰ 结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️  训练耗时: {duration}")
        print(f"📁 结果保存在: models/dataset_EMSC_big/network_{config['structure']}/")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 实验 {config['name']} 失败!")
        print(f"错误码: {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n🛑 实验 {config['name']} 被用户中断")
        return False

def list_experiments():
    """列出所有可用的实验配置"""
    experiments = create_experiment_config()
    
    print("📋 可用的实验配置:")
    print("-" * 80)
    print(f"{'名称':<15} {'网络结构':<15} {'描述':<30}")
    print("-" * 80)
    
    for exp in experiments:
        print(f"{exp['name']:<15} {exp['structure']:<15} {exp['description']:<30}")
    
    print("-" * 80)
    print(f"总共 {len(experiments)} 个配置")

def check_results():
    """检查已有的训练结果"""
    base_dir = "models/dataset_EMSC_big"
    
    if not os.path.exists(base_dir):
        print(f"❌ 结果目录不存在: {base_dir}")
        return
    
    print("📊 已有的训练结果:")
    print("-" * 60)
    
    network_dirs = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path) and item.startswith("network_"):
            network_dirs.append(item)
    
    if not network_dirs:
        print("🔍 未找到任何网络训练结果")
        return
    
    for network_dir in sorted(network_dirs):
        network_path = os.path.join(base_dir, network_dir)
        structure = network_dir.replace("network_", "")
        
        # 检查文件
        files = os.listdir(network_path)
        has_model = any(f.endswith('.h5') for f in files)
        has_history = any(f.endswith('.json') for f in files)
        has_plots = any(f.endswith('.png') for f in files)
        
        status = "✅" if has_model else "⏳"
        
        print(f"{status} {structure:<15} ", end="")
        if has_model:
            print("模型 ", end="")
        if has_history:
            print("历史 ", end="")
        if has_plots:
            print("图表 ", end="")
        print()
    
    print("-" * 60)

def main():
    parser = argparse.ArgumentParser(description="EMSC网络实验管理")
    parser.add_argument("--list", action="store_true", help="列出所有实验配置")
    parser.add_argument("--check", action="store_true", help="检查训练结果")
    parser.add_argument("--run", nargs="+", help="运行指定的实验配置(名称)")
    parser.add_argument("--run_all", action="store_true", help="运行所有实验配置")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=200, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="学习率")
    parser.add_argument("--batch_size", type=int, help="批次大小(默认自动)")
    parser.add_argument("--dataset", default="big", help="数据集类型")
    parser.add_argument("--save_frequency", type=int, default=10, help="保存频率")
    parser.add_argument("--resume", action="store_true", help="恢复训练")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "gpu", "cpu"],
                       help="指定设备类型: auto(自动), gpu(强制GPU), cpu(强制CPU)")
    
    args = parser.parse_args()
    
    if args.list:
        list_experiments()
        return
    
    if args.check:
        check_results()
        return
    
    experiments = create_experiment_config()
    experiment_dict = {exp['name']: exp for exp in experiments}
    
    if args.run:
        # 运行指定的实验
        for exp_name in args.run:
            if exp_name not in experiment_dict:
                print(f"❌ 未知的实验配置: {exp_name}")
                print("💡 使用 --list 查看可用配置")
                continue
            
            success = run_single_experiment(experiment_dict[exp_name], args)
            if not success and len(args.run) > 1:
                response = input("\n⚠️  实验失败，是否继续下一个实验? (y/n): ")
                if response.lower() != 'y':
                    break
    
    elif args.run_all:
        # 运行所有实验
        print(f"🚀 准备运行 {len(experiments)} 个实验配置")
        
        success_count = 0
        for i, exp in enumerate(experiments, 1):
            print(f"\n📊 进度: {i}/{len(experiments)}")
            success = run_single_experiment(exp, args)
            if success:
                success_count += 1
            elif i < len(experiments):
                response = input("\n⚠️  实验失败，是否继续下一个实验? (y/n): ")
                if response.lower() != 'y':
                    break
        
        print(f"\n📈 实验总结:")
        print(f"✅ 成功: {success_count}/{len(experiments)}")
        print(f"❌ 失败: {len(experiments) - success_count}/{len(experiments)}")
    
    else:
        parser.print_help()
        print("\n💡 使用示例:")
        print("  python run_experiments.py --list                    # 列出所有配置")
        print("  python run_experiments.py --check                   # 检查结果")
        print("  python run_experiments.py --run standard            # 运行标准配置")
        print("  python run_experiments.py --run standard large      # 运行多个配置")
        print("  python run_experiments.py --run_all                 # 运行所有配置")

if __name__ == '__main__':
    main() 