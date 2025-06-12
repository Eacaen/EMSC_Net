#!/usr/bin/env python3
"""
EMSC网络结构对比分析脚本
比较不同网络结构的训练效果和性能
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_training_history(network_dir):
    """加载训练历史"""
    history_files = [
        'training_history.json',
        'training_history_final.json'
    ]
    
    for history_file in history_files:
        history_path = os.path.join(network_dir, history_file)
        if os.path.exists(history_path):
            try:
                with open(history_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️  加载历史文件失败 {history_path}: {e}")
                continue
    
    return None

def get_network_results():
    """获取所有网络的训练结果"""
    base_dir = "models/dataset_EMSC_big"
    
    if not os.path.exists(base_dir):
        print(f"❌ 结果目录不存在: {base_dir}")
        return {}
    
    results = {}
    
    for item in os.listdir(base_dir):
        if not item.startswith("network_"):
            continue
            
        network_path = os.path.join(base_dir, item)
        if not os.path.isdir(network_path):
            continue
        
        structure = item.replace("network_", "")
        print(f"📊 加载网络 {structure} 的结果...")
        
        history = load_training_history(network_path)
        if history is None:
            print(f"⚠️  跳过 {structure}: 无法加载训练历史")
            continue
        
        # 提取关键指标
        try:
            final_train_loss = history['loss'][-1] if 'loss' in history else None
            final_val_loss = history['val_loss'][-1] if 'val_loss' in history else None
            min_val_loss = min(history['val_loss']) if 'val_loss' in history else None
            total_epochs = len(history['loss']) if 'loss' in history else 0
            
            results[structure] = {
                'history': history,
                'final_train_loss': final_train_loss,
                'final_val_loss': final_val_loss,
                'min_val_loss': min_val_loss,
                'total_epochs': total_epochs,
                'path': network_path
            }
            
            print(f"  ✅ 加载成功: {total_epochs} epochs, 最小验证损失: {min_val_loss:.6f}")
            
        except Exception as e:
            print(f"  ❌ 处理失败: {e}")
    
    return results

def plot_loss_comparison(results):
    """绘制损失对比图"""
    plt.figure(figsize=(15, 10))
    
    # 训练损失对比
    plt.subplot(2, 2, 1)
    for structure, data in results.items():
        history = data['history']
        if 'loss' in history:
            epochs = range(1, len(history['loss']) + 1)
            plt.plot(epochs, history['loss'], label=f'{structure} (训练)', alpha=0.7)
    
    plt.title('训练损失对比', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # 验证损失对比
    plt.subplot(2, 2, 2)
    for structure, data in results.items():
        history = data['history']
        if 'val_loss' in history:
            epochs = range(1, len(history['val_loss']) + 1)
            plt.plot(epochs, history['val_loss'], label=f'{structure} (验证)', alpha=0.7)
    
    plt.title('验证损失对比', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # 最终损失对比
    plt.subplot(2, 2, 3)
    structures = list(results.keys())
    final_train_losses = [results[s]['final_train_loss'] for s in structures]
    final_val_losses = [results[s]['final_val_loss'] for s in structures]
    
    x = np.arange(len(structures))
    width = 0.35
    
    plt.bar(x - width/2, final_train_losses, width, label='训练损失', alpha=0.7)
    plt.bar(x + width/2, final_val_losses, width, label='验证损失', alpha=0.7)
    
    plt.title('最终损失对比', fontsize=14, fontweight='bold')
    plt.xlabel('网络结构')
    plt.ylabel('Loss')
    plt.xticks(x, structures, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # 最小验证损失对比
    plt.subplot(2, 2, 4)
    min_val_losses = [results[s]['min_val_loss'] for s in structures]
    
    bars = plt.bar(structures, min_val_losses, alpha=0.7, color='green')
    plt.title('最小验证损失对比', fontsize=14, fontweight='bold')
    plt.xlabel('网络结构')
    plt.ylabel('Minimum Validation Loss')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # 添加数值标签
    for bar, loss in zip(bars, min_val_losses):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                f'{loss:.4f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # 保存图表
    output_path = "models/network_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 对比图表已保存: {output_path}")
    
    plt.show()

def create_performance_summary(results):
    """创建性能总结表"""
    print("\n📈 网络性能总结")
    print("=" * 100)
    print(f"{'网络结构':<15} {'总参数':<10} {'训练轮数':<10} {'最终训练损失':<15} {'最终验证损失':<15} {'最小验证损失':<15}")
    print("=" * 100)
    
    # 按最小验证损失排序
    sorted_results = sorted(results.items(), key=lambda x: x[1]['min_val_loss'])
    
    for structure, data in sorted_results:
        # 估算参数数量（简化计算）
        parts = structure.split('-')
        if len(parts) == 5:
            input_dim, hidden1, hidden2, state_dim, output_dim = map(int, parts)
            # 简化的参数估算
            params = (input_dim + state_dim + 1 + 3) * hidden1 * 2  # 内部层1
            params += hidden1 * hidden1 * 2  # 内部层2  
            params += hidden1 * 3  # 门控参数
            params += hidden1 * state_dim  # 候选状态
            params += state_dim * output_dim  # 输出层
        else:
            params = "未知"
        
        print(f"{structure:<15} {params:<10} {data['total_epochs']:<10} "
              f"{data['final_train_loss']:<15.6f} {data['final_val_loss']:<15.6f} "
              f"{data['min_val_loss']:<15.6f}")
    
    print("=" * 100)
    
    # 找出最佳网络
    best_network = min(results.items(), key=lambda x: x[1]['min_val_loss'])
    print(f"\n🏆 最佳网络: {best_network[0]}")
    print(f"   最小验证损失: {best_network[1]['min_val_loss']:.6f}")
    print(f"   训练轮数: {best_network[1]['total_epochs']}")

def create_detailed_report(results):
    """创建详细报告"""
    report_path = "models/network_comparison_report.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("EMSC网络结构对比报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"报告生成时间: {os.popen('date').read()}\n")
        f.write(f"对比网络数量: {len(results)}\n\n")
        
        # 按性能排序
        sorted_results = sorted(results.items(), key=lambda x: x[1]['min_val_loss'])
        
        f.write("网络性能排名:\n")
        f.write("-" * 30 + "\n")
        for i, (structure, data) in enumerate(sorted_results, 1):
            f.write(f"{i}. {structure}\n")
            f.write(f"   最小验证损失: {data['min_val_loss']:.6f}\n")
            f.write(f"   最终验证损失: {data['final_val_loss']:.6f}\n")
            f.write(f"   训练轮数: {data['total_epochs']}\n\n")
        
        f.write("详细分析:\n")
        f.write("-" * 30 + "\n")
        
        # 分析不同维度的影响
        hidden_dim_analysis = {}
        state_dim_analysis = {}
        
        for structure, data in results.items():
            parts = structure.split('-')
            if len(parts) == 5:
                hidden_dim = int(parts[1])
                state_dim = int(parts[3])
                
                if hidden_dim not in hidden_dim_analysis:
                    hidden_dim_analysis[hidden_dim] = []
                hidden_dim_analysis[hidden_dim].append(data['min_val_loss'])
                
                if state_dim not in state_dim_analysis:
                    state_dim_analysis[state_dim] = []
                state_dim_analysis[state_dim].append(data['min_val_loss'])
        
        f.write("隐藏层维度影响:\n")
        for dim, losses in sorted(hidden_dim_analysis.items()):
            avg_loss = np.mean(losses)
            f.write(f"  {dim}维: 平均验证损失 {avg_loss:.6f}\n")
        
        f.write("\n状态维度影响:\n")
        for dim, losses in sorted(state_dim_analysis.items()):
            avg_loss = np.mean(losses)
            f.write(f"  {dim}维: 平均验证损失 {avg_loss:.6f}\n")
    
    print(f"📄 详细报告已保存: {report_path}")

def main():
    print("🔍 EMSC网络结构对比分析")
    print("=" * 50)
    
    # 加载所有网络结果
    results = get_network_results()
    
    if not results:
        print("❌ 未找到任何训练结果")
        print("💡 请先运行一些实验:")
        print("   python run_experiments.py --run standard")
        return
    
    print(f"\n📊 找到 {len(results)} 个网络的训练结果")
    
    # 创建对比图表
    plot_loss_comparison(results)
    
    # 创建性能总结
    create_performance_summary(results)
    
    # 创建详细报告
    create_detailed_report(results)
    
    print(f"\n✅ 对比分析完成!")
    print(f"📁 查看结果:")
    print(f"   - 对比图表: models/network_comparison.png")
    print(f"   - 详细报告: models/network_comparison_report.txt")

if __name__ == '__main__':
    main() 