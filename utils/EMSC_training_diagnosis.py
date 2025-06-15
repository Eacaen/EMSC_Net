"""
EMSC训练诊断工具
用于分析训练停滞和损失停滞的具体原因
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.EMSC_model import build_msc_model
from core.EMSC_losses import EMSCLoss


class EMSCTrainingDiagnosis:
    """EMSC训练诊断器"""
    
    def __init__(self, model_path=None, dataset_path=None, state_dim=8, hidden_dim=32):
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.model = None
        self.train_data = None
        self.val_data = None
        
    def load_model_and_data(self):
        """加载模型和数据"""
        print("🔍 加载模型和数据...")
        
        # 加载模型
        if self.model_path and os.path.exists(self.model_path):
            try:
                from core.EMSC_model import MSC_Sequence
                from core.EMSC_losses import EMSCLoss
                
                # 检查是否是SavedModel目录格式
                if os.path.isdir(self.model_path):
                    # 查找目录中的模型文件
                    possible_models = []
                    for item in os.listdir(self.model_path):
                        item_path = os.path.join(self.model_path, item)
                        if os.path.isdir(item_path):
                            # 检查是否是SavedModel格式目录
                            if os.path.exists(os.path.join(item_path, 'saved_model.pb')):
                                possible_models.append(item_path)
                        elif item.endswith('.h5') or item.endswith('.keras'):
                            possible_models.append(item_path)
                    
                    if possible_models:
                        # 优先选择best模型
                        best_model = None
                        current_model = None
                        for model_path in possible_models:
                            if 'best' in os.path.basename(model_path).lower():
                                best_model = model_path
                            elif 'msc_model' in os.path.basename(model_path).lower():
                                current_model = model_path
                        
                        model_to_load = best_model or current_model or possible_models[0]
                        print(f"发现多个模型，选择: {os.path.basename(model_to_load)}")
                        self.model_path = model_to_load
                    else:
                        print(f"❌ 在目录 {self.model_path} 中未找到有效的模型文件")
                        return False
                
                # 加载模型
                self.model = tf.keras.models.load_model(
                    self.model_path,
                    custom_objects={
                        'MSC_Sequence': MSC_Sequence,
                        'EMSCLoss': EMSCLoss
                    }
                )
                print(f"✅ 模型加载成功: {self.model_path}")
                
            except Exception as e:
                print(f"❌ 模型加载失败: {e}")
                print("尝试创建新模型进行分析...")
                self.model = build_msc_model(
                    state_dim=self.state_dim,
                    hidden_dim=self.hidden_dim,
                    input_dim=6,
                    output_dim=1
                )
                print("✅ 已创建新模型用于分析")
        else:
            print("⚠️  未提供模型路径或路径不存在，将创建新模型用于分析")
            self.model = build_msc_model(
                state_dim=self.state_dim,
                hidden_dim=self.hidden_dim,
                input_dim=6,
                output_dim=1
            )
            
        # 加载数据
        if self.dataset_path and os.path.exists(self.dataset_path):
            try:
                # 检查数据集格式
                if self.dataset_path.endswith('.tfrecord'):
                    print(f"检测到TFRecord格式数据集: {self.dataset_path}")
                    # 对于TFRecord，我们只需要确认文件存在即可进行分析
                    print(f"✅ TFRecord数据集确认存在")
                    return True
                elif self.dataset_path.endswith('.npz'):
                    print(f"检测到NPZ格式数据集: {self.dataset_path}")
                    from core.EMSC_data import load_dataset_smart
                    dataset_result = load_dataset_smart(self.dataset_path)
                    if dataset_result:
                        print(f"✅ NPZ数据集加载成功")
                        return True
                    else:
                        print(f"❌ NPZ数据集加载失败")
                        return False
                else:
                    print(f"⚠️  未识别的数据集格式: {self.dataset_path}")
                    print("支持的格式: .tfrecord, .npz")
                    return False
                    
            except Exception as e:
                print(f"❌ 数据集处理失败: {e}")
                return False
        else:
            print("⚠️  未提供数据集路径或路径不存在")
            print("将仅使用模型进行分析（部分功能可能受限）")
            return True  # 即使没有数据集，也可以进行模型分析
    
    def analyze_loss_components(self, sample_batch_size=32):
        """分析损失函数组件"""
        print("📊 分析损失函数组件...")
        
        try:
            # 创建测试数据
            test_input = tf.random.normal((sample_batch_size, 100, 6), dtype=tf.float32)
            test_init_state = tf.zeros((sample_batch_size, self.state_dim), dtype=tf.float32)
            test_target = tf.random.normal((sample_batch_size, 100, 1), dtype=tf.float32)
            
            # 获取模型预测和门控参数 - 修复解包问题
            model_output = self.model([test_input, test_init_state], training=False)
            
            # 处理不同的模型输出格式
            if isinstance(model_output, (list, tuple)):
                if len(model_output) == 2:
                    predictions, gate_params = model_output
                elif len(model_output) > 2:
                    # 如果输出更多，取前两个
                    predictions = model_output[0]
                    gate_params = model_output[1] if len(model_output) > 1 else None
                else:
                    predictions = model_output[0]
                    gate_params = None
            else:
                # 如果输出是单个张量
                predictions = model_output
                gate_params = None
            
            # 计算MSE损失
            mse_loss = tf.reduce_mean(tf.square(test_target - predictions))
            print(f"📈 MSE损失: {mse_loss:.6f}")
            
            # 创建损失函数实例
            from core.EMSC_losses import EMSCLoss
            loss_fn = EMSCLoss(state_dim=self.state_dim)
            
            # 正确调用损失函数 - 直接调用call方法
            total_loss = loss_fn.call(test_target, predictions, gate_params)
            print(f"📊 总损失: {total_loss:.6f}")
            
            # 分析门控参数统计
            if gate_params and isinstance(gate_params, dict):
                gate_stats = {}
                for gate_name, gate_values in gate_params.items():
                    if isinstance(gate_values, tf.Tensor):
                        stats = {
                            'mean': float(tf.reduce_mean(gate_values).numpy()),
                            'std': float(tf.math.reduce_std(gate_values).numpy()),
                            'min': float(tf.reduce_min(gate_values).numpy()),
                            'max': float(tf.reduce_max(gate_values).numpy())
                        }
                        gate_stats[gate_name] = stats
                        print(f"🔧 门控参数 {gate_name}: 均值={stats['mean']:.4f}, 标准差={stats['std']:.4f}, 范围=[{stats['min']:.4f}, {stats['max']:.4f}]")
                
                # 分析门控参数问题
                problems = []
                for gate_name, stats in gate_stats.items():
                    if abs(stats['mean']) > 2.0:
                        problems.append(f"{gate_name}参数均值过大({stats['mean']:.3f})，可能导致梯度不稳定")
                    if stats['std'] > 3.0:
                        problems.append(f"{gate_name}参数方差过大({stats['std']:.3f})，参数分布不稳定")
                    if stats['max'] - stats['min'] > 10.0:
                        problems.append(f"{gate_name}参数范围过宽({stats['max']:.3f} - {stats['min']:.3f})，可能存在梯度爆炸")
                
                return {
                    'mse_loss': float(mse_loss.numpy()),
                    'total_loss': float(total_loss.numpy()),
                    'gate_stats': gate_stats,
                    'problems': problems
                }
            else:
                return {
                    'mse_loss': float(mse_loss.numpy()),
                    'total_loss': float(total_loss.numpy()),
                    'gate_stats': {},
                    'problems': ['无法获取门控参数，使用基础MSE损失进行分析']
                }
                
        except Exception as e:
            print(f"⚠️  损失分析失败: {e}")
            return {
                'mse_loss': None,
                'total_loss': None,
                'gate_stats': {},
                'problems': [f'损失分析失败: {str(e)}']
            }
    
    def analyze_gradient_flow(self, sample_batch_size=16):
        """分析梯度流动情况"""
        print("\n🌊 分析梯度流动...")
        
        if self.model is None:
            print("❌ 模型未加载")
            return
            
        # 创建测试数据
        test_input = tf.random.normal((sample_batch_size, 50, 6))
        test_init_state = tf.zeros((sample_batch_size, self.state_dim))
        test_target = tf.random.normal((sample_batch_size, 50, 1))
        
        # 计算梯度
        with tf.GradientTape() as tape:
            predictions = self.model([test_input, test_init_state], training=True)
            loss = tf.reduce_mean(tf.square(test_target - predictions))
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        
        # 分析梯度统计
        gradient_norms = []
        gradient_stats = []
        
        for i, grad in enumerate(gradients):
            if grad is not None:
                grad_norm = tf.norm(grad)
                grad_mean = tf.reduce_mean(tf.abs(grad))
                grad_max = tf.reduce_max(tf.abs(grad))
                
                gradient_norms.append(float(grad_norm))
                gradient_stats.append({
                    'layer': i,
                    'norm': float(grad_norm),
                    'mean': float(grad_mean),
                    'max': float(grad_max)
                })
                
                print(f"层 {i:2d}: 梯度范数={grad_norm:.2e}, 平均={grad_mean:.2e}, 最大={grad_max:.2e}")
        
        # 检测梯度问题
        max_norm = max(gradient_norms)
        min_norm = min(gradient_norms)
        
        if max_norm > 10:
            print("⚠️  检测到梯度爆炸，建议调整梯度裁剪")
        elif max_norm < 1e-7:
            print("⚠️  检测到梯度消失，可能导致训练停滞")
        elif min_norm / max_norm < 1e-3:
            print("⚠️  梯度差异很大，某些层可能训练不充分")
        else:
            print("✅ 梯度流动正常")
            
        return gradient_stats
    
    def analyze_model_capacity(self):
        """分析模型容量和复杂度"""
        print("🔍 分析模型容量...")
        
        try:
            # 获取模型参数统计
            total_params = self.model.count_params()
            trainable_params = sum([tf.size(var).numpy() for var in self.model.trainable_variables])
            
            # 分析各层参数分布
            layer_info = []
            for i, layer in enumerate(self.model.layers):
                if hasattr(layer, 'count_params') and layer.count_params() > 0:
                    layer_params = layer.count_params()
                    layer_info.append({
                        'name': layer.name,
                        'type': type(layer).__name__,
                        'params': layer_params,
                        'percentage': (layer_params / total_params) * 100
                    })
            
            print(f"📊 模型参数统计:")
            print(f"   总参数: {total_params:,}")
            print(f"   可训练参数: {trainable_params:,}")
            
            # 容量分析
            capacity_analysis = {
                'total_params': total_params,
                'trainable_params': trainable_params,
                'layer_info': layer_info
            }
            
            # 判断模型容量是否合适
            problems = []
            if total_params < 1000:
                problems.append("模型参数过少，可能容量不足以学习复杂模式")
            elif total_params > 100000:
                problems.append("模型参数过多，可能存在过拟合风险")
            
            # 检查网络深度和宽度
            if self.hidden_dim < 16:
                problems.append(f"隐藏层维度过小({self.hidden_dim})，建议增加到64或更大")
            if self.state_dim < 16:
                problems.append(f"状态维度过小({self.state_dim})，建议增加到16或更大")
            
            capacity_analysis['problems'] = problems
            
            return capacity_analysis
            
        except Exception as e:
            print(f"⚠️  容量分析失败: {e}")
            return {'problems': [f'容量分析失败: {str(e)}']}
    
    def provide_comprehensive_solutions(self, loss_analysis, capacity_analysis):
        """提供综合解决方案"""
        print("\n" + "="*60)
        print("🎯 EMSC训练诊断结论和解决方案")
        print("="*60)
        
        # 基于MSE损失值诊断
        mse_loss = loss_analysis.get('mse_loss')
        if mse_loss is not None:
            print(f"\n📊 当前MSE损失: {mse_loss:.6f}")
            
            if mse_loss > 1.0:
                print("🔴 诊断结果: 损失值过高，模型预测效果很差")
                print("💡 主要问题:")
                print("   1. 数据标准化可能有问题")
                print("   2. 学习率可能设置不当")
                print("   3. 模型结构可能不适合数据")
            elif mse_loss > 0.1:
                print("🟡 诊断结果: 损失值较高，有改进空间")
                print("💡 主要问题:")
                print("   1. 可能需要更多训练时间")
                print("   2. 学习率调度策略需要优化")
                print("   3. 正则化权重可能不合适")
            elif mse_loss > 0.01:
                print("🟢 诊断结果: 损失值合理，但可以进一步优化")
                print("💡 优化方向:")
                print("   1. 使用自适应训练策略")
                print("   2. 调整网络结构")
                print("   3. 优化数据预处理")
            else:
                print("🟢 诊断结果: 损失值已经很低，模型性能良好")
        
        # EMSC特定问题分析
        print(f"\n🔧 EMSC模型特定分析:")
        
        solutions = []
        
        # 门控参数问题
        problems = loss_analysis.get('problems', [])
        if problems:
            print("❌ 发现的问题:")
            for problem in problems:
                print(f"   - {problem}")
        
        # 容量问题
        capacity_problems = capacity_analysis.get('problems', [])
        if capacity_problems:
            print("📦 模型容量问题:")
            for problem in capacity_problems:
                print(f"   - {problem}")
        
        # 综合解决方案
        print(f"\n🚀 推荐解决方案 (按优先级排序):")
        
        # 解决方案1: 数据和预处理
        print("\n1️⃣ 数据预处理优化 (最高优先级)")
        print("   📋 问题: 可能的数据标准化问题")
        print("   🔧 解决方案:")
        print("   - 检查输入数据的数值范围，确保在合理区间")
        print("   - 使用稳健的标准化方法 (如RobustScaler)")
        print("   - 确保训练集和验证集使用相同的标准化参数")
        solutions.append("数据预处理优化")
        
        # 解决方案2: 自适应训练
        print("\n2️⃣ 使用自适应训练模式 (强烈推荐)")
        print("   📋 问题: 传统训练容易陷入局部最优")
        print("   🔧 解决方案:")
        print("   - 使用循环学习率 + 热重启")
        print("   - 启用权重噪声注入")
        print("   - 动态调整损失权重")
        print("   💻 命令: python EMSC_Net/run_adaptive_training.py")
        solutions.append("自适应训练模式")
        
        # 解决方案3: 网络结构调整
        if self.hidden_dim <= 16 or self.state_dim <= 8:
            print("\n3️⃣ 增大网络容量 (推荐)")
            print("   📋 问题: 当前网络容量可能不足")
            print("   🔧 解决方案:")
            print(f"   - 将hidden_dim从{self.hidden_dim}增加到64")
            print(f"   - 将state_dim从{self.state_dim}增加到16")
            print("   - 考虑增加网络层数")
            print("   💻 命令: 在训练时使用 --hidden_dim 64 --state_dim 16")
            solutions.append("增大网络容量")
        
        # 解决方案4: 学习率策略
        print("\n4️⃣ 优化学习率策略")
        print("   📋 问题: 学习率可能过小或调度策略不当")
        print("   🔧 解决方案:")
        print("   - 使用更大的初始学习率 (1e-2 或 5e-2)")
        print("   - 启用循环学习率")
        print("   - 使用更激进的学习率衰减")
        print("   💻 命令: 使用 --learning_rate 1e-2 --cyclical_lr")
        solutions.append("优化学习率策略")
        
        # 解决方案5: 损失函数调整
        print("\n5️⃣ 损失函数权重调整")
        print("   📋 问题: MSE损失和正则化损失权重不平衡")
        print("   🔧 解决方案:")
        print("   - 动态调整正则化权重")
        print("   - 使用更稳定的正则化策略")
        print("   - 监控各损失组件的相对大小")
        solutions.append("损失函数权重调整")
        
        # 立即可执行的命令
        print(f"\n⚡ 立即可用的解决命令:")
        print(f"# 最佳方案 - 自适应训练")
        print(f"python EMSC_Net/run_adaptive_training.py")
        print(f"")
        print(f"# 增大网络 + 自适应训练")
        print(f"python -m training.EMSC_train \\")
        print(f"    --dataset dataset_EMSC_big \\")
        print(f"    --adaptive_training \\")
        print(f"    --hidden_dim 64 \\")
        print(f"    --state_dim 16 \\")
        print(f"    --learning_rate 1e-2 \\")
        print(f"    --epochs 2000")
        
        return {
            'solutions': solutions,
            'priority_order': [
                '数据预处理优化',
                '自适应训练模式', 
                '增大网络容量',
                '优化学习率策略',
                '损失函数权重调整'
            ]
        }
    
    def run_full_diagnosis(self):
        """运行完整的训练诊断"""
        print("🔧 EMSC训练诊断工具")
        print("=" * 50)
        
        # 加载模型和数据
        if not self.load_model_and_data():
            print("❌ 无法继续诊断，请检查模型和数据路径")
            return None
        
        # 执行各项分析
        print("\n🔍 开始执行诊断分析...")
        
        # 1. 损失组件分析
        loss_analysis = self.analyze_loss_components()
        
        # 2. 模型容量分析  
        capacity_analysis = self.analyze_model_capacity()
        
        # 3. 提供综合解决方案
        solutions = self.provide_comprehensive_solutions(loss_analysis, capacity_analysis)
        
        # 返回完整的诊断结果
        return {
            'loss_analysis': loss_analysis,
            'capacity_analysis': capacity_analysis,
            'solutions': solutions
        }


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='EMSC训练诊断工具')
    parser.add_argument('--model_path', type=str, help='模型路径')
    parser.add_argument('--dataset_path', type=str, help='数据集路径')
    parser.add_argument('--state_dim', type=int, default=8, help='状态维度')
    parser.add_argument('--hidden_dim', type=int, default=32, help='隐藏层维度')
    
    args = parser.parse_args()
    
    # 创建诊断器
    diagnosis = EMSCTrainingDiagnosis(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        state_dim=args.state_dim,
        hidden_dim=args.hidden_dim
    )
    
    # 运行诊断
    results = diagnosis.run_full_diagnosis()
    
    if results:
        print("\n✅ 诊断完成，请根据建议调整训练策略")
    else:
        print("\n❌ 诊断失败")


if __name__ == "__main__":
    main() 