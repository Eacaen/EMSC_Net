#!/usr/bin/env python3
"""
GPU环境验证脚本 - EMSC兼容版本
验证GPU配置是否正确，测试EMSC模型兼容性
"""

import tensorflow as tf
import numpy as np

def verify_gpu_environment():
    """验证GPU环境配置"""
    print("🔍 GPU环境验证")
    print("=" * 50)
    
    # 检查TensorFlow版本
    print(f"TensorFlow版本: {tf.__version__}")
    
    # 检查GPU设备
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("❌ 未检测到GPU设备")
        return False
    
    print(f"✅ 检测到 {len(gpus)} 个GPU设备:")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu}")
    
    # 检查GPU内存配置
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU内存增长已启用")
    except Exception as e:
        print(f"⚠️  GPU内存配置警告: {e}")
    
    # 检查关键配置
    print(f"XLA JIT状态: {tf.config.optimizer.get_jit()}")
    print(f"TensorFloat-32状态: {tf.config.experimental.is_tensor_float_32_execution_enabled()}")
    print(f"Mixed Precision策略: {tf.keras.mixed_precision.global_policy().name}")
    
    return True

def test_while_loop_compatibility():
    """测试while_loop兼容性（EMSC核心功能）"""
    print("\n🔄 While Loop兼容性测试")
    print("-" * 30)
    
    try:
        # 模拟EMSC中的while_loop结构
        @tf.function
        def emsc_style_loop(initial_state, inputs):
            """模拟EMSC中的while_loop操作"""
            
            def condition(i, state, inputs):
                return i < tf.shape(inputs)[1]
            
            def body(i, state, inputs):
                # 模拟MSC_Cell的操作
                current_input = inputs[:, i, :]
                
                # 模拟状态更新
                new_state = state + tf.reduce_mean(current_input, axis=1, keepdims=True)
                
                return i + 1, new_state, inputs
            
            # 使用while_loop
            final_i, final_state, _ = tf.while_loop(
                condition, body,
                [0, initial_state, inputs],
                shape_invariants=[
                    tf.TensorShape([]),
                    tf.TensorShape([None, None]),
                    tf.TensorShape([None, None, None])
                ]
            )
            
            return final_state
        
        # 测试执行
        with tf.device('/GPU:0'):
            batch_size = 4
            sequence_length = 10
            state_dim = 8
            input_dim = 6
            
            initial_state = tf.zeros((batch_size, state_dim), dtype=tf.float32)
            inputs = tf.random.normal((batch_size, sequence_length, input_dim), dtype=tf.float32)
            
            result = emsc_style_loop(initial_state, inputs)
            
        print(f"✅ While Loop测试成功")
        print(f"  输入形状: {inputs.shape}")
        print(f"  初始状态: {initial_state.shape}")
        print(f"  输出状态: {result.shape}")
        print(f"  输出范围: [{result.numpy().min():.3f}, {result.numpy().max():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ While Loop测试失败: {e}")
        return False

def test_gradient_computation():
    """测试梯度计算（关键的数值稳定性）"""
    print("\n📊 梯度计算测试")
    print("-" * 30)
    
    try:
        # 创建一个简单的模型来测试梯度
        class SimpleModel(tf.keras.Model):
            def __init__(self):
                super().__init__()
                self.dense1 = tf.keras.layers.Dense(10, activation='tanh')
                self.dense2 = tf.keras.layers.Dense(1)
            
            def call(self, inputs):
                x = self.dense1(inputs)
                return self.dense2(x)
        
        model = SimpleModel()
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01, clipnorm=1.0, clipvalue=0.5)
        
        # 模拟训练步骤
        with tf.device('/GPU:0'):
            x = tf.random.normal((32, 6), dtype=tf.float32)
            y = tf.random.normal((32, 1), dtype=tf.float32)
            
            with tf.GradientTape() as tape:
                predictions = model(x)
                loss = tf.reduce_mean(tf.square(predictions - y))
            
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        # 检查梯度是否有效
        has_valid_gradients = all(
            grad is not None and not tf.reduce_any(tf.math.is_nan(grad))
            for grad in gradients
        )
        
        if has_valid_gradients:
            print("✅ 梯度计算正常")
            print(f"  损失值: {loss.numpy():.6f}")
            print(f"  梯度数量: {len(gradients)}")
            print("  梯度范围:")
            for i, grad in enumerate(gradients):
                if grad is not None:
                    print(f"    Layer {i}: [{grad.numpy().min():.6f}, {grad.numpy().max():.6f}]")
        else:
            print("❌ 检测到无效梯度（NaN或None）")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 梯度计算测试失败: {e}")
        return False

def main():
    """主函数"""
    print("EMSC GPU环境验证工具 - 兼容版本")
    print("=" * 50)
    
    # 配置GPU环境（完全模拟EMSC_train.py的配置）
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # 启用内存增长
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # 设置可见设备
            tf.config.set_visible_devices(gpus[0], 'GPU')
            
            # 配置数值精度（禁用TensorFloat-32）
            tf.config.experimental.enable_tensor_float_32_execution(False)
            
            # 禁用XLA（EMSC模型使用while_loop，与XLA不兼容）
            tf.config.optimizer.set_jit(False)
            
            print("✅ GPU环境配置完成（EMSC兼容模式）")
            
        except Exception as e:
            print(f"❌ GPU环境配置失败: {e}")
            return
    
    # 强制使用float32（禁用混合精度）
    tf.keras.mixed_precision.set_global_policy('float32')
    tf.keras.backend.set_floatx('float32')
    print("✅ 已设置float32精度策略（禁用混合精度）")
    
    # 执行验证测试
    print()
    gpu_ok = verify_gpu_environment()
    while_loop_ok = test_while_loop_compatibility()
    gradient_ok = test_gradient_computation()
    
    # 总结
    print("\n📋 验证总结")
    print("=" * 50)
    if gpu_ok and while_loop_ok and gradient_ok:
        print("🎉 所有测试通过！GPU环境已准备就绪")
        print("\n💡 EMSC GPU配置说明:")
        print("  ✅ 已禁用XLA JIT编译")
        print("     - 原因：EMSC使用tf.while_loop，创建动态控制流")
        print("     - XLA要求静态图结构，与while_loop不兼容")
        print("  ✅ 已禁用数值检查（tf.debugging.enable_check_numerics）")
        print("     - 原因：与XLA编译冲突")
        print("     - 替代：梯度裁剪 + EMSCLoss数值保护")
        print("  ✅ 已禁用混合精度训练")
        print("     - 原因：确保GPU和CPU数值一致性")
        print("  ✅ 已禁用TensorFloat-32")
        print("     - 原因：提高数值精度")
        print("  ✅ 梯度裁剪已启用")
        print("     - clipnorm=1.0, clipvalue=0.5")
    else:
        print("⚠️  部分测试失败，请检查配置")
        if not gpu_ok:
            print("  - GPU基础环境有问题")
        if not while_loop_ok:
            print("  - While Loop兼容性有问题")
        if not gradient_ok:
            print("  - 梯度计算有问题")
    
    print("\n🚀 现在可以运行EMSC训练:")
    print("  cd EMSC_Net")
    print("  python train.py --epochs 200 --batch_size 128")
    print("\n📝 如果仍有问题，建议:")
    print("  1. 重启Python环境")
    print("  2. 检查CUDA版本兼容性")
    print("  3. 尝试更小的batch_size")

if __name__ == '__main__':
    main() 