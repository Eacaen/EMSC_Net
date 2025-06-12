#!/usr/bin/env python3
"""
GPU预热测试脚本
测试EMSC模型在本地GPU环境下的预热和优化效果
"""

import os
import time
import numpy as np
import tensorflow as tf
from EMSC_train import detect_environment, setup_gpu_environment, warmup_gpu_model
from EMSC_model import build_msc_model

def test_gpu_warmup():
    """测试GPU预热功能"""
    print("🧪 EMSC GPU预热测试")
    print("=" * 50)
    
    # 检测环境
    env_type = detect_environment()
    print(f"🔍 检测到环境类型: {env_type}")
    
    # 设置GPU环境
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("❌ 未检测到GPU设备")
        return False
    
    print(f"🎮 发现GPU设备: {len(gpus)}")
    setup_gpu_environment(gpus)
    
    # 强制使用float32精度
    tf.keras.mixed_precision.set_global_policy('float32')
    tf.keras.backend.set_floatx('float32')
    
    # 创建测试模型
    print("\n🏗️  创建EMSC测试模型...")
    model = build_msc_model(
        state_dim=8,
        input_dim=6,
        output_dim=1,
        hidden_dim=32,
        num_internal_layers=2,
        max_sequence_length=5000
    )
    
    # 编译模型
    model.compile(
        optimizer='adam',
        loss='mse',
        jit_compile=False  # EMSC不支持JIT编译
    )
    
    print("✅ 模型创建和编译完成")
    
    # 测试1: 不预热的执行时间
    print("\n⏱️  测试1: 第一次执行（不预热）")
    test_input = tf.random.normal((2, 1000, 6), dtype=tf.float32)
    test_init_state = tf.zeros((2, 8), dtype=tf.float32)
    
    start_time = time.time()
    try:
        output1 = model([test_input, test_init_state], training=False)
        first_exec_time = time.time() - start_time
        print(f"  🕐 第一次执行时间: {first_exec_time:.2f}秒")
        print(f"  📊 输出形状: {output1.shape}")
    except Exception as e:
        print(f"  ❌ 第一次执行失败: {e}")
        return False
    
    # 测试2: 预热后的执行时间
    print("\n🔥 执行GPU模型预热...")
    warmup_success = warmup_gpu_model(model, sample_batch_size=1, max_sequence_length=200)
    
    if not warmup_success:
        print("⚠️  预热失败，继续测试...")
    
    # 测试3: 预热后的执行时间
    print("\n⏱️  测试2: 预热后执行")
    start_time = time.time()
    try:
        output2 = model([test_input, test_init_state], training=False)
        second_exec_time = time.time() - start_time
        print(f"  ⚡ 预热后执行时间: {second_exec_time:.2f}秒")
        print(f"  📊 输出形状: {output2.shape}")
        
        # 计算加速比
        if first_exec_time > 0:
            speedup = first_exec_time / second_exec_time
            print(f"  🚀 加速比: {speedup:.2f}x")
        
    except Exception as e:
        print(f"  ❌ 预热后执行失败: {e}")
        return False
    
    # 测试4: 连续执行性能
    print("\n⏱️  测试3: 连续执行性能")
    exec_times = []
    for i in range(3):
        start_time = time.time()
        _ = model([test_input, test_init_state], training=False)
        exec_time = time.time() - start_time
        exec_times.append(exec_time)
        print(f"  执行 {i+1}: {exec_time:.2f}秒")
    
    avg_time = np.mean(exec_times)
    print(f"  📈 平均执行时间: {avg_time:.2f}秒")
    print(f"  📉 时间标准差: {np.std(exec_times):.3f}秒")
    
    # 性能评估
    print("\n📋 性能评估总结")
    print("-" * 30)
    print(f"环境类型: {env_type}")
    print(f"第一次执行: {first_exec_time:.2f}秒")
    print(f"预热后执行: {second_exec_time:.2f}秒")
    print(f"平均执行时间: {avg_time:.2f}秒")
    
    if warmup_success:
        print("✅ GPU预热功能正常工作")
    else:
        print("⚠️  GPU预热遇到问题，但执行仍可继续")
    
    if second_exec_time < first_exec_time * 0.5:
        print("🎉 预热显著提升了执行速度！")
    elif second_exec_time < first_exec_time:
        print("✅ 预热有助于提升执行速度")
    else:
        print("🤔 预热效果不明显，可能需要进一步优化")
    
    print("\n💡 建议:")
    if env_type == 'local':
        print("  - 本地环境已应用优化配置")
        print("  - 训练前会自动执行预热")
        print("  - 建议使用较小的batch_size")
    else:
        print("  - 云环境配置适合大规模训练")
        print("  - 可以使用较大的batch_size")
    
    return True

def test_training_warmup():
    """测试训练时的预热效果"""
    print("\n🏋️  训练预热效果测试")
    print("-" * 30)
    
    # 创建小规模训练数据
    batch_size = 2
    sequence_length = 500
    
    train_input = tf.random.normal((batch_size, sequence_length, 6), dtype=tf.float32)
    train_init_state = tf.zeros((batch_size, 8), dtype=tf.float32)
    train_target = tf.random.normal((batch_size, sequence_length, 1), dtype=tf.float32)
    
    # 创建模型
    model = build_msc_model(
        state_dim=8, input_dim=6, output_dim=1,
        hidden_dim=32, max_sequence_length=5000
    )
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse',
        jit_compile=False
    )
    
    # 执行预热
    print("🔥 执行训练预热...")
    warmup_gpu_model(model, sample_batch_size=1, max_sequence_length=100)
    
    # 测试训练步骤
    print("🏃 执行训练步骤...")
    start_time = time.time()
    
    history = model.fit(
        [train_input, train_init_state],
        train_target,
        epochs=2,
        batch_size=batch_size,
        verbose=1
    )
    
    training_time = time.time() - start_time
    print(f"✅ 训练完成，总时间: {training_time:.2f}秒")
    
    return True

if __name__ == "__main__":
    print("🚀 开始EMSC GPU优化测试")
    
    try:
        # 基础预热测试
        basic_success = test_gpu_warmup()
        
        if basic_success:
            # 训练预热测试
            training_success = test_training_warmup()
            
            if training_success:
                print("\n🎉 所有测试通过！")
                print("现在可以正常运行EMSC训练:")
                print("  python EMSC_train.py --epochs 100 --device gpu")
            else:
                print("\n⚠️  训练测试遇到问题")
        else:
            print("\n❌ 基础测试失败")
            
    except KeyboardInterrupt:
        print("\n🛑 测试被用户中断")
    except Exception as e:
        print(f"\n💥 测试过程中出现异常: {e}")
        import traceback
        traceback.print_exc() 