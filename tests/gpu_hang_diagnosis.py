#!/usr/bin/env python3
"""
GPU训练卡顿深度诊断脚本
找出EMSC训练第一个epoch卡死的确切原因
"""

import os
import time
import numpy as np
import tensorflow as tf
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.EMSC_train import detect_environment, setup_gpu_environment
from core.EMSC_model import build_msc_model
from core.EMSC_data import load_dataset_from_npz, create_tf_dataset
from core.EMSC_losses import EMSCLoss

def diagnose_sequence_length_issue():
    """诊断序列长度对性能的影响"""
    print("🔍 EMSC训练卡顿诊断")
    print("=" * 50)
    
    # 设置环境
    env_type = detect_environment()
    print(f"环境类型: {env_type}")
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        setup_gpu_environment(gpus)
    
    tf.keras.mixed_precision.set_global_policy('float32')
    print("已设置float32精度")
    
    # 测试不同序列长度
    test_lengths = [100, 500, 1000, 2000, 3000, 5000]
    
    print("\n📏 测试不同序列长度的性能")
    print("-" * 30)
    
    for length in test_lengths:
        print(f"\n测试序列长度: {length}")
        
        try:
            # 创建模型
            model = build_msc_model(
                state_dim=8,
                input_dim=6,
                output_dim=1,
                hidden_dim=32,
                num_internal_layers=2,
                max_sequence_length=length + 100  # 留点余量
            )
            
            # 编译模型
            model.compile(
                optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-3),
                loss=EMSCLoss(state_dim=8),
                jit_compile=False
            )
            
            # 创建测试数据
            X_data = [np.random.randn(length, 6).astype(np.float32)]
            Y_data = [np.random.randn(length, 1).astype(np.float32)]
            init_states = np.zeros((1, 8), dtype=np.float32)
            
            # 测试预测时间
            print(f"  测试预测...")
            start_time = time.time()
            
            # 执行预测
            output = model([X_data, init_states], training=False)
            
            pred_time = time.time() - start_time
            print(f"  预测时间: {pred_time:.2f}秒")
            
            # 如果预测时间超过30秒，跳过训练测试
            if pred_time > 30:
                print(f"  ⚠️ 预测时间过长，跳过训练测试")
                continue
            
            # 测试训练时间
            print(f"  测试训练...")
            dataset = create_tf_dataset(
                X_data, Y_data, init_states,
                batch_size=1, shuffle=False, num_parallel_calls=1
            )
            
            start_time = time.time()
            
            # 设置训练超时
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("训练超时")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(60)  # 60秒超时
            
            try:
                history = model.fit(
                    dataset,
                    epochs=1,
                    verbose=0
                )
                
                signal.alarm(0)  # 取消超时
                train_time = time.time() - start_time
                print(f"  ✅ 训练时间: {train_time:.2f}秒")
                
            except TimeoutError:
                signal.alarm(0)
                print(f"  ❌ 训练超时 (>60秒)")
                print(f"  💡 序列长度 {length} 导致训练卡死")
                break
                
        except Exception as e:
            print(f"  ❌ 测试失败: {e}")
            continue
    
    print("\n📊 建议:")
    print("1. 如果5000长度的序列卡死，考虑:")
    print("   - 减少序列长度到1000-2000")
    print("   - 使用序列截断策略") 
    print("   - 增加预热步骤")
    print("2. 如果所有长度都正常，问题可能是:")
    print("   - 特定数据的问题")
    print("   - 批处理大小过大")
    print("   - 内存不足")

def test_real_data():
    """测试真实数据"""
    print("\n📂 测试真实数据")
    print("-" * 30)
    
    dataset_path = "/Users/tianyunhu/Documents/temp/code/Test_app/EMSC_Model/msc_models/dataset/dataset.npz"
    
    if not os.path.exists(dataset_path):
        print(f"数据集不存在: {dataset_path}")
        return
    
    # 加载数据
    X_paths, Y_paths = load_dataset_from_npz(dataset_path)
    if X_paths is None:
        print("数据加载失败")
        return
    
    print(f"数据集信息:")
    print(f"  序列数量: {len(X_paths)}")
    print(f"  序列长度: {len(X_paths[0])}")
    
    # 创建模型
    model = build_msc_model(
        state_dim=8, input_dim=6, output_dim=1,
        hidden_dim=32, max_sequence_length=6000
    )
    
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-3),
        loss=EMSCLoss(state_dim=8),
        jit_compile=False
    )
    
    # 测试单个样本
    print("\n测试单个样本训练...")
    
    X_sample = [X_paths[0]]
    Y_sample = [Y_paths[0]]
    init_states = np.zeros((1, 8), dtype=np.float32)
    
    dataset = create_tf_dataset(
        X_sample, Y_sample, init_states,
        batch_size=1, shuffle=False, num_parallel_calls=1
    )
    
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError("训练超时")
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(120)  # 2分钟超时
    
    try:
        start_time = time.time()
        history = model.fit(dataset, epochs=1, verbose=1)
        train_time = time.time() - start_time
        
        signal.alarm(0)
        print(f"✅ 单样本训练成功: {train_time:.2f}秒")
        
    except TimeoutError:
        signal.alarm(0)
        print("❌ 单样本训练超时")
        print("💡 确认是5000长度序列导致的问题")
    except Exception as e:
        signal.alarm(0)
        print(f"❌ 训练失败: {e}")

if __name__ == "__main__":
    try:
        diagnose_sequence_length_issue()
        test_real_data()
    except KeyboardInterrupt:
        print("\n🛑 诊断被用户中断")
    except Exception as e:
        print(f"\n💥 诊断异常: {e}")
        import traceback
        traceback.print_exc() 