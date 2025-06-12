import time
import numpy as np
import tensorflow as tf
from EMSC_data import create_tf_dataset
from EMSC_loss import EMSCLoss
from timeout import timeout

def test_training_step(model, X_paths, Y_paths):
    """测试训练步骤"""
    print("\n🏃 测试训练步骤")
    print("-" * 30)
    
    if model is None or X_paths is None or Y_paths is None:
        print("    ❌ 模型或数据为空，跳过测试")
        return False
    
    try:
        # 使用小批量数据测试
        batch_size = 1  # 使用单个样本避免批次问题
        
        print(f"  准备训练数据 (批次大小: {batch_size})...")
        
        # 取单个样本
        X_sample = [X_paths[0]]  # 保持列表格式
        Y_sample = [Y_paths[0]]  # 保持列表格式
        init_states = np.zeros((batch_size, 8), dtype=np.float32)
        
        print(f"  数据形状检查:")
        print(f"    X_sample[0] shape: {np.array(X_sample[0]).shape}")
        print(f"    Y_sample[0] shape: {np.array(Y_sample[0]).shape}")
        print(f"    init_states shape: {init_states.shape}")
        
        # 创建TensorFlow数据集
        print("  创建TensorFlow数据集...")
        try:
            train_dataset = create_tf_dataset(
                X_sample, Y_sample, init_states,
                batch_size=batch_size,
                shuffle=False,
                num_parallel_calls=1
            )
            print("    ✅ 数据集创建成功")
        except Exception as e:
            print(f"    ❌ 数据集创建失败: {e}")
            return False
        
        print("  执行训练步骤...")
        with timeout(120):  # 2分钟超时
            start_time = time.time()
            
            # 创建自定义损失函数
            custom_loss = EMSCLoss(state_dim=8)
            model.compile(
                optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-3),  # 使用legacy优化器
                loss=custom_loss,
                jit_compile=False
            )
            
            # 执行一个训练步骤
            history = model.fit(
                train_dataset,
                epochs=1,
                verbose=1
            )
            
            exec_time = time.time() - start_time
            print(f"    ✅ 训练步骤完成: {exec_time:.2f}秒")
            return True
            
    except TimeoutError as e:
        print(f"    ❌ 训练步骤超时: {e}")
        return False
    except Exception as e:
        print(f"    ❌ 训练步骤失败: {e}")
        import traceback
        traceback.print_exc()
        return False 