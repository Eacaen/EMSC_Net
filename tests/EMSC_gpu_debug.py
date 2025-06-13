"""
GPU数值稳定性诊断工具
用于识别和修复GPU训练中的NaN问题
"""

import tensorflow as tf
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.EMSC_losses import EMSCLoss


class GPUNumericalDiagnostic:
    """GPU数值稳定性诊断器"""
    
    def __init__(self):
        self.issues_found = []
    
    def diagnose_environment(self):
        """诊断GPU环境配置"""
        print("🔍 GPU环境诊断...")
        
        # 检查GPU设备
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            print("❌ 未检测到GPU设备")
            return False
        
        print(f"✅ 检测到{len(gpus)}个GPU设备")
        
        # 检查混合精度设置
        policy = tf.keras.mixed_precision.global_policy()
        print(f"🔍 当前精度策略: {policy.name}")
        
        if 'mixed_float16' in policy.name:
            print("⚠️  检测到混合精度，可能导致数值不稳定")
            self.issues_found.append("Mixed precision detected")
        
        # 检查TensorFloat-32设置
        try:
            tf32_enabled = tf.config.experimental.tensor_float_32_execution_enabled()
            if tf32_enabled:
                print("⚠️  TensorFloat-32已启用，可能影响数值精度")
                self.issues_found.append("TensorFloat-32 enabled")
            else:
                print("✅ TensorFloat-32已禁用")
        except:
            print("⚠️  无法检查TensorFloat-32状态")
        
        return True
    
    def test_basic_operations(self):
        """测试基本GPU操作的数值稳定性"""
        print("\n🧪 测试基本GPU操作...")
        
        with tf.device('/GPU:0'):
            # 测试基本数学运算
            a = tf.constant([1e10, 1e-10, 0.0], dtype=tf.float32)
            b = tf.constant([1e-10, 1e10, 1.0], dtype=tf.float32)
            
            # 加法
            add_result = a + b
            if tf.reduce_any(tf.math.is_nan(add_result)):
                print("❌ 基本加法产生NaN")
                self.issues_found.append("Basic addition produces NaN")
            else:
                print("✅ 基本加法正常")
            
            # 除法
            div_result = a / tf.maximum(b, 1e-7)
            if tf.reduce_any(tf.math.is_nan(div_result)):
                print("❌ 安全除法产生NaN")
                self.issues_found.append("Safe division produces NaN")
            else:
                print("✅ 安全除法正常")
            
            # 平方根
            sqrt_result = tf.sqrt(tf.abs(a))
            if tf.reduce_any(tf.math.is_nan(sqrt_result)):
                print("❌ 平方根计算产生NaN")
                self.issues_found.append("Square root produces NaN")
            else:
                print("✅ 平方根计算正常")
    
    def test_loss_function(self):
        """测试EMSC损失函数的数值稳定性"""
        print("\n🧪 测试EMSC损失函数...")
        
        with tf.device('/GPU:0'):
            # 创建测试数据
            batch_size, seq_len = 2, 10
            y_true = tf.random.normal((batch_size, seq_len, 1), dtype=tf.float32)
            y_pred = tf.random.normal((batch_size, seq_len, 1), dtype=tf.float32)
            
            # 测试基本MSE损失
            loss_fn = EMSCLoss(state_dim=8)
            
            try:
                basic_loss = loss_fn(y_true, y_pred)
                if tf.math.is_nan(basic_loss):
                    print("❌ 基本MSE损失产生NaN")
                    self.issues_found.append("Basic MSE loss produces NaN")
                else:
                    print(f"✅ 基本MSE损失正常: {basic_loss:.6f}")
            except Exception as e:
                print(f"❌ 损失计算出错: {e}")
                self.issues_found.append(f"Loss computation error: {e}")
            
            # 测试带门控参数的损失
            try:
                state_dim = 8
                gate_params = {
                    'alpha': tf.random.normal((batch_size, seq_len, state_dim), dtype=tf.float32),
                    'beta': tf.random.normal((batch_size, seq_len, state_dim), dtype=tf.float32),
                    'gamma': tf.random.normal((batch_size, seq_len, state_dim), dtype=tf.float32)
                }
                
                gate_loss = loss_fn(y_true, y_pred, gate_params)
                if tf.math.is_nan(gate_loss):
                    print("❌ 门控损失产生NaN")
                    self.issues_found.append("Gate loss produces NaN")
                else:
                    print(f"✅ 门控损失正常: {gate_loss:.6f}")
            except Exception as e:
                print(f"❌ 门控损失计算出错: {e}")
                self.issues_found.append(f"Gate loss computation error: {e}")
    
    def test_extreme_values(self):
        """测试极值情况"""
        print("\n🧪 测试极值情况...")
        
        with tf.device('/GPU:0'):
            loss_fn = EMSCLoss(state_dim=8)
            
            # 测试极大值
            y_true_large = tf.constant([[[1e6]], [[1e6]]], dtype=tf.float32)
            y_pred_large = tf.constant([[[1e6 + 1]], [[1e6 - 1]]], dtype=tf.float32)
            
            try:
                large_loss = loss_fn(y_true_large, y_pred_large)
                if tf.math.is_nan(large_loss):
                    print("❌ 极大值测试产生NaN")
                    self.issues_found.append("Large values produce NaN")
                else:
                    print(f"✅ 极大值测试正常: {large_loss:.6f}")
            except Exception as e:
                print(f"❌ 极大值测试出错: {e}")
            
            # 测试极小值
            y_true_small = tf.constant([[[1e-10]], [[1e-10]]], dtype=tf.float32)
            y_pred_small = tf.constant([[[1e-10 + 1e-12]], [[1e-10 - 1e-12]]], dtype=tf.float32)
            
            try:
                small_loss = loss_fn(y_true_small, y_pred_small)
                if tf.math.is_nan(small_loss):
                    print("❌ 极小值测试产生NaN")
                    self.issues_found.append("Small values produce NaN")
                else:
                    print(f"✅ 极小值测试正常: {small_loss:.6f}")
            except Exception as e:
                print(f"❌ 极小值测试出错: {e}")
    
    def provide_recommendations(self):
        """提供修复建议"""
        print("\n💡 修复建议:")
        
        if not self.issues_found:
            print("✅ 未发现数值稳定性问题！")
            return
        
        print("发现以下问题:")
        for issue in self.issues_found:
            print(f"   - {issue}")
        
        print("\n建议修复方案:")
        print("1. 确保使用float32精度:")
        print("   tf.keras.mixed_precision.set_global_policy('float32')")
        
        print("2. 禁用TensorFloat-32:")
        print("   tf.config.experimental.enable_tensor_float_32_execution(False)")
        
        print("3. 启用数值检查:")
        print("   tf.debugging.enable_check_numerics()")
        
        print("4. 在损失函数中添加数值保护:")
        print("   - 检查NaN/Inf并替换为有效值")
        print("   - 使用安全除法避免除零")
        print("   - 限制数值范围避免溢出")
        
        print("5. 降低学习率:")
        print("   learning_rate = 1e-4  # 或更小")
        
        print("6. 使用梯度裁剪:")
        print("   optimizer = Adam(learning_rate=1e-3, clipnorm=1.0)")
    
    def run_full_diagnosis(self):
        """运行完整诊断"""
        print("🚀 GPU数值稳定性完整诊断")
        print("=" * 50)
        
        if not self.diagnose_environment():
            return
        
        self.test_basic_operations()
        self.test_loss_function()
        self.test_extreme_values()
        
        print("\n" + "=" * 50)
        self.provide_recommendations()


def run_gpu_debug():
    """运行GPU调试"""
    diagnostic = GPUNumericalDiagnostic()
    diagnostic.run_full_diagnosis()


if __name__ == "__main__":
    run_gpu_debug() 