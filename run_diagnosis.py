#!/usr/bin/env python3
"""
EMSC训练诊断运行脚本
简化的诊断工具启动脚本
"""

import os
import sys
import json
import numpy as np

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def main():
    """主函数"""
    print("🔧 EMSC训练诊断工具启动器")
    print("=" * 50)
    
    # 用户的具体路径
    model_path = "/Users/tianyunhu/Documents/temp/code/Test_app/EMSC_Model/msc_models/dataset_EMSC_big/network_6-8-8-8-1"
    dataset_path = "/Users/tianyunhu/Documents/temp/code/Test_app/EMSC_Model/msc_models/dataset_EMSC_big/dataset_EMSC_big.tfrecord"
    state_dim = 8
    hidden_dim = 8
    
    print(f"🔍 配置信息:")
    print(f"   模型路径: {model_path}")
    print(f"   数据集路径: {dataset_path}")
    print(f"   状态维度: {state_dim}")
    print(f"   隐藏层维度: {hidden_dim}")
    print()
    
    # 检查路径是否存在
    if not os.path.exists(model_path):
        print(f"⚠️  模型路径不存在: {model_path}")
        print("将使用默认参数创建新模型进行分析")
    else:
        print(f"✅ 模型路径存在")
    
    if not os.path.exists(dataset_path):
        print(f"⚠️  数据集路径不存在: {dataset_path}")
        print("将仅进行模型分析")
    else:
        print(f"✅ 数据集路径存在")
    
    print()
    
    try:
        # 导入并运行诊断
        from utils.EMSC_training_diagnosis import EMSCTrainingDiagnosis
        
        print("🚀 启动诊断...")
        
        # 创建诊断器
        diagnosis = EMSCTrainingDiagnosis(
            model_path=model_path,
            dataset_path=dataset_path,
            state_dim=state_dim,
            hidden_dim=hidden_dim
        )
        
        # 运行诊断
        results = diagnosis.run_full_diagnosis()
        
        if results:
            print("\n" + "="*60)
            print("✅ 诊断完成!")
            print("请根据上述建议调整您的训练策略")
            print("="*60)
            
            # 保存诊断结果
            results_file = "emsc_diagnosis_results.json"
            
            # 转换结果为可序列化格式，处理numpy类型
            def convert_to_serializable(obj):
                """递归转换对象为JSON可序列化格式"""
                if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
                    return int(obj)
                elif isinstance(obj, (np.float64, np.float32, np.float16)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_to_serializable(value) for key, value in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [convert_to_serializable(item) for item in obj]
                else:
                    return obj
            
            serializable_results = convert_to_serializable({
                'loss_analysis': results.get('loss_analysis', {}),
                'capacity_analysis': results.get('capacity_analysis', {}),
                'solutions': results.get('solutions', [])
            })
            
            try:
                with open(results_file, 'w', encoding='utf-8') as f:
                    json.dump(serializable_results, f, indent=2, ensure_ascii=False)
                print(f"\n📄 诊断结果已保存到: {results_file}")
            except Exception as e:
                print(f"保存结果时出错: {e}")
                
        else:
            print("\n❌ 诊断过程中出现问题")
            
    except ImportError as e:
        print(f"❌ 导入模块失败: {e}")
        print("请确保您在正确的项目目录中运行此脚本")
        print("或者尝试: python -m EMSC_Net.run_diagnosis")
        
    except Exception as e:
        print(f"❌ 诊断过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 