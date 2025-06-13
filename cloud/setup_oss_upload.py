 #!/usr/bin/env python3
"""OSS上传功能安装和配置脚本"""

import os
import sys
import subprocess
import json

def install_oss_dependencies():
    """安装OSS相关依赖"""
    print("📦 安装OSS上传依赖...")
    try:
        import oss2
        print("✅ oss2 已安装")
        return True
    except ImportError:
        print("📥 正在安装 oss2...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'oss2'])
            print("✅ oss2 安装成功")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ oss2 安装失败: {e}")
            return False

def check_existing_oss_config():
    """检查现有OSS配置"""
    from .EMSC_oss_uploader import check_oss_config_exists
    
    existing_configs = check_oss_config_exists()
    if existing_configs:
        print(f"✅ 发现现有OSS配置文件:")
        for i, config_path in enumerate(existing_configs, 1):
            print(f"   {i}. {config_path}")
        
        # 测试第一个配置文件
        try:
            from .EMSC_oss_config import load_oss_config
            config = load_oss_config(existing_configs[0])
            print(f"✅ 配置文件有效:")
            print(f"   Bucket: {config['bucket_name']}")
            print(f"   Endpoint: {config['endpoint']}")
            return existing_configs[0]
        except Exception as e:
            print(f"⚠️  配置文件格式有问题: {e}")
            return None
    else:
        print("❌ 未找到现有OSS配置文件")
        print("请先配置OSS下载功能，或手动创建oss_config.json")
        return None

def main():
    print("🚀 EMSC OSS上传功能安装向导")
    print("=" * 50)
    
    # 1. 安装依赖
    if not install_oss_dependencies():
        print("❌ 依赖安装失败，退出")
        sys.exit(1)
    
    # 2. 检查现有OSS配置
    config_path = check_existing_oss_config()
    
    if config_path:
        print("\n🎉 OSS上传功能配置完成!")
        print(f"使用配置文件: {config_path}")
        
        # 测试上传功能
        try:
            from .EMSC_oss_uploader import EMSCOSSUploader
            uploader = EMSCOSSUploader(config_path)
            if uploader.bucket:
                print("✅ OSS连接测试成功")
            else:
                print("⚠️  OSS连接测试失败")
        except Exception as e:
            print(f"⚠️  OSS连接测试失败: {e}")
        
        print("\n📖 使用方法:")
        print("1. 训练时自动上传 (云环境):")
        print("   python EMSC_train.py --epochs 100")
        print("2. 手动上传训练结果:")
        print("   python upload_results_to_oss.py ./network_6-32-32-8-1")
    else:
        print("\n❌ 配置失败")
        print("请先配置OSS下载功能，确保有有效的oss_config.json文件")

if __name__ == "__main__":
    main()