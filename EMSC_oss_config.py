"""
阿里云OSS配置文件
"""

import os
import json


# OSS配置模板
OSS_CONFIG_TEMPLATE = {
    "access_key_id": "your_access_key_id",
    "access_key_secret": "your_access_key_secret", 
    "endpoint": "https://oss-cn-hangzhou.aliyuncs.com",  # 根据您的区域调整
    "bucket_name": "your_bucket_name",
    "dataset_path": "path/to/your/dataset.npz"  # OSS中数据集的路径
}


def create_oss_config_file(config_path="oss_config.json"):
    """
    创建OSS配置文件模板
    
    Args:
        config_path: 配置文件路径
    """
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(OSS_CONFIG_TEMPLATE, f, indent=4, ensure_ascii=False)
    
    print(f"📝 OSS配置文件已创建: {config_path}")
    print(f"请编辑该文件，填入您的OSS配置信息")


def load_oss_config(config_path="oss_config.json"):
    """
    加载OSS配置
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        dict: OSS配置信息
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 验证配置完整性
        required_keys = ['access_key_id', 'access_key_secret', 'endpoint', 'bucket_name']
        for key in required_keys:
            if not config.get(key) or config[key] == f"your_{key}":
                raise ValueError(f"请在 {config_path} 中配置正确的 {key}")
        
        return config
        
    except FileNotFoundError:
        print(f"❌ 配置文件不存在: {config_path}")
        print(f"正在创建配置文件模板...")
        create_oss_config_file(config_path)
        raise ValueError(f"请编辑 {config_path} 并填入正确的OSS配置")
    
    except json.JSONDecodeError:
        raise ValueError(f"配置文件格式错误: {config_path}")


def get_oss_config_from_env():
    """
    从环境变量获取OSS配置
    
    Returns:
        dict: OSS配置信息
    """
    config = {
        'access_key_id': os.getenv('OSS_ACCESS_KEY_ID'),
        'access_key_secret': os.getenv('OSS_ACCESS_KEY_SECRET'),
        'endpoint': os.getenv('OSS_ENDPOINT', 'https://oss-cn-hangzhou.aliyuncs.com'),
        'bucket_name': os.getenv('OSS_BUCKET_NAME'),
        'dataset_path': os.getenv('OSS_DATASET_PATH', 'dataset.npz')
    }
    
    # 检查必要参数
    if not all([config['access_key_id'], config['access_key_secret'], config['bucket_name']]):
        return None
    
    return config


if __name__ == "__main__":
    # 创建配置文件模板
    create_oss_config_file()
    
    print(f"\n💡 配置方式:")
    print(f"方式1 - 配置文件:")
    print(f"   编辑 oss_config.json")
    print(f"方式2 - 环境变量:")
    print(f"   export OSS_ACCESS_KEY_ID='your_key'")
    print(f"   export OSS_ACCESS_KEY_SECRET='your_secret'")
    print(f"   export OSS_BUCKET_NAME='your_bucket'")
    print(f"   export OSS_ENDPOINT='https://oss-cn-hangzhou.aliyuncs.com'")
    print(f"   export OSS_DATASET_PATH='path/to/dataset.npz'") 