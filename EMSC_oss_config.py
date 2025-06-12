"""
阿里云OSS配置文件
"""

import json


def load_oss_config(config_path):
    """
    从指定路径加载OSS配置
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        dict: OSS配置信息
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 验证配置完整性
        required_keys = ['access_key_id', 'access_key_secret', 'endpoint', 'bucket_name', 'dataset_path']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"配置文件缺少必要字段: {key}")
            if not config[key]:
                raise ValueError(f"配置字段不能为空: {key}")
        
        return config
        
    except FileNotFoundError:
        raise ValueError(f"配置文件不存在: {config_path}")
    
    except json.JSONDecodeError:
        raise ValueError(f"配置文件格式错误: {config_path}") 