"""
阿里云OSS数据集下载器
"""

import os
import time
from pathlib import Path
from EMSC_oss_config import load_oss_config


class OSSDatasetDownloader:
    """
    OSS数据集下载器
    """
    
    def __init__(self, local_data_dir="/data/emsc_dataset"):
        self.local_data_dir = Path(local_data_dir)
        self.local_data_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🔧 OSS下载器初始化:")
        print(f"- 本地数据目录: {self.local_data_dir}")
    
    def download_dataset_from_oss(self, oss_config, oss_dataset_path, local_dataset_path):
        """
        从OSS下载数据集到本地
        
        Args:
            oss_config: OSS配置字典
            oss_dataset_path: OSS中的数据集路径
            local_dataset_path: 本地保存路径
        
        Returns:
            bool: 下载是否成功
        """
        print(f"📥 开始从OSS下载数据集...")
        print(f"OSS路径: {oss_dataset_path}")
        print(f"本地路径: {local_dataset_path}")
        
        try:
            # 导入oss2库
            import oss2
            
            # 确保本地目录存在
            Path(local_dataset_path).parent.mkdir(parents=True, exist_ok=True)
            
            # 初始化OSS客户端
            auth = oss2.Auth(oss_config['access_key_id'], oss_config['access_key_secret'])
            bucket = oss2.Bucket(auth, oss_config['endpoint'], oss_config['bucket_name'])
            
            print(f"🔗 连接OSS: {oss_config['endpoint']}/{oss_config['bucket_name']}")
            
            # 检查文件是否存在
            try:
                object_info = bucket.head_object(oss_dataset_path)
                file_size = object_info.content_length
                print(f"📊 文件信息: {file_size / (1024*1024):.1f}MB")
            except oss2.exceptions.NoSuchKey:
                print(f"❌ OSS中不存在文件: {oss_dataset_path}")
                return False
            
            # 下载文件到本地磁盘
            start_time = time.time()
            print(f"⬇️  开始下载...")
            
            bucket.get_object_to_file(oss_dataset_path, local_dataset_path)
            
            download_time = time.time() - start_time
            actual_file_size = os.path.getsize(local_dataset_path)
            download_speed = actual_file_size / download_time / (1024*1024)  # MB/s
            
            print(f"✅ 下载完成:")
            print(f"- 文件大小: {actual_file_size / (1024*1024):.1f}MB")
            print(f"- 下载时间: {download_time:.1f}秒")
            print(f"- 下载速度: {download_speed:.1f}MB/s")
            
            return True
                
        except ImportError:
            print(f"❌ 请安装oss2库: pip install oss2")
            return False
        except oss2.exceptions.AccessDenied:
            print(f"❌ OSS访问被拒绝，请检查AccessKey和权限")
            return False
        except oss2.exceptions.InvalidBucketName:
            print(f"❌ 无效的Bucket名称: {oss_config['bucket_name']}")
            return False
        except Exception as e:
            print(f"❌ 下载失败: {e}")
            return False
    
    def download_with_config(self, config_path, local_dataset_path=None):
        """
        使用配置文件下载数据集
        
        Args:
            config_path: OSS配置文件路径
            local_dataset_path: 本地保存路径，None则使用默认路径
        
        Returns:
            str: 下载成功的数据集路径，失败返回None
        """
        if local_dataset_path is None:
            local_dataset_path = self.local_data_dir / "dataset.npz"
        
        try:
            # 加载OSS配置
            oss_config = load_oss_config(config_path)
            oss_dataset_path = oss_config['dataset_path']
            
            # 检查本地文件是否存在
            if os.path.exists(local_dataset_path):
                file_size = os.path.getsize(local_dataset_path)
                print(f"✅ 本地数据集已存在: {local_dataset_path}")
                print(f"   文件大小: {file_size / (1024*1024):.1f}MB")
                return str(local_dataset_path)
            
            # 下载数据集
            success = self.download_dataset_from_oss(
                oss_config, oss_dataset_path, local_dataset_path
            )
            
            if success:
                return str(local_dataset_path)
            else:
                return None
                
        except ValueError as e:
            print(f"❌ 配置错误: {e}")
            return None
        except Exception as e:
            print(f"❌ 下载失败: {e}")
            return None


def download_dataset(config_path, local_path=None):
    """
    简单的数据集下载函数
    
    Args:
        config_path: OSS配置文件路径
        local_path: 本地保存路径
    
    Returns:
        str: 数据集路径
    """
    downloader = OSSDatasetDownloader()
    dataset_path = downloader.download_with_config(config_path, local_path)
    
    if dataset_path is None:
        raise ValueError("数据集下载失败")
    
    return dataset_path


if __name__ == "__main__":
    # 测试下载器
    import sys
    
    if len(sys.argv) < 2:
        print("使用方法: python EMSC_oss_downloader.py <config_path> [local_path]")
        print("示例: python EMSC_oss_downloader.py oss_config.json")
        sys.exit(1)
    
    config_path = sys.argv[1]
    local_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        dataset_path = download_dataset(config_path, local_path)
        print(f"✅ 数据集下载成功: {dataset_path}")
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        sys.exit(1) 