"""
阿里云OSS数据集下载器
基于用户提供的oss2代码模板
"""

import os
import time
import shutil
from pathlib import Path
from EMSC_oss_config import load_oss_config, get_oss_config_from_env


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
        从OSS下载数据集到本地 - 基于用户提供的oss2模板
        
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
            
            # 初始化OSS客户端 - 使用用户提供的模板
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
            
            # 下载文件到本地磁盘 - 使用用户提供的方法
            start_time = time.time()
            print(f"⬇️  开始下载...")
            
            # 使用bucket.get_object_to_file方法（如用户模板）
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
    
    def check_and_download_dataset(self, target_dataset_path=None):
        """
        检查数据集是否存在，不存在则从OSS下载
        
        Args:
            target_dataset_path: 目标数据集路径，None则使用默认路径
        
        Returns:
            str: 实际数据集路径
        """
        if target_dataset_path is None:
            target_dataset_path = self.local_data_dir / "dataset.npz"
        
        # 检查本地文件是否存在
        if os.path.exists(target_dataset_path):
            file_size = os.path.getsize(target_dataset_path)
            print(f"✅ 本地数据集已存在: {target_dataset_path}")
            print(f"   文件大小: {file_size / (1024*1024):.1f}MB")
            return str(target_dataset_path)
        
        print(f"📂 本地数据集不存在，开始从OSS下载...")
        
        # 尝试从环境变量获取OSS配置
        oss_config = get_oss_config_from_env()
        
        if oss_config is None:
            # 尝试从配置文件获取
            try:
                oss_config = load_oss_config()
            except ValueError as e:
                print(f"❌ OSS配置错误: {e}")
                return None
        
        # 下载数据集
        oss_dataset_path = oss_config.get('dataset_path', 'dataset.npz')
        success = self.download_dataset_from_oss(
            oss_config, oss_dataset_path, target_dataset_path
        )
        
        if success:
            return str(target_dataset_path)
        else:
            return None
    
    def download_with_progress(self, oss_config, oss_dataset_path, local_dataset_path):
        """
        带进度显示的下载
        """
        print(f"📥 开始带进度的OSS下载...")
        
        try:
            import oss2
            
            # 初始化OSS客户端
            auth = oss2.Auth(oss_config['access_key_id'], oss_config['access_key_secret'])
            bucket = oss2.Bucket(auth, oss_config['endpoint'], oss_config['bucket_name'])
            
            # 获取文件大小
            object_info = bucket.head_object(oss_dataset_path)
            total_size = object_info.content_length
            
            print(f"📊 开始下载 {total_size / (1024*1024):.1f}MB...")
            
            # 分块下载并显示进度
            chunk_size = 1024 * 1024  # 1MB
            downloaded = 0
            start_time = time.time()
            
            with open(local_dataset_path, 'wb') as local_file:
                object_stream = bucket.get_object(oss_dataset_path)
                
                for chunk in object_stream:
                    local_file.write(chunk)
                    downloaded += len(chunk)
                    
                    # 每5MB显示一次进度
                    if downloaded % (5 * 1024 * 1024) == 0 or downloaded == total_size:
                        progress = downloaded / total_size * 100
                        elapsed = time.time() - start_time
                        speed = downloaded / elapsed / (1024*1024) if elapsed > 0 else 0
                        
                        print(f"   进度: {progress:.1f}% "
                              f"({downloaded / (1024*1024):.1f}MB/"
                              f"{total_size / (1024*1024):.1f}MB) "
                              f"速度: {speed:.1f}MB/s")
            
            total_time = time.time() - start_time
            avg_speed = total_size / total_time / (1024*1024)
            
            print(f"✅ 下载完成! 平均速度: {avg_speed:.1f}MB/s")
            return True
            
        except Exception as e:
            print(f"❌ 带进度下载失败: {e}")
            return False


def auto_setup_dataset(dataset_arg=None):
    """
    自动设置数据集 - 集成到训练脚本使用
    
    Args:
        dataset_arg: 用户指定的数据集路径
    
    Returns:
        str: 实际可用的数据集路径
    """
    downloader = OSSDatasetDownloader()
    
    # 如果用户指定了数据集路径
    if dataset_arg:
        if os.path.exists(dataset_arg):
            print(f"✅ 使用指定的数据集: {dataset_arg}")
            return dataset_arg
        else:
            print(f"⚠️  指定的数据集不存在: {dataset_arg}")
            print(f"将尝试从OSS下载到该位置...")
            # 尝试下载到指定位置
            # 这里可以添加下载逻辑
    
    # 检查并下载默认数据集
    dataset_path = downloader.check_and_download_dataset()
    
    if dataset_path is None:
        print(f"❌ 无法获取数据集！")
        print(f"💡 解决方案:")
        print(f"   1. 配置OSS信息: python EMSC_Net/EMSC_oss_config.py")
        print(f"   2. 手动下载数据集到本地")
        print(f"   3. 检查OSS权限配置")
        raise ValueError("数据集获取失败")
    
    return dataset_path


if __name__ == "__main__":
    # 测试下载器
    downloader = OSSDatasetDownloader()
    
    print(f"🧪 OSS下载器测试")
    
    # 创建配置文件
    from EMSC_oss_config import create_oss_config_file
    create_oss_config_file()
    
    print(f"\n💡 使用步骤:")
    print(f"1. 编辑 oss_config.json 配置文件")
    print(f"2. 运行训练时自动下载: python EMSC_Net/train.py")
    print(f"3. 或手动下载: python EMSC_Net/EMSC_oss_downloader.py")
    
    # 尝试自动下载（如果配置了的话）
    try:
        dataset_path = auto_setup_dataset()
        print(f"✅ 数据集就绪: {dataset_path}")
    except:
        print(f"⚠️  需要先配置OSS信息") 