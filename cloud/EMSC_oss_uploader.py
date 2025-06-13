"""
EMSC训练结果OSS上传模块
在云环境下将训练结果压缩并上传到OSS
"""

import os
import json
import zipfile
import shutil
import time
from datetime import datetime
from pathlib import Path
import oss2
from oss2.exceptions import OssError
from .EMSC_oss_config import load_oss_config

class EMSCOSSUploader:
    """EMSC训练结果OSS上传器"""
    
    def __init__(self, oss_config_path=None):
        """
        初始化OSS上传器
        
        Args:
            oss_config_path: OSS配置文件路径，如果为None则自动查找
        """
        self.oss_config_path = oss_config_path or self._find_oss_config()
        self.bucket = None
        self.config = None
        
        if self.oss_config_path and os.path.exists(self.oss_config_path):
            self._load_oss_config()
        else:
            print(f"⚠️  OSS配置文件未找到: {self.oss_config_path}")
    
    def _find_oss_config(self):
        """自动查找OSS配置文件"""
        possible_paths = [
            '/mnt/data/msc_models/dataset_EMSC_big/oss_config.json',
            './oss_config.json',
            '../oss_config.json',
            os.path.expanduser('~/oss_config.json')
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def _load_oss_config(self):
        """加载OSS配置 - 复用现有的OSS下载配置"""
        try:
            # 使用现有的OSS配置加载函数
            self.config = load_oss_config(self.oss_config_path)
            
            # 初始化OSS客户端
            auth = oss2.Auth(
                self.config['access_key_id'], 
                self.config['access_key_secret']
            )
            
            self.bucket = oss2.Bucket(
                auth, 
                self.config['endpoint'], 
                self.config['bucket_name']
            )
            
            print(f"✅ OSS配置加载成功 (复用现有配置)")
            print(f"   Bucket: {self.config['bucket_name']}")
            print(f"   Endpoint: {self.config['endpoint']}")
            
        except Exception as e:
            print(f"❌ OSS配置加载失败: {e}")
            self.bucket = None
            self.config = None
    
    def create_training_archive(self, source_dir, output_filename=None, include_patterns=None, exclude_patterns=None):
        """
        创建训练结果压缩包
        
        Args:
            source_dir: 源目录路径
            output_filename: 输出文件名，如果为None则自动生成
            include_patterns: 包含的文件模式列表
            exclude_patterns: 排除的文件模式列表
        
        Returns:
            str: 压缩包路径
        """
        if not os.path.exists(source_dir):
            raise ValueError(f"源目录不存在: {source_dir}")
        
        # 自动生成文件名
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dir_name = os.path.basename(source_dir.rstrip('/'))
            output_filename = f"{dir_name}_{timestamp}.zip"
        
        # 确保输出目录存在
        output_dir = os.path.dirname(output_filename) or '.'
        os.makedirs(output_dir, exist_ok=True)
        
        # 默认包含和排除模式
        if include_patterns is None:
            include_patterns = [
                '*.h5',           # 模型文件
                '*.keras',        # Keras模型文件
                '*.json',         # 配置文件
                '*.png',          # 图表文件
                '*.jpg',          # 图表文件
                '*.txt',          # 日志文件
                '*.csv',          # 训练历史
                '*.md',           # 文档文件
                'saved_model/*',  # SavedModel格式
                'best_*',         # 最佳模型
                'training_*',     # 训练相关文件
                'history_*',      # 历史文件
                'checkpoint*'     # 检查点文件
            ]
        
        if exclude_patterns is None:
            exclude_patterns = [
                '*.pyc',          # Python缓存文件
                '__pycache__/*',  # Python缓存目录
                '*.tmp',          # 临时文件
                '*.log',          # 大型日志文件
                '.DS_Store',      # macOS系统文件
                'Thumbs.db'       # Windows系统文件
            ]
        
        print(f"📦 创建训练结果压缩包...")
        print(f"   源目录: {source_dir}")
        print(f"   输出文件: {output_filename}")
        
        total_files = 0
        total_size = 0
        
        try:
            with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
                for root, dirs, files in os.walk(source_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        relative_path = os.path.relpath(file_path, source_dir)
                        
                        # 检查是否应该包含此文件
                        should_include = self._should_include_file(relative_path, include_patterns, exclude_patterns)
                        
                        if should_include:
                            try:
                                zipf.write(file_path, relative_path)
                                file_size = os.path.getsize(file_path)
                                total_files += 1
                                total_size += file_size
                                
                                if total_files % 10 == 0:
                                    print(f"   已处理 {total_files} 个文件...")
                                    
                            except Exception as e:
                                print(f"   ⚠️  跳过文件 {relative_path}: {e}")
            
            final_size = os.path.getsize(output_filename)
            compression_ratio = (1 - final_size / total_size) * 100 if total_size > 0 else 0
            
            print(f"✅ 压缩包创建完成:")
            print(f"   包含文件: {total_files}")
            print(f"   原始大小: {total_size / (1024*1024):.1f}MB")
            print(f"   压缩大小: {final_size / (1024*1024):.1f}MB")
            print(f"   压缩率: {compression_ratio:.1f}%")
            
            return output_filename
            
        except Exception as e:
            print(f"❌ 创建压缩包失败: {e}")
            if os.path.exists(output_filename):
                os.remove(output_filename)
            raise
    
    def _should_include_file(self, file_path, include_patterns, exclude_patterns):
        """检查文件是否应该被包含"""
        import fnmatch
        
        # 首先检查排除模式
        for pattern in exclude_patterns:
            if fnmatch.fnmatch(file_path, pattern) or fnmatch.fnmatch(os.path.basename(file_path), pattern):
                return False
        
        # 然后检查包含模式
        for pattern in include_patterns:
            if fnmatch.fnmatch(file_path, pattern) or fnmatch.fnmatch(os.path.basename(file_path), pattern):
                return True
        
        return False
    
    def upload_to_oss(self, local_file, oss_path=None, progress_callback=None):
        """
        上传文件到OSS
        
        Args:
            local_file: 本地文件路径
            oss_path: OSS路径，如果为None则自动生成
            progress_callback: 进度回调函数
        
        Returns:
            str: OSS文件路径
        """
        if not self.bucket:
            raise ValueError("OSS未配置或配置无效")
        
        if not os.path.exists(local_file):
            raise ValueError(f"本地文件不存在: {local_file}")
        
        # 自动生成OSS路径
        if oss_path is None:
            filename = os.path.basename(local_file)
            timestamp = datetime.now().strftime("%Y/%m/%d")
            oss_path = f"emsc_training_results/{timestamp}/{filename}"
        
        file_size = os.path.getsize(local_file)
        print(f"📤 上传文件到OSS...")
        print(f"   本地文件: {local_file}")
        print(f"   OSS路径: {oss_path}")
        print(f"   文件大小: {file_size / (1024*1024):.1f}MB")
        
        try:
            start_time = time.time()
            
            # 使用分片上传处理大文件
            if file_size > 100 * 1024 * 1024:  # 100MB以上使用分片上传
                print("   使用分片上传...")
                self._multipart_upload(local_file, oss_path, progress_callback)
            else:
                print("   使用普通上传...")
                with open(local_file, 'rb') as fileobj:
                    self.bucket.put_object(oss_path, fileobj)
            
            upload_time = time.time() - start_time
            upload_speed = file_size / upload_time / (1024*1024)  # MB/s
            
            print(f"✅ 上传完成:")
            print(f"   耗时: {upload_time:.1f}秒")
            print(f"   速度: {upload_speed:.1f}MB/s")
            print(f"   OSS URL: https://{self.config['bucket_name']}.{self.config['endpoint'].replace('https://', '')}/{oss_path}")
            
            return oss_path
            
        except OssError as e:
            print(f"❌ OSS上传失败: {e}")
            raise
        except Exception as e:
            print(f"❌ 上传失败: {e}")
            raise
    
    def _multipart_upload(self, local_file, oss_path, progress_callback=None):
        """分片上传大文件"""
        try:
            # 初始化分片上传
            upload_id = self.bucket.init_multipart_upload(oss_path).upload_id
            
            parts = []
            part_size = 10 * 1024 * 1024  # 10MB per part
            file_size = os.path.getsize(local_file)
            part_count = (file_size + part_size - 1) // part_size
            
            print(f"   分片数量: {part_count}")
            
            with open(local_file, 'rb') as fileobj:
                for part_number in range(1, part_count + 1):
                    data = fileobj.read(part_size)
                    if not data:
                        break
                    
                    result = self.bucket.upload_part(oss_path, upload_id, part_number, data)
                    parts.append(oss2.models.PartInfo(part_number, result.etag))
                    
                    if progress_callback:
                        progress = part_number / part_count * 100
                        progress_callback(progress)
                    
                    if part_number % 10 == 0:
                        print(f"   已上传 {part_number}/{part_count} 分片...")
            
            # 完成分片上传
            self.bucket.complete_multipart_upload(oss_path, upload_id, parts)
            
        except Exception as e:
            # 如果失败，取消分片上传
            try:
                self.bucket.abort_multipart_upload(oss_path, upload_id)
            except:
                pass
            raise
    
    def upload_training_results(self, training_dir, cleanup_local=True):
        """
        完整的训练结果上传流程
        
        Args:
            training_dir: 训练结果目录
            cleanup_local: 是否清理本地压缩包
        
        Returns:
            dict: 上传结果信息
        """
        if not self.bucket:
            print("⚠️  OSS未配置，跳过上传")
            return None
        
        try:
            # 1. 创建压缩包
            archive_path = self.create_training_archive(training_dir)
            
            # 2. 上传到OSS
            oss_path = self.upload_to_oss(archive_path)
            
            # 3. 清理本地压缩包（可选）
            if cleanup_local:
                os.remove(archive_path)
                print(f"🗑️  已清理本地压缩包: {archive_path}")
            
            # 4. 返回结果信息
            result = {
                'success': True,
                'training_dir': training_dir,
                'archive_path': archive_path if not cleanup_local else None,
                'oss_path': oss_path,
                'upload_time': datetime.now().isoformat(),
                'oss_url': f"https://{self.config['bucket_name']}.{self.config['endpoint'].replace('https://', '')}/{oss_path}"
            }
            
            print(f"🎉 训练结果上传完成!")
            print(f"   OSS路径: {oss_path}")
            
            return result
            
        except Exception as e:
            print(f"❌ 训练结果上传失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'training_dir': training_dir
            }

def check_oss_config_exists():
    """检查现有OSS配置文件"""
    possible_paths = [
        '/mnt/data/msc_models/dataset_EMSC_big/oss_config.json',
        './oss_config.json',
        '../oss_config.json',
        os.path.expanduser('~/oss_config.json')
    ]
    
    existing_configs = []
    for path in possible_paths:
        if os.path.exists(path):
            existing_configs.append(path)
    
    return existing_configs

# 使用示例
if __name__ == "__main__":
    # 检查现有配置
    existing_configs = check_oss_config_exists()
    if existing_configs:
        print(f"发现现有OSS配置: {existing_configs[0]}")
        
        # 创建上传器
        uploader = EMSCOSSUploader(existing_configs[0])
        
        # 上传训练结果
        training_dir = "./network_6-32-32-8-1"
        result = uploader.upload_training_results(training_dir)
        
        if result and result['success']:
            print(f"✅ 上传成功: {result['oss_url']}")
        else:
            print("❌ 上传失败")
    else:
        print("❌ 未找到OSS配置文件，请先配置OSS下载功能") 