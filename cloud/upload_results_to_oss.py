 #!/usr/bin/env python3
"""独立的训练结果OSS上传工具"""

import os
import sys
import argparse
from .EMSC_oss_uploader import EMSCOSSUploader

def main():
    parser = argparse.ArgumentParser(description='上传EMSC训练结果到OSS')
    parser.add_argument('training_dir', help='训练结果目录路径')
    parser.add_argument('--oss-config', help='OSS配置文件路径 (可选，会自动查找现有配置)')
    parser.add_argument('--keep-local', action='store_true', help='保留本地压缩包')
    
    args = parser.parse_args()
    
    # 检查训练目录
    if not os.path.exists(args.training_dir):
        print(f"❌ 训练目录不存在: {args.training_dir}")
        sys.exit(1)
    
    print(f"🚀 开始上传训练结果")
    print(f"训练目录: {args.training_dir}")
    
    try:
        # 如果没有指定配置文件，自动查找现有配置
        config_path = args.oss_config
        if not config_path:
            from .EMSC_oss_uploader import check_oss_config_exists
            existing_configs = check_oss_config_exists()
            if existing_configs:
                config_path = existing_configs[0]
                print(f"📁 使用现有OSS配置: {config_path}")
            else:
                print("❌ 未找到OSS配置文件")
                print("请先配置OSS下载功能，或使用 --oss-config 指定配置文件")
                sys.exit(1)
        
        # 创建上传器
        uploader = EMSCOSSUploader(oss_config_path=config_path)
        
        if not uploader.bucket:
            print("❌ OSS配置无效，请检查配置文件")
            print("运行 python setup_oss_upload.py 检查配置")
            sys.exit(1)
        
        # 执行上传
        upload_result = uploader.upload_training_results(
            training_dir=args.training_dir,
            cleanup_local=not args.keep_local
        )
        
        if upload_result and upload_result['success']:
            print(f"\n🎉 上传成功!")
            print(f"OSS路径: {upload_result['oss_path']}")
            print(f"访问URL: {upload_result['oss_url']}")
            
            # 保存上传信息
            import json
            upload_info_path = os.path.join(args.training_dir, 'oss_upload_info.json')
            with open(upload_info_path, 'w', encoding='utf-8') as f:
                json.dump(upload_result, f, indent=2, ensure_ascii=False)
            print(f"上传信息已保存: {upload_info_path}")
            
        else:
            print("❌ 上传失败")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ 上传过程中出错: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()