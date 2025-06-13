 # EMSC 项目结构说明

本文档描述了EMSC (Enhanced Minimal State Cell) 项目的文件夹结构和组织方式。

## 📁 项目结构

```
EMSC_Net/
├── core/                    # 🔧 核心组件
│   ├── EMSC_model.py       # 模型定义和构建
│   ├── EMSC_data.py        # 数据加载和处理
│   ├── EMSC_losses.py      # 损失函数定义
│   └── __init__.py
│
├── training/               # 🚀 训练相关
│   ├── EMSC_train.py       # 主训练脚本
│   ├── EMSC_callbacks.py   # 训练回调函数
│   ├── EMSC_dynamic_batch.py # 动态批次调整
│   ├── EMSC_cpu_monitor.py # CPU监控
│   ├── EMSC_staged_training.py # 分阶段训练
│   └── __init__.py
│
├── prediction/             # 🔮 预测相关
│   ├── EMSC_predict.py     # 基础预测功能
│   ├── EMSC_predict_auto.py # 自动预测
│   └── __init__.py
│
├── cloud/                  # ☁️ 云服务集成
│   ├── EMSC_cloud_io_optimizer.py # 云I/O优化
│   ├── EMSC_oss_config.py  # OSS配置
│   ├── EMSC_oss_downloader.py # OSS下载
│   ├── EMSC_oss_uploader.py # OSS上传
│   ├── setup_oss_upload.py # OSS设置脚本
│   ├── upload_results_to_oss.py # 手动上传工具
│   └── __init__.py
│
├── utils/                  # 🛠️ 工具和辅助功能
│   ├── EMSC_config.py      # 配置管理
│   ├── EMSC_dataset_generator.py # 数据集生成
│   ├── EMSC_performance.py # 性能分析
│   ├── EMSC_utils.py       # 通用工具
│   ├── EMSC_window_sampler.py # 窗口采样
│   └── __init__.py
│
├── tests/                  # 🧪 测试和调试
│   ├── EMSC_cpu_stress_test.py # CPU压力测试
│   ├── EMSC_gpu_debug.py   # GPU调试
│   ├── EMSC_gpu_verify.py  # GPU验证
│   ├── diagnose_gpu_hang.py # GPU挂起诊断
│   ├── gpu_hang_diagnosis.py # GPU挂起分析
│   ├── test_gpu_warmup.py  # GPU预热测试
│   └── __init__.py
│
├── scripts/                # 📜 独立脚本和实验
│   ├── compare_networks.py # 网络比较
│   ├── run_experiments.py  # 实验运行
│   ├── train.py           # 简单训练脚本
│   ├── normalization_examples.py # 标准化示例
│   └── __init__.py
│
├── docs/                   # 📚 文档
│   ├── GPU_OPTIMIZATION_LOCAL_CLOUD.md # GPU优化指南
│   ├── OSS_INTEGRATION_SUMMARY.md # OSS集成总结
│   ├── OSS_UPLOAD_GUIDE.md # OSS上传指南
│   ├── device_usage_examples.md # 设备使用示例
│   └── PROJECT_STRUCTURE.md # 本文档
│
├── __init__.py             # 主包初始化
└── training_config.json    # 训练配置文件
```

## 🎯 模块功能说明

### Core 核心模块
- **EMSC_model.py**: 定义EMSC模型架构，包含状态细胞和循环结构
- **EMSC_data.py**: 数据加载、预处理和TensorFlow数据集创建
- **EMSC_losses.py**: 自定义损失函数，针对EMSC模型优化

### Training 训练模块
- **EMSC_train.py**: 主训练脚本，支持GPU/CPU、本地/云环境
- **EMSC_callbacks.py**: 训练回调，包括进度监控、模型保存等
- **EMSC_dynamic_batch.py**: 动态批次大小调整，优化资源利用
- **EMSC_cpu_monitor.py**: CPU使用率监控
- **EMSC_staged_training.py**: 分阶段训练策略

### Prediction 预测模块
- **EMSC_predict.py**: 基础预测功能，支持单序列和批量预测
- **EMSC_predict_auto.py**: 自动预测，智能模型选择和参数优化

### Cloud 云服务模块
- **EMSC_cloud_io_optimizer.py**: 云环境I/O优化
- **EMSC_oss_*.py**: 阿里云OSS集成，支持数据上传下载
- **setup_oss_upload.py**: OSS环境设置
- **upload_results_to_oss.py**: 手动上传工具

### Utils 工具模块
- **EMSC_config.py**: 配置文件管理和参数解析
- **EMSC_dataset_generator.py**: 数据集生成和预处理
- **EMSC_performance.py**: 性能分析和基准测试
- **EMSC_utils.py**: 通用工具函数
- **EMSC_window_sampler.py**: 时间窗口采样

### Tests 测试模块
- **GPU相关测试**: GPU调试、验证、预热测试
- **CPU相关测试**: CPU压力测试和性能分析
- **诊断工具**: 训练挂起问题诊断

### Scripts 脚本模块
- **实验脚本**: 网络比较、实验运行
- **工具脚本**: 简化的训练和测试脚本
- **示例代码**: 使用示例和最佳实践

## 🚀 快速开始

### 基础使用
```python
from EMSC_Net import build_msc_model, train_main, create_training_config

# 创建模型
model = build_msc_model(state_dim=8, hidden_dim=64)

# 开始训练
train_main()
```

### 高级使用
```python
from EMSC_Net.training import EMSC_train
from EMSC_Net.cloud import EMSCOSSUploader
from EMSC_Net.utils import EMSC_config

# 自定义训练配置
config = EMSC_config.create_training_config(
    state_dim=16,
    hidden_dim=128,
    learning_rate=0.001
)

# 云环境训练
EMSC_train.main()

# 上传结果到OSS
if EMSC_Net.OSS_AVAILABLE:
    uploader = EMSCOSSUploader()
    uploader.upload_training_results("./network_6-64-64-8-1")
```

## 📋 导入路径更新

由于文件重新组织，需要更新导入路径：

### 旧导入方式
```python
from EMSC_model import build_msc_model
from EMSC_train import main
```

### 新导入方式
```python
from EMSC_Net.core.EMSC_model import build_msc_model
from EMSC_Net.training.EMSC_train import main

# 或使用便捷导入
from EMSC_Net import build_msc_model, train_main
```

## 🔧 开发指南

1. **添加新功能**: 根据功能类型放入相应文件夹
2. **修改导入**: 更新相关文件的导入路径
3. **测试**: 在tests文件夹中添加相应测试
4. **文档**: 更新docs文件夹中的相关文档

## 📝 注意事项

- 所有Python包文件夹都包含`__init__.py`文件
- 主包提供便捷的导入接口
- 云服务功能为可选依赖，需要安装oss2
- 建议使用相对导入来保持模块间的依赖关系