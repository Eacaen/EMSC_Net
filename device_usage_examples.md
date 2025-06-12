# EMSC设备选择使用指南

## 🎯 设备参数说明

新增的 `--device` 参数允许用户指定训练时使用的设备类型：

- `auto` (默认): 自动选择，GPU优先，如果没有GPU则回退到CPU
- `gpu`: 强制使用GPU，如果没有GPU则报错退出
- `cpu`: 强制使用CPU，即使有GPU也不使用

## 📝 使用示例

### 1. 自动设备选择 (默认行为)
```bash
# 不指定设备参数，默认auto模式
python train.py --epochs 200 --state_dim 8 --hidden_dim 32

# 显式指定auto模式
python train.py --device auto --epochs 200 --state_dim 8 --hidden_dim 32
```

### 2. 强制使用GPU
```bash
# 强制使用GPU，如果没有GPU会报错
python train.py --device gpu --epochs 200 --state_dim 8 --hidden_dim 32
```

### 3. 强制使用CPU
```bash
# 强制使用CPU，即使有GPU也不使用
python train.py --device cpu --epochs 200 --state_dim 8 --hidden_dim 32
```

## 🚀 批量实验示例

### 使用实验管理脚本

```bash
# 自动设备选择运行标准配置
python run_experiments.py --run standard --device auto

# 强制GPU运行大型网络
python run_experiments.py --run xlarge --device gpu --epochs 500

# 强制CPU运行所有配置
python run_experiments.py --run_all --device cpu --epochs 100

# 在不同设备上对比性能
python run_experiments.py --run standard --device gpu --epochs 200
python run_experiments.py --run standard --device cpu --epochs 200
```

## 🔍 使用场景

### 何时使用 `--device gpu`
- 确保在GPU上训练，避免意外使用CPU
- 在多GPU环境中明确指定使用GPU
- 性能基准测试时需要确保设备一致性

### 何时使用 `--device cpu`
- GPU内存不足，强制使用CPU训练
- 调试模型，CPU更容易定位问题
- 对比GPU和CPU的数值一致性
- 在没有GPU的机器上确保正常运行

### 何时使用 `--device auto` (默认)
- 一般使用场景，让系统自动选择最优设备
- 跨环境脚本，适应不同的硬件配置
- 开发阶段，不需要特别指定设备

## ⚙️ 配置说明

### GPU环境配置
- ✅ 禁用XLA JIT编译 (EMSC while_loop兼容性)
- ✅ 禁用TensorFloat-32 (精度优先)
- ✅ 启用GPU内存增长
- ✅ 梯度裁剪: clipnorm=1.0, clipvalue=0.5
- ✅ 数值稳定性: EMSCLoss保护

### CPU环境配置
- ✅ 多线程优化 (使用所有CPU核心)
- ✅ OpenMP并行化
- ✅ OneDNN优化
- ✅ 动态批次大小调整 (可选)
- ✅ CPU使用率监控 (可选)

## 📊 性能对比

可以使用相同的网络结构在不同设备上训练，然后对比性能：

```bash
# GPU训练
python train.py --device gpu --state_dim 8 --hidden_dim 32 --epochs 200

# CPU训练  
python train.py --device cpu --state_dim 8 --hidden_dim 32 --epochs 200

# 对比结果
python compare_networks.py
```

结果会保存在不同的文件夹中：
- `models/dataset_EMSC_big/network_6-32-32-8-1/` (GPU或auto模式)
- 设备信息会在训练日志中显示

## 🛠️ 故障排除

### GPU强制模式报错
```
RuntimeError: 用户指定使用GPU，但未检测到任何GPU设备！
```
**解决方案**: 
- 检查GPU驱动和CUDA安装
- 使用 `--device auto` 或 `--device cpu`
- 运行 `python EMSC_gpu_verify.py` 检查GPU环境

### CPU强制模式在GPU机器上
即使有GPU，使用 `--device cpu` 也会强制使用CPU，这是正常行为。

## 💡 最佳实践

1. **开发阶段**: 使用 `--device auto`，让系统自动选择
2. **生产训练**: 使用 `--device gpu` 确保使用GPU
3. **调试问题**: 使用 `--device cpu` 获得更好的错误信息
4. **性能测试**: 分别使用不同设备训练相同配置进行对比
5. **批量实验**: 在实验脚本中指定设备保证一致性 