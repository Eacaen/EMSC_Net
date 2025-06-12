"""
EMSC模型主训练脚本
使用模块化结构组织训练流程，支持多CPU训练
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import mixed_precision

# 导入自定义模块
from EMSC_model import build_msc_model
from EMSC_data import EMSCDataGenerator, create_tf_dataset, load_dataset_from_npz
from EMSC_callbacks import MSCProgressCallback, create_early_stopping_callback, create_learning_rate_scheduler
from EMSC_cpu_monitor import create_cpu_monitor_callback
from EMSC_dynamic_batch import DynamicBatchTrainer, create_dynamic_batch_callback
try:
    from EMSC_cloud_io_optimizer import CloudIOOptimizer, create_cloud_optimized_training_config
    CLOUD_OPTIMIZER_AVAILABLE = True
except ImportError:
    CLOUD_OPTIMIZER_AVAILABLE = False

try:
    from EMSC_oss_downloader import download_dataset
    OSS_DOWNLOADER_AVAILABLE = True
except ImportError:
    OSS_DOWNLOADER_AVAILABLE = False
from EMSC_config import (create_training_config, save_training_config, 
                        parse_training_args, get_dataset_paths)
from EMSC_utils import (load_or_create_model_with_history, 
                       resume_training_from_checkpoint,
                       plot_final_training_summary,
                       print_training_summary)
from EMSC_losses import EMSCLoss

def check_environment(device_preference='auto'):
    """
    检查并配置训练环境
    
    Args:
        device_preference: 设备偏好 ('auto', 'gpu', 'cpu')
                          - 'auto': 自动选择，GPU优先，回退到CPU
                          - 'gpu': 强制使用GPU，如果不可用则报错
                          - 'cpu': 强制使用CPU
    """
    print("检查训练环境...")
    print(f"TensorFlow版本: {tf.__version__}")
    print(f"当前工作目录: {os.getcwd()}")
    print(f"设备偏好: {device_preference}")
    
    # 检查GPU可用性
    gpus = tf.config.list_physical_devices('GPU')
    print(f"检测到的GPU设备: {len(gpus)}")
    
    # 根据用户偏好决定使用的设备
    if device_preference == 'cpu':
        print("🖥️  用户指定使用CPU，跳过GPU配置")
        # 强制禁用GPU，即使有GPU也不使用
        if gpus:
            tf.config.set_visible_devices([], 'GPU')
            print("✅ 已禁用所有GPU设备，强制使用CPU")
        return setup_cpu_environment()
    
    elif device_preference == 'gpu':
        print("🎮 用户指定强制使用GPU")
        if not gpus:
            raise RuntimeError("❌ 用户指定使用GPU，但未检测到任何GPU设备！")
        return setup_gpu_environment(gpus)
    
    else:  # device_preference == 'auto'
        print("🔄 自动设备选择模式 (GPU优先)")
        if gpus:
            return setup_gpu_environment(gpus)
        else:
            print("未发现GPU设备，自动切换到CPU训练")
            return setup_cpu_environment()

def detect_environment():
    """
    检测运行环境：本地 vs 云环境
    
    Returns:
        str: 'local' 或 'cloud'
    """
    # 检测云环境的特征
    cloud_indicators = [
        '/mnt/data',  # 阿里云挂载路径
        '/opt/ml',    # AWS SageMaker
        '/kaggle',    # Kaggle
    ]
    
    # 检查环境变量
    env_cloud_indicators = [
        'KUBERNETES_SERVICE_HOST',  # K8s环境
        'CLOUD_SHELL',             # 云shell
        'COLAB_GPU',               # Google Colab
    ]
    
    # 检查路径指标
    for indicator in cloud_indicators:
        if os.path.exists(indicator):
            return 'cloud'
    
    # 检查OSS配置文件
    if os.path.exists('/mnt/data/msc_models/dataset_EMSC_big/oss_config.json'):
        return 'cloud'
    
    # 检查环境变量
    for env_var in env_cloud_indicators:
        if os.environ.get(env_var):
            return 'cloud'
    
    # 检查CPU核心数（云环境通常有更多核心）
    cpu_count = os.cpu_count() or 4
    if cpu_count >= 32:  # 高核心数可能是云环境
        print(f"🔍 检测到高核心数CPU ({cpu_count})，可能是云环境")
    
    return 'local'

def setup_gpu_environment(gpus):
    """设置GPU训练环境 - 区分本地和云环境"""
    env_type = detect_environment()
    print(f"🎮 配置GPU训练环境 ({env_type})")
    print(f"发现 {len(gpus)} 个GPU设备:")
    
    for gpu in gpus:
        print(f"- {gpu}")
        # 配置GPU内存增长
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"已为 {gpu} 启用内存增长")
        except RuntimeError as e:
            print(f"配置GPU内存增长时出错: {e}")
    
    # 设置GPU为默认设备
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        
        if env_type == 'local':
            # 本地GPU优化配置
            print("🏠 本地GPU环境优化:")
            
            # 本地环境启用TF32以提升性能
            tf.config.experimental.enable_tensor_float_32_execution(True)
            print("✅ 启用TensorFloat-32（本地GPU性能优化）")
            
            # 本地环境的线程配置
            tf.config.threading.set_inter_op_parallelism_threads(0)  # 使用默认
            tf.config.threading.set_intra_op_parallelism_threads(0)  # 使用默认
            print("✅ 使用默认线程配置（本地GPU优化）")
            
        else:
            # 云GPU优化配置
            print("☁️  云GPU环境优化:")
            
            # 云环境禁用TF32以提高精度稳定性
            tf.config.experimental.enable_tensor_float_32_execution(False)
            print("✅ 禁用TensorFloat-32（云环境精度优先）")
            
            # 云环境的线程配置（更保守）
            tf.config.threading.set_inter_op_parallelism_threads(4)
            tf.config.threading.set_intra_op_parallelism_threads(8)
            print("✅ 配置保守线程设置（云环境稳定性优先）")
        
        # 通用GPU配置
        tf.config.optimizer.set_jit(False)
        print("ℹ️  禁用XLA JIT编译（EMSC while_loop兼容性）")
        
        print(f"✅ GPU环境配置完成: {gpus[0]}")
        return None  # 使用GPU时不需要返回worker数
    except RuntimeError as e:
        raise RuntimeError(f"❌ GPU设备配置失败: {e}")

def setup_cpu_environment():
    """设置CPU训练环境"""
    print("🖥️  配置CPU训练环境")
    
    # 获取CPU核心数
    cpu_count = os.cpu_count()
    if cpu_count is None:
        cpu_count = 4  # 默认值
    
    # 针对阿里云等云环境的CPU优化配置
    # 使用所有可用CPU核心，不保留
    num_workers = cpu_count
    
    # 设置TensorFlow线程配置 - 更激进的设置
    tf.config.threading.set_inter_op_parallelism_threads(num_workers)
    tf.config.threading.set_intra_op_parallelism_threads(num_workers)
    
    # 设置OpenMP线程数（用于NumPy、MKL等库）
    os.environ['OMP_NUM_THREADS'] = str(num_workers)
    os.environ['MKL_NUM_THREADS'] = str(num_workers)
    os.environ['OPENBLAS_NUM_THREADS'] = str(num_workers)
    os.environ['NUMEXPR_NUM_THREADS'] = str(num_workers)
    
    # 优化TensorFlow的CPU性能
    os.environ['TF_NUM_INTEROP_THREADS'] = str(num_workers)
    os.environ['TF_NUM_INTRAOP_THREADS'] = str(num_workers)
    
    # 启用所有CPU优化
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'  # 启用OneDNN优化
    
    print(f"阿里云CPU环境优化配置完成:")
    print(f"- 总CPU核心数: {cpu_count}")
    print(f"- 训练使用线程数: {num_workers}")
    print(f"- inter_op_parallelism_threads: {num_workers}")
    print(f"- intra_op_parallelism_threads: {num_workers}")
    print(f"- OMP_NUM_THREADS: {num_workers}")
    print(f"- 已启用OneDNN优化")
    
    # 检查内存使用情况
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"\n系统内存信息:")
        print(f"总内存: {memory.total / (1024**3):.1f} GB")
        print(f"可用内存: {memory.available / (1024**3):.1f} GB")
        print(f"内存使用率: {memory.percent}%")
        
        # 显示CPU信息
        print(f"\nCPU信息:")
        print(f"物理CPU核心数: {psutil.cpu_count(logical=False)}")
        print(f"逻辑CPU核心数: {psutil.cpu_count(logical=True)}")
    except ImportError:
        print("未安装psutil，跳过系统信息检查")
    
    return num_workers

def get_optimal_batch_size(num_samples, num_workers):
    """
    计算最优批处理大小 - 区分本地和云环境
    
    Args:
        num_samples: 训练样本数量
        num_workers: 工作线程数（CPU模式）或None（GPU模式）
    
    Returns:
        int: 最优批处理大小
    """
    if num_workers is None:  # GPU模式
        env_type = detect_environment()
        
        if env_type == 'local':
            # 本地GPU环境 - 使用中等批处理大小
            base_batch = min(64, max(16, num_samples // 100))  # 更保守的基础大小
            batch_size = base_batch
            print(f"本地GPU批次大小: {batch_size}")
        else:
            # 云GPU环境 - 使用较大的批处理大小
            batch_size = min(128, num_samples // 50)
            print(f"云GPU批次大小: {batch_size}")
        
        return (batch_size // 8) * 8  # 确保是8的倍数
    
    # CPU模式下的批处理大小计算 - 更积极的配置
    # 基础批次大小 - 为CPU训练增加更大的基数
    base_batch = min(64, max(32, num_samples // 50))  # 增加基础批次大小
    
    # 根据CPU线程数调整 - 让每个线程处理更多数据
    # 使用更激进的倍数，充分利用多核CPU
    if num_workers >= 16:  # 高核心数CPU（阿里云高配）
        multiplier = 2
    elif num_workers >= 8:  # 中等核心数CPU
        multiplier = 3
    else:  # 低核心数CPU
        multiplier = 4
    
    optimal_batch = base_batch * multiplier
    
    # 确保批次大小合理
    optimal_batch = min(optimal_batch, num_samples)  # 不超过样本总数
    optimal_batch = max(16, optimal_batch)  # 最小16
    
    # 确保是8的倍数（对内存对齐和向量化有利）
    optimal_batch = (optimal_batch // 8) * 8
    
    print(f"CPU批次大小计算: 基础={base_batch}, 线程数={num_workers}, 倍数={multiplier}, 最终={optimal_batch}")
    
    return optimal_batch

def warmup_gpu_model(model, sample_batch_size=1, max_sequence_length=100):
    """
    GPU模型预热，预编译tf.while_loop图
    解决第一次执行时的长时间卡顿问题
    """
    print("🔥 GPU模型预热中...")
    
    try:
        # 创建小规模的样本数据进行预热
        warmup_input = tf.random.normal((sample_batch_size, max_sequence_length, 6), dtype=tf.float32)
        warmup_init_state = tf.zeros((sample_batch_size, 8), dtype=tf.float32)
        
        # 第一次前向传播 - 编译while_loop图
        print("  🔄 执行第一次前向传播（图编译）...")
        start_time = tf.timestamp()
        
        _ = model([warmup_input, warmup_init_state], training=False)
        
        compile_time = tf.timestamp() - start_time
        print(f"  ✅ 图编译完成，耗时: {compile_time:.2f}秒")
        
        # 第二次前向传播 - 验证编译效果
        print("  🔄 执行第二次前向传播（验证加速）...")
        start_time = tf.timestamp()
        
        _ = model([warmup_input, warmup_init_state], training=False)
        
        exec_time = tf.timestamp() - start_time
        print(f"  ⚡ 执行时间: {exec_time:.2f}秒")
        
        print("🎉 GPU模型预热完成！训练将立即开始")
        return True
        
    except Exception as e:
        print(f"⚠️  GPU预热失败: {e}")
        print("继续训练，但第一个epoch可能较慢...")
        return False

def main(args=None):
    # 解析命令行参数（如果没有传入则解析）
    if args is None:
        args = parse_training_args()
    
    # 检查并配置环境
    num_workers = check_environment(device_preference=args.device)
    
    # 强制禁用混合精度，确保CPU和GPU数值一致性
    tf.keras.mixed_precision.set_global_policy('float32')
    tf.keras.backend.set_floatx('float32')
    print("强制使用float32精度训练（禁用混合精度）")
    
    # 云环境I/O优化
    cloud_optimizer = None
    if args.cloud_io_optimize:
        if CLOUD_OPTIMIZER_AVAILABLE:
            print("🌥️  启用阿里云I/O优化...")
            cloud_optimizer = CloudIOOptimizer(
                io_buffer_size=128,      # 更大的I/O缓冲
                prefetch_factor=16,      # 激进预取
                io_threads=min(32, num_workers * 2) if num_workers else 16,
                memory_cache_size=1024   # 1GB内存缓存
            )
            cloud_optimizer.optimize_cloud_environment()
        else:
            print("⚠️  云优化模块不可用，跳过优化")
    
    # 获取数据集路径
    paths = get_dataset_paths(args.dataset)
    base_dataset_dir = paths['dataset_dir']
    
    # 根据网络结构创建子文件夹
    network_structure = f"6-{args.hidden_dim}-{args.hidden_dim}-{args.state_dim}-1"
    dataset_dir = os.path.join(base_dataset_dir, f"network_{network_structure}")
    
    # 确保目录存在
    os.makedirs(dataset_dir, exist_ok=True)
    print(f"网络结构: {network_structure}")
    print(f"模型保存目录: {dataset_dir}")
    
    model_name = paths['model_name']
    best_model_name = paths['best_model_name']
    dataset_path = paths['dataset_path']
    
    # 创建和保存训练配置
    training_config = create_training_config(
        state_dim=args.state_dim,
        input_dim=6,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_frequency=args.save_frequency
    )
    save_training_config(training_config, dataset_dir)
    
    # 数据集优先级策略
    print(f"🔍 数据集优先级检查...")
    
    # 检查是否为云环境（通过OSS配置文件存在判断）
    oss_config_path = '/mnt/data/msc_models/dataset_EMSC_big/oss_config.json'
    is_cloud_environment = os.path.exists(oss_config_path)
    
    if is_cloud_environment:
        print(f"🌥️  检测到云环境，启用云数据集优先级策略")
        
        # 云环境优先级：当前目录 -> OSS下载 -> OSS内路径
        current_dir_dataset = os.path.join(os.getcwd(), os.path.basename(dataset_path))
        oss_internal_path = "/mnt/data/msc_models/dataset_EMSC_big/dataset_EMSC_big.npz"
        
        print(f"优先级1: 当前目录 - {current_dir_dataset}")
        print(f"优先级2: OSS下载到当前目录")
        print(f"优先级3: OSS内路径 - {oss_internal_path}")
        
        # 优先级1: 检查当前运行目录
        if os.path.exists(current_dir_dataset):
            file_size = os.path.getsize(current_dir_dataset)
            print(f"✅ 使用当前目录数据集: {current_dir_dataset}")
            print(f"   文件大小: {file_size / (1024*1024):.1f}MB")
            dataset_path = current_dir_dataset
        
        # 优先级2: 从OSS下载到当前目录
        elif OSS_DOWNLOADER_AVAILABLE:
            try:
                print(f"📥 当前目录无数据集，尝试从OSS下载...")
                print(f"OSS配置文件: {oss_config_path}")
                print(f"下载到: {current_dir_dataset}")
                
                downloaded_path = download_dataset(oss_config_path, current_dir_dataset)
                print(f"✅ 数据集下载完成: {downloaded_path}")
                dataset_path = downloaded_path
                
            except Exception as e:
                print(f"⚠️  OSS下载失败: {e}")
                
                # 优先级3: 使用OSS内路径
                if os.path.exists(oss_internal_path):
                    file_size = os.path.getsize(oss_internal_path)
                    print(f"✅ 使用OSS内路径数据集: {oss_internal_path}")
                    print(f"   文件大小: {file_size / (1024*1024):.1f}MB")
                    dataset_path = oss_internal_path
                else:
                    print(f"❌ OSS内路径也不存在: {oss_internal_path}")
                    raise ValueError(f"所有数据集路径都不可用")
        
        # 如果OSS下载器不可用，直接尝试OSS内路径
        else:
            if os.path.exists(oss_internal_path):
                file_size = os.path.getsize(oss_internal_path)
                print(f"✅ OSS下载器不可用，使用OSS内路径: {oss_internal_path}")
                print(f"   文件大小: {file_size / (1024*1024):.1f}MB")
                dataset_path = oss_internal_path
            else:
                print(f"❌ OSS内路径不存在: {oss_internal_path}")
                raise ValueError(f"数据集不可用，OSS下载器不可用且OSS内路径不存在")
    
    else:
        # 单机环境：直接使用给定路径
        print(f"💻 检测到单机环境，使用给定路径")
        print(f"数据集路径: {dataset_path}")
        
        if os.path.exists(dataset_path):
            file_size = os.path.getsize(dataset_path)
            print(f"✅ 使用指定数据集: {dataset_path}")
            print(f"   文件大小: {file_size / (1024*1024):.1f}MB")
        else:
            print(f"❌ 指定的数据集不存在: {dataset_path}")
            raise ValueError(f"数据集不存在: {dataset_path}")
    
    # 加载数据集
    print(f"📂 加载数据集: {dataset_path}")
    X_paths, Y_paths = load_dataset_from_npz(dataset_path)
    if X_paths is None or Y_paths is None:
        print(f"❌ 数据集加载失败!")
        print(f"💡 解决方案:")
        print(f"   1. 检查数据集文件是否完整: {dataset_path}")
        if is_cloud_environment:
            print(f"   2. 删除当前目录的数据集文件，重新从OSS下载")
            print(f"   3. 检查OSS配置文件: {oss_config_path}")
            print(f"   4. 检查OSS内路径: /mnt/data/msc_models/dataset_EMSC_big/dataset_EMSC_big.npz")
        else:
            print(f"   2. 检查数据集路径是否正确")
            print(f"   3. 确认数据集文件格式正确")
        raise ValueError("数据集加载失败")
    
    # 准备训练数据
    print("准备训练序列...")
    init_states = np.zeros((len(X_paths), training_config['STATE_DIM']), dtype=np.float32)
    
    # 随机打乱序列
    print("随机打乱训练序列...")
    np.random.seed(training_config['random_seed'])
    indices = np.random.permutation(len(X_paths))
    X_paths = [X_paths[i] for i in indices]
    Y_paths = [Y_paths[i] for i in indices]
    init_states = init_states[indices]
    
    # 划分训练集和验证集
    train_size = int(training_config['train_test_split_ratio'] * len(X_paths))
    X_train = X_paths[:train_size]
    Y_train = Y_paths[:train_size]
    init_states_train = init_states[:train_size]
    
    X_val = X_paths[train_size:]
    Y_val = Y_paths[train_size:]
    init_states_val = init_states[train_size:]
    
    print(f"训练集大小: {len(X_train)}")
    print(f"验证集大小: {len(X_val)}")
    
    # 确定批处理大小：优先使用用户指定的batch_size，否则自动计算
    if args.batch_size is not None:
        batch_size = args.batch_size
        optimal_batch_size = get_optimal_batch_size(len(X_train), num_workers)
        print(f"使用用户指定的batch_size: {batch_size}")
        if batch_size != optimal_batch_size:
            print(f"注意：建议的batch_size为: {optimal_batch_size}")
    else:
        batch_size = get_optimal_batch_size(len(X_train), num_workers)
        print(f"未指定batch_size，使用自动计算值: {batch_size}")
    
    # 创建TensorFlow数据集 - 针对云环境优化
    print("创建TensorFlow数据集...")
    
    if cloud_optimizer:
        # 使用云优化的数据集创建
        print("🌥️  使用云环境优化数据集...")
        train_dataset = cloud_optimizer.create_optimized_dataset(
            X_train, Y_train, init_states_train, batch_size
        )
        val_dataset = cloud_optimizer.create_optimized_dataset(
            X_val, Y_val, init_states_val, batch_size
        )
        
        # 云环境性能监控
        from EMSC_cloud_io_optimizer import monitor_cloud_performance
        monitor_cloud_performance()
        
    else:
        # 标准数据集创建 - 区分环境优化数据加载并行度
        if num_workers is not None:  # CPU模式
            data_parallel_calls = min(num_workers, 16)  # 限制最大并行度避免过度竞争
            prefetch_buffer = min(batch_size * 4, 64)  # 预取缓冲区
            print(f"CPU优化: 数据并行度={data_parallel_calls}, 预取缓冲={prefetch_buffer}")
        else:  # GPU模式
            env_type = detect_environment()
            if env_type == 'local':
                # 本地GPU环境 - 使用较小的并行度和缓冲区
                data_parallel_calls = 4  # 本地环境使用固定并行度
                prefetch_buffer = max(2, batch_size // 8)  # 较小的预取缓冲
                print(f"本地GPU优化: 数据并行度={data_parallel_calls}, 预取缓冲={prefetch_buffer}")
            else:
                # 云GPU环境 - 使用自动调优
                data_parallel_calls = tf.data.AUTOTUNE
                prefetch_buffer = tf.data.AUTOTUNE
                print(f"云GPU优化: 使用AUTOTUNE")
        
        train_dataset = create_tf_dataset(
            X_train, Y_train, init_states_train,
            batch_size=batch_size,
            shuffle=True,
            num_parallel_calls=data_parallel_calls
        ).prefetch(prefetch_buffer)
        
        val_dataset = create_tf_dataset(
            X_val, Y_val, init_states_val,
            batch_size=batch_size,
            shuffle=False,
            num_parallel_calls=data_parallel_calls
        ).prefetch(prefetch_buffer)
    
    print(f"数据加载配置:")
    print(f"- 最终使用的批处理大小: {batch_size}")
    print(f"- 数据加载线程数: {num_workers if num_workers is not None else 'GPU模式'}")
    
    # 计算最大序列长度
    max_seq_length = max(len(x) for x in X_paths)
    print(f"数据集中最大序列长度: {max_seq_length}")
    
    # 加载或创建模型
    epoch_offset = 0
    is_new_model = None
    if args.resume:
        print("尝试恢复训练...")
        model, epoch_offset = resume_training_from_checkpoint(
            model_path=dataset_dir,
            model_name=model_name,
            best_model_name=best_model_name,
            resume_from_best=True,
            state_dim=args.state_dim,
            input_dim=6,
            output_dim=1,
            hidden_dim=args.hidden_dim,
            num_internal_layers=2,
            max_sequence_length=max_seq_length
        )
        
        if model is None:
            print("无法恢复训练，将创建新模型")
            model, is_new_model = load_or_create_model_with_history(
                model_path=dataset_dir,
                model_name=model_name,
                best_model_name=best_model_name,
                state_dim=args.state_dim,
                input_dim=6,
                output_dim=1,
                hidden_dim=args.hidden_dim,
                num_internal_layers=2,
                max_sequence_length=max_seq_length
            )
    else:
        print("从头开始训练...")
        model, is_new_model = load_or_create_model_with_history(
            model_path=dataset_dir,
            model_name=model_name,
            best_model_name=best_model_name,
            state_dim=args.state_dim,
            input_dim=6,
            output_dim=1,
            hidden_dim=args.hidden_dim,
            num_internal_layers=2,
            max_sequence_length=max_seq_length
        )
    
    # 编译模型，添加梯度裁剪以提高数值稳定性
    optimizer = Adam(
        learning_rate=args.learning_rate,
        clipnorm=1.0,      # 梯度裁剪，防止梯度爆炸
        clipvalue=0.5      # 梯度值裁剪
    )
    custom_loss = EMSCLoss(state_dim=args.state_dim)
    model.compile(
        optimizer=optimizer,
        loss=custom_loss,
        # EMSC模型使用tf.while_loop，与JIT编译不兼容
        # while_loop创建动态控制流，JIT编译要求静态图结构
        jit_compile=False
    )
    
    if is_new_model:
        model.summary()
    
    # GPU模型预热（仅限GPU模式）
    if num_workers is None:  # GPU模式
        env_type = detect_environment()
        if env_type == 'local':
            # 本地GPU需要预热来避免卡顿
            print("🏠 本地GPU环境 - 执行模型预热")
            warmup_gpu_model(model, sample_batch_size=1, max_sequence_length=min(200, max_seq_length))
        else:
            # 云GPU环境可选预热
            print("☁️  云GPU环境 - 跳过预热（可选优化）")
    
    # 创建回调
    progress_callback = MSCProgressCallback(
        save_path=dataset_dir,
        model_name=model_name,
        best_model_name=best_model_name,
        save_frequency=args.save_frequency
    )
    
    early_stopping = create_early_stopping_callback()
    
    # 创建学习率调度器
    lr_scheduler = create_learning_rate_scheduler(
        initial_learning_rate=args.learning_rate,
        decay_type='validation',  # 使用基于验证损失的动态调整
        decay_steps=args.epochs,  # 总epochs数
        decay_rate=0.9,          # 指数衰减率（当使用exponential时）
        min_learning_rate=1e-6,  # 最小学习率
        patience=min(int(args.epochs/50),50),              # 验证损失不改善的容忍轮数
        factor=0.5,             # 学习率衰减因子
        verbose=1               # 打印学习率变化
    )
    
    # 创建CPU监控回调（仅CPU训练模式且用户启用时，且不与动态批次冲突）
    cpu_monitor = None
    if num_workers is not None and args.monitor_cpu and not args.dynamic_batch:
        cpu_monitor = create_cpu_monitor_callback(monitor_interval=30, verbose=True)
        print("已启用CPU使用率监控")
    elif num_workers is not None and args.monitor_cpu and args.dynamic_batch:
        print("注意：动态批次调整已包含CPU监控功能，--monitor_cpu将被忽略")
    
    # 准备回调列表
    callbacks = [progress_callback, early_stopping, lr_scheduler]
    if cpu_monitor is not None:
        callbacks.append(cpu_monitor)
    
    # 训练模型
    remaining_epochs = args.epochs
    if remaining_epochs <= 0:
        print(f"模型已经训练了 {epoch_offset} epochs，达到设定的总epochs {args.epochs}")
        print("如需继续训练，请增加总epochs数")
    else:
        print(f"\n开始训练 MSC 模型...")
        print(f"已完成epochs: {epoch_offset}")
        print(f"剩余epochs: {remaining_epochs}")
        print(f"总epochs目标: {args.epochs + epoch_offset}")
        print(f"批次大小: {batch_size}")
        print(f"保存频率: 每 {args.save_frequency} epochs")
        print(f"早停设置: patience={15}, min_delta={1e-4}")
        print(f"学习率调度: 初始={args.learning_rate}, 最小={1e-6}, 动态调整")
        print(f"模型保存路径: {dataset_dir}")
        print(f"训练数据大小: {len(X_train)}")
        print(f"验证数据大小: {len(X_val)}")
        env_info = f"({'本地' if detect_environment() == 'local' else '云'})" if num_workers is None else ""
        print(f"训练模式: {'GPU ' + env_info if num_workers is None else 'CPU (多线程)'}")
        if num_workers is None:
            env_type = detect_environment()
            print(f"GPU优化设置 ({env_type}):")
            print(f"  - XLA JIT编译: 已禁用 (while_loop兼容性)")
            if env_type == 'local':
                print(f"  - TensorFloat-32: 已启用 (本地性能优化)")
                print(f"  - 线程配置: 使用默认 (本地优化)")
            else:
                print(f"  - TensorFloat-32: 已禁用 (云环境精度优先)")
                print(f"  - 线程配置: 保守设置 (云环境稳定性)")
            print(f"  - 梯度裁剪: clipnorm=1.0, clipvalue=0.5")
            print(f"  - 数值稳定性: EMSCLoss保护 + 梯度裁剪")
        
        # 使用性能优化的训练配置 - 针对阿里云CPU优化
        if num_workers is not None and args.dynamic_batch:  # CPU模式 + 动态批次
            print(f"🚀 启用动态批次大小调整 (目标CPU使用率: {args.target_cpu_usage}%)")
            
            # 使用动态批次训练器
            dynamic_trainer = DynamicBatchTrainer(
                model=model,
                train_data_info=(X_train, Y_train, init_states_train),
                val_data_info=(X_val, Y_val, init_states_val),
                initial_batch_size=batch_size
            )
            
            # 添加动态批次回调（替换CPU监控）
            dynamic_callbacks = [progress_callback, early_stopping, lr_scheduler]
            dynamic_callbacks.append(create_dynamic_batch_callback(
                target_cpu_usage=args.target_cpu_usage,
                min_batch_size=16,
                max_batch_size=min(512, len(X_train)),
                verbose=True
            ))
            
            history = dynamic_trainer.fit(
                epochs=args.epochs,
                initial_epoch=epoch_offset,
                verbose=1,
                callbacks=dynamic_callbacks,
                use_multiprocessing=True,
                workers=min(num_workers, 32),
                max_queue_size=max(20, num_workers * 2)
            )
            
        elif num_workers is not None:  # CPU模式 - 传统训练
            # CPU训练配置 - 更激进的多进程设置
            max_queue = max(20, num_workers * 2)  # 增加队列大小
            cpu_workers = min(num_workers, 32)    # 限制最大进程数避免过度开销
            print(f"CPU训练配置: workers={cpu_workers}, max_queue_size={max_queue}")
            
            history = model.fit(
                 train_dataset,
                 validation_data=val_dataset,
                 epochs=args.epochs,
                 initial_epoch=epoch_offset,
                 verbose=1,
                 callbacks=callbacks,
                 use_multiprocessing=True,  # 启用多进程
                 workers=cpu_workers,
                 max_queue_size=max_queue
             )
        else:  # GPU模式
            env_type = detect_environment()
            if env_type == 'local':
                # 本地GPU训练配置 - 优化性能
                print(f"🏠 本地GPU训练配置: 禁用多进程，优化内存使用")
                history = model.fit(
                    train_dataset,
                    validation_data=val_dataset,
                    epochs=args.epochs,
                    initial_epoch=epoch_offset,
                    verbose=1,
                    callbacks=callbacks,
                    use_multiprocessing=False,  # 本地GPU避免多进程竞争
                    workers=1,                  # 单工作线程
                    max_queue_size=2            # 较小的队列减少内存占用
                )
            else:
                # 云GPU训练配置 - 标准设置
                print(f"☁️  云GPU训练配置: 标准设置")
                history = model.fit(
                    train_dataset,
                    validation_data=val_dataset,
                    epochs=args.epochs,
                    initial_epoch=epoch_offset,
                    verbose=1,
                    callbacks=callbacks,
                    use_multiprocessing=False,
                    workers=1,
                    max_queue_size=10
                )
        
        # 训练完成后最终保存
        print("\n训练完成，执行最终保存...")
        final_model_path = progress_callback._safe_save_model(model, is_best=False)
        
        # 保存最终的训练历史和图表
        progress_callback._save_training_history()
        progress_callback._plot_training_history()
        
        # 绘制最终训练总结
        plot_final_training_summary(
            history, epoch_offset, args.epochs,
            progress_callback, dataset_dir
        )
        
        # 打印训练总结
        print_training_summary(
            progress_callback, dataset_dir,
            best_model_name, model_name
        )

if __name__ == '__main__':
    main()