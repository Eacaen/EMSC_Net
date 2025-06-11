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
from EMSC_config import (create_training_config, save_training_config, 
                        parse_training_args, get_dataset_paths)
from EMSC_utils import (load_or_create_model_with_history, 
                       resume_training_from_checkpoint,
                       plot_final_training_summary,
                       print_training_summary)
from EMSC_losses import EMSCLoss

def check_environment():
    """检查并配置训练环境，优先使用GPU，回退到CPU"""
    print("检查训练环境...")
    print(f"TensorFlow版本: {tf.__version__}")
    print(f"当前工作目录: {os.getcwd()}")
    
    # 检查GPU可用性
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
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
            print(f"使用GPU设备: {gpus[0]}")
            return None  # 使用GPU时不需要返回worker数
        except RuntimeError as e:
            print(f"设置GPU设备时出错: {e}")
    
    # 如果没有GPU或GPU设置失败，配置CPU环境
    print("未发现GPU设备或GPU设置失败，将使用CPU训练")
    
    # 获取CPU核心数
    cpu_count = os.cpu_count()
    if cpu_count is None:
        cpu_count = 4  # 默认值
    
    # 设置线程数
    # 保留CPU核心给系统和其他进程
    num_workers = max(1, cpu_count - 1)  # 保留1个CPU核心给系统和其他进程
    
    # 设置TensorFlow线程配置
    tf.config.threading.set_inter_op_parallelism_threads(num_workers)
    tf.config.threading.set_intra_op_parallelism_threads(num_workers)
    
    # 设置OpenMP线程数（用于NumPy等库）
    os.environ['OMP_NUM_THREADS'] = str(num_workers)
    os.environ['MKL_NUM_THREADS'] = str(num_workers)
    
    print(f"CPU环境配置完成:")
    print(f"- 总CPU核心数: {cpu_count}")
    print(f"- 训练使用线程数: {num_workers}")
    
    # 检查内存使用情况
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"\n系统内存信息:")
        print(f"总内存: {memory.total / (1024**3):.1f} GB")
        print(f"可用内存: {memory.available / (1024**3):.1f} GB")
        print(f"内存使用率: {memory.percent}%")
    except ImportError:
        print("未安装psutil，跳过内存检查")
    
    return num_workers

def get_optimal_batch_size(num_samples, num_workers):
    """
    计算最优批处理大小
    
    Args:
        num_samples: 训练样本数量
        num_workers: 工作线程数（CPU模式）或None（GPU模式）
    
    Returns:
        int: 最优批处理大小
    """
    if num_workers is None:  # GPU模式
        # GPU模式下使用较大的批处理大小
        return min(128, num_samples // 50)
    
    # CPU模式下的批处理大小计算
    base_batch = min(32, num_samples // 100)
    base_batch = max(8, base_batch)
    
    # 根据CPU线程数调整
    optimal_batch = base_batch * num_workers
    
    # 确保是8的倍数（对内存对齐有利）
    optimal_batch = (optimal_batch // 8) * 8
    
    return optimal_batch

def main():
    # 检查并配置环境
    num_workers = check_environment()
    
    # 设置TensorFlow的默认数据类型
    tf.keras.backend.set_floatx('float32')
    
    # 解析命令行参数
    args = parse_training_args()
    
    # 获取数据集路径
    paths = get_dataset_paths(args.dataset)
    dataset_dir = paths['dataset_dir']
    model_name = paths['model_name']
    best_model_name = paths['best_model_name']
    dataset_path = paths['dataset_path']
    
    # 创建和保存训练配置
    training_config = create_training_config(
        state_dim=args.state_dim,
        input_dim=6,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        target_sequence_length=1000,
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_frequency=args.save_frequency
    )
    save_training_config(training_config, dataset_dir)
    
    # 加载数据集
    print(f"尝试加载数据集: {dataset_path}")
    X_paths, Y_paths = load_dataset_from_npz(dataset_path)
    if X_paths is None or Y_paths is None:
        raise ValueError("未能成功加载数据集")
    
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
    
    # 计算最优批处理大小
    optimal_batch_size = get_optimal_batch_size(len(X_train), num_workers)
    
    # 创建TensorFlow数据集
    print("创建TensorFlow数据集...")
    train_dataset = create_tf_dataset(
        X_train, Y_train, init_states_train,
        batch_size=optimal_batch_size,
        shuffle=True,
        num_parallel_calls=tf.data.AUTOTUNE  # 自动优化并行度
    )
    
    val_dataset = create_tf_dataset(
        X_val, Y_val, init_states_val,
        batch_size=optimal_batch_size,
        shuffle=False,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    print(f"数据加载优化完成:")
    print(f"- 优化后的批处理大小: {optimal_batch_size}")
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
    
    # 编译模型
    optimizer = Adam(args.learning_rate)
    custom_loss = EMSCLoss(state_dim=args.state_dim)
    model.compile(
        optimizer=optimizer,
        loss=custom_loss,
        # 暂时禁用JIT编译以避免XLA要求固定tensor大小的问题
        jit_compile=False
    )
    
    if is_new_model:
        model.summary()
    
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
        patience=5,              # 验证损失不改善的容忍轮数
        factor=0.5,             # 学习率衰减因子
        verbose=1               # 打印学习率变化
    )
    
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
        print(f"批次大小: {optimal_batch_size}")
        print(f"保存频率: 每 {args.save_frequency} epochs")
        print(f"早停设置: patience={15}, min_delta={1e-4}")
        print(f"学习率调度: 初始={args.learning_rate}, 最小={1e-6}, 动态调整")
        print(f"模型保存路径: {dataset_dir}")
        print(f"训练数据大小: {len(X_train)}")
        print(f"验证数据大小: {len(X_val)}")
        print(f"训练模式: {'GPU' if num_workers is None else 'CPU (多线程)'}")
        
        # 使用性能优化的训练配置
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=args.epochs,
            initial_epoch=epoch_offset,
            verbose=1,
            callbacks=[progress_callback, early_stopping, lr_scheduler],
            # 仅在CPU模式下启用多进程
            use_multiprocessing=num_workers is not None,
            workers=num_workers if num_workers is not None else 1,
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