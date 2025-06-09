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
from EMSC_callbacks import MSCProgressCallback, create_early_stopping_callback
from EMSC_config import (create_training_config, save_training_config, 
                        parse_training_args, get_dataset_paths)
from EMSC_utils import (load_or_create_model_with_history, 
                       resume_training_from_checkpoint,
                       plot_final_training_summary,
                       print_training_summary)
from EMSC_losses import EMSCLoss

def setup_cpu_environment():
    """设置CPU训练环境"""
    # 获取CPU核心数
    cpu_count = os.cpu_count()
    if cpu_count is None:
        cpu_count = 4  # 默认值
    
    # 设置线程数
    # 保留一半CPU核心给系统和其他进程
    num_workers = max(1, cpu_count // 2)
    
    # 设置TensorFlow线程配置
    tf.config.threading.set_inter_op_parallelism_threads(num_workers)
    tf.config.threading.set_intra_op_parallelism_threads(num_workers)
    
    # 设置OpenMP线程数（用于NumPy等库）
    os.environ['OMP_NUM_THREADS'] = str(num_workers)
    os.environ['MKL_NUM_THREADS'] = str(num_workers)
    
    print(f"CPU环境配置完成:")
    print(f"- 总CPU核心数: {cpu_count}")
    print(f"- 训练使用线程数: {num_workers}")
    
    return num_workers

def get_optimal_batch_size(num_samples, num_workers):
    """
    计算CPU训练的最优批处理大小
    
    Args:
        num_samples: 训练样本数量
        num_workers: 工作线程数
    
    Returns:
        int: 最优批处理大小
    """
    # 基础批处理大小
    base_batch = min(32, num_samples // 100)
    base_batch = max(8, base_batch)
    
    # 根据CPU线程数调整
    optimal_batch = base_batch * num_workers
    
    # 确保是8的倍数（对内存对齐有利）
    optimal_batch = (optimal_batch // 8) * 8
    
    return optimal_batch

def main():
    # 设置CPU环境
    num_workers = setup_cpu_environment()
    
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
    
    # 创建数据生成器
    print("创建数据生成器...")
    train_generator = EMSCDataGenerator(
        X_train, Y_train, init_states_train,
        batch_size=optimal_batch_size,
        shuffle=True,
        num_workers=num_workers,  # 使用多线程
        prefetch_factor=2  # 预加载因子
    )
    
    val_generator = EMSCDataGenerator(
        X_val, Y_val, init_states_val,
        batch_size=optimal_batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=2
    )
    
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
    print(f"- 训练集批次数: {len(train_generator)}")
    print(f"- 验证集批次数: {len(val_generator)}")
    print(f"- 数据加载线程数: {num_workers}")
    
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
            num_internal_layers=2
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
                num_internal_layers=2
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
            num_internal_layers=2
        )
    
    # 编译模型
    optimizer = Adam(args.learning_rate)
    custom_loss = EMSCLoss(state_dim=args.state_dim)
    model.compile(
        optimizer=optimizer,
        loss=custom_loss,
        # 启用JIT编译以提高CPU性能
        jit_compile=True
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
        print(f"早停设置: patience={50}, min_delta={1e-4}")
        print(f"模型保存路径: {dataset_dir}")
        print(f"训练数据大小: {len(X_train)}")
        print(f"验证数据大小: {len(X_val)}")
        print(f"CPU线程数: {num_workers}")
        
        # 使用性能优化的训练配置
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=args.epochs,
            initial_epoch=epoch_offset,
            verbose=1,
            callbacks=[progress_callback, early_stopping],
            # 启用多进程和多线程
            use_multiprocessing=True,
            workers=num_workers,
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