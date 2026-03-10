#!/usr/bin/env python3
"""
分析NPZ文件数据结构的脚本
用于分析motion数据文件的结构和内容
"""

import numpy as np
import os
import sys
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

def analyze_npz_file(npz_path: str):
    """分析NPZ文件的数据结构"""
    
    if not os.path.exists(npz_path):
        print(f"错误: 文件不存在: {npz_path}")
        return
    
    print("=" * 80)
    print(f"分析文件: {npz_path}")
    print("=" * 80)
    print()
    
    # 加载文件
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        print(f"错误: 无法加载文件: {e}")
        return
    
    # 获取所有keys
    keys = data.files
    print(f"📦 文件包含 {len(keys)} 个数据键:")
    print(f"Keys: {keys}")
    print()
    
    # 分析每个key
    for key in keys:
        print("-" * 80)
        print(f"🔑 Key: '{key}'")
        print("-" * 80)
        
        arr = data[key]
        
        # 基本属性
        print(f"  类型 (type): {type(arr)}")
        
        # 如果是object数组，需要特殊处理
        if isinstance(arr, np.ndarray) and arr.dtype == object:
            print(f"  数据类型 (dtype): {arr.dtype} (object array)")
            print(f"  形状 (shape): {arr.shape}")
            print(f"  数组长度: {len(arr)}")
            
            # 分析object数组中的元素
            if len(arr) > 0:
                print(f"\n  📊 Object数组内容分析:")
                print(f"    第一个元素类型: {type(arr[0])}")
                
                # 检查第一个元素是否是numpy数组
                if isinstance(arr[0], np.ndarray):
                    print(f"    第一个元素shape: {arr[0].shape}")
                    print(f"    第一个元素dtype: {arr[0].dtype}")
                    
                    # 统计所有元素的shape
                    shapes = [item.shape if isinstance(item, np.ndarray) else None for item in arr[:10]]
                    print(f"    前10个元素的shapes: {shapes[:10]}")
                    
                    # 如果所有元素shape相同，显示统计信息
                    if len(set([item.shape if isinstance(item, np.ndarray) else None for item in arr])) == 1:
                        # 尝试堆叠所有元素
                        try:
                            stacked = np.stack([item for item in arr if isinstance(item, np.ndarray)])
                            print(f"\n    ✅ 所有元素shape相同，可以堆叠")
                            print(f"    堆叠后shape: {stacked.shape}")
                            print(f"    堆叠后dtype: {stacked.dtype}")
                            
                            # 统计信息
                            print(f"\n    📈 统计信息:")
                            print(f"      Min: {np.min(stacked):.6f}")
                            print(f"      Max: {np.max(stacked):.6f}")
                            print(f"      Mean: {np.mean(stacked):.6f}")
                            print(f"      Std: {np.std(stacked):.6f}")
                            
                            # 显示第一个和最后一个元素的部分数据
                            print(f"\n    📝 示例数据 (第一个元素的前5个值):")
                            if stacked.ndim >= 1:
                                print(f"      {stacked[0, :min(5, stacked.shape[-1])]}")
                            
                        except Exception as e:
                            print(f"    ⚠️  无法堆叠: {e}")
                            # 分析单个元素
                            if len(arr) > 0 and isinstance(arr[0], np.ndarray):
                                first_elem = arr[0]
                                print(f"\n    📝 第一个元素统计:")
                                print(f"      Shape: {first_elem.shape}")
                                print(f"      Dtype: {first_elem.dtype}")
                                print(f"      Min: {np.min(first_elem):.6f}")
                                print(f"      Max: {np.max(first_elem):.6f}")
                                print(f"      Mean: {np.mean(first_elem):.6f}")
                                print(f"      Std: {np.std(first_elem):.6f}")
                                print(f"      前5个值: {first_elem.flatten()[:5]}")
                else:
                    print(f"    第一个元素内容: {arr[0]}")
                    print(f"    第一个元素类型: {type(arr[0])}")
                    
                    # 如果是字符串数组（如joint_sequence），显示所有内容
                    if isinstance(arr[0], str):
                        print(f"\n    📝 完整内容 ({len(arr)} 个元素):")
                        for i, item in enumerate(arr):
                            print(f"      [{i:2d}] {item}")
            
            # 显示前几个元素的信息
            print(f"\n  📋 前3个元素预览:")
            for i in range(min(3, len(arr))):
                elem = arr[i]
                if isinstance(elem, np.ndarray):
                    print(f"    [{i}] shape={elem.shape}, dtype={elem.dtype}, "
                          f"min={np.min(elem):.4f}, max={np.max(elem):.4f}, mean={np.mean(elem):.4f}")
                else:
                    print(f"    [{i}] type={type(elem)}, value={elem}")
        
        else:
            # 普通numpy数组
            print(f"  数据类型 (dtype): {arr.dtype}")
            print(f"  形状 (shape): {arr.shape}")
            print(f"  维度 (ndim): {arr.ndim}")
            print(f"  总元素数: {arr.size}")
            print(f"  内存大小: {arr.nbytes / 1024 / 1024:.2f} MB")
            
            # 统计信息
            if arr.size > 0:
                print(f"\n  📈 统计信息:")
                print(f"    Min: {np.min(arr):.6f}")
                print(f"    Max: {np.max(arr):.6f}")
                print(f"    Mean: {np.mean(arr):.6f}")
                print(f"    Std: {np.std(arr):.6f}")
                print(f"    Median: {np.median(arr):.6f}")
                
                # 显示部分数据
                print(f"\n  📝 数据预览:")
                if arr.ndim == 1:
                    print(f"    前10个值: {arr[:10]}")
                    print(f"    后10个值: {arr[-10:]}")
                elif arr.ndim == 2:
                    print(f"    形状: {arr.shape[0]} x {arr.shape[1]}")
                    print(f"    第一行前10个值: {arr[0, :10]}")
                    print(f"    第一列前10个值: {arr[:10, 0]}")
                else:
                    print(f"    第一个元素: {arr.flat[:10]}")
        
        print()
    
    # 总结
    print("=" * 80)
    print("📊 数据总结")
    print("=" * 80)
    
    # 计算总大小
    total_size = 0
    motion_lengths = []
    for key in keys:
        arr = data[key]
        if isinstance(arr, np.ndarray):
            if arr.dtype == object:
                # object数组的大小估算
                if len(arr) > 0 and isinstance(arr[0], np.ndarray):
                    # 对于motion数据，收集长度信息
                    if key in ['real_dof_positions', 'real_dof_velocities', 'real_dof_positions_cmd', 'real_dof_torques']:
                        for item in arr:
                            if isinstance(item, np.ndarray):
                                motion_lengths.append(item.shape[0])
                    # 估算大小
                    total_size += sum([item.nbytes if isinstance(item, np.ndarray) else sys.getsizeof(item) 
                                     for item in arr[:100]])  # 只估算前100个
                else:
                    total_size += sum([sys.getsizeof(item) for item in arr[:100]])
            else:
                total_size += arr.nbytes
    
    print(f"总数据大小 (估算): {total_size / 1024 / 1024:.2f} MB")
    
    # Motion长度统计
    if motion_lengths:
        motion_lengths = np.array(motion_lengths)
        print(f"\n📏 Motion片段长度统计:")
        print(f"  总motion数量: {len(motion_lengths)}")
        print(f"  最短长度: {np.min(motion_lengths)} 步")
        print(f"  最长长度: {np.max(motion_lengths)} 步")
        print(f"  平均长度: {np.mean(motion_lengths):.1f} 步")
        print(f"  中位数长度: {np.median(motion_lengths):.1f} 步")
        print(f"  标准差: {np.std(motion_lengths):.1f} 步")
        
        # 长度分布
        unique_lengths, counts = np.unique(motion_lengths, return_counts=True)
        print(f"\n  长度分布 (前10个最常见的长度):")
        sorted_indices = np.argsort(counts)[::-1][:10]
        for idx in sorted_indices:
            print(f"    {unique_lengths[idx]} 步: {counts[idx]} 个motion ({counts[idx]/len(motion_lengths)*100:.1f}%)")
    
    print()
    
    # Payload分布分析
    if 'payloads' in keys:
        payloads = data['payloads']
        if isinstance(payloads, np.ndarray) and payloads.dtype != object:
            print("📦 Payload分布分析:")
            unique_payloads, counts = np.unique(payloads, return_counts=True)
            print(f"  不同payload值数量: {len(unique_payloads)}")
            print(f"  Payload值分布:")
            for payload, count in zip(unique_payloads, counts):
                print(f"    {payload:.1f} kg: {count} 个motion ({count/len(payloads)*100:.1f}%)")
            print()
    
    # 时间信息（基于50Hz采样率）
    if len(motion_lengths) > 0:
        print("⏱️  时间信息 (基于50Hz采样率):")
        print(f"  最短motion时长: {np.min(motion_lengths) / 50:.2f} 秒")
        print(f"  最长motion时长: {np.max(motion_lengths) / 50:.2f} 秒")
        print(f"  平均motion时长: {np.mean(motion_lengths) / 50:.2f} 秒")
        print(f"  总数据时长: {np.sum(motion_lengths) / 50:.2f} 秒 ({np.sum(motion_lengths) / 50 / 60:.2f} 分钟)")
        print()
    
    # 根据key名称提供解释
    print("📖 数据键说明:")
    key_descriptions = {
        'real_dof_positions': '真实关节位置数据 (rad)',
        'real_dof_velocities': '真实关节速度数据 (rad/s)',
        'real_dof_positions_cmd': '目标关节位置（命令）(rad)',
        'real_dof_torques': '真实关节力矩数据 (N·m)',
        'joint_sequence': '关节序列（31个关节的名称列表）',
        'payloads': '负载质量数据 (kg)'
    }
    
    for key in keys:
        desc = key_descriptions.get(key, '未知')
        print(f"  - {key}: {desc}")
    
    print()
    print("=" * 80)
    print("✅ 分析完成！")
    print("=" * 80)
    
    return data  # 返回数据以便后续绘图使用


def validate_data_source(npz_path: str, motion_idx: int = 0):
    """
    Validate data source file and data structure.
    
    Args:
        npz_path: Path to NPZ file
        motion_idx: Motion index to validate
        
    Returns:
        tuple: (is_valid: bool, file_info: dict, data: np.lib.npyio.NpzFile or None, error_msg: str or None)
    """
    file_info = {}
    error_msg = None
    data = None
    
    # Check file existence and get file info
    if not os.path.exists(npz_path):
        error_msg = f"File does not exist: {npz_path}"
        return False, file_info, None, error_msg
    
    # Get file metadata
    try:
        file_stat = os.stat(npz_path)
        file_info['path'] = os.path.abspath(npz_path)
        file_info['size'] = file_stat.st_size
        file_info['size_mb'] = file_stat.st_size / (1024 * 1024)
        file_info['modified_time'] = datetime.fromtimestamp(file_stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
        file_info['readable'] = os.access(npz_path, os.R_OK)
    except Exception as e:
        error_msg = f"Failed to get file info: {e}"
        return False, file_info, None, error_msg
    
    if not file_info['readable']:
        error_msg = f"File is not readable: {npz_path}"
        return False, file_info, None, error_msg
    
    # Try to load the file
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        error_msg = f"Failed to load NPZ file: {e}"
        return False, file_info, None, error_msg
    
    # Validate required keys
    required_keys = ['real_dof_positions', 'real_dof_positions_cmd']
    missing_keys = [key for key in required_keys if key not in data.files]
    if missing_keys:
        error_msg = f"Missing required data keys: {missing_keys}"
        return False, file_info, data, error_msg
    
    # Get motion data
    real_positions = data['real_dof_positions']
    cmd_positions = data['real_dof_positions_cmd']
    
    # Check motion index validity
    if motion_idx >= len(real_positions) or motion_idx >= len(cmd_positions):
        error_msg = f"Motion index {motion_idx} out of range (total {len(real_positions)} motions)"
        return False, file_info, data, error_msg
    
    # Check data shapes
    try:
        real_pos = real_positions[motion_idx]
        cmd_pos = cmd_positions[motion_idx]
        
        if not isinstance(real_pos, np.ndarray) or not isinstance(cmd_pos, np.ndarray):
            error_msg = f"Motion data at index {motion_idx} is not a numpy array"
            return False, file_info, data, error_msg
        
        if real_pos.shape != cmd_pos.shape:
            error_msg = f"Shape mismatch between real and command positions: {real_pos.shape} vs {cmd_pos.shape}"
            return False, file_info, data, error_msg
        
        # Add data shape info
        file_info['num_motions'] = len(real_positions)
        file_info['motion_shape'] = real_pos.shape
        file_info['num_joints'] = real_pos.shape[1] if len(real_pos.shape) > 1 else 1
        file_info['time_steps'] = real_pos.shape[0] if len(real_pos.shape) > 0 else 0
        
    except Exception as e:
        error_msg = f"Error accessing motion data: {e}"
        return False, file_info, data, error_msg
    
    return True, file_info, data, None


def plot_position_comparison(npz_path: str, motion_idx: int = 0, save_path: str = None, 
                             sample_rate: float = 50.0, max_joints_per_figure: int = 8):
    """
    Plot comparison between real_dof_positions and real_dof_positions_cmd.
    
    Args:
        npz_path: Path to NPZ file
        motion_idx: Motion index to plot (default: 0, first motion)
        save_path: Path to save figures (if None, auto-generated)
        sample_rate: Sampling rate in Hz (default: 50.0)
        max_joints_per_figure: Maximum joints per figure (default: 8)
    """
    print("=" * 80)
    print(f"📊 Plotting Position Comparison")
    print("=" * 80)
    print(f"File: {npz_path}")
    print(f"Motion index: {motion_idx}")
    print()
    
    # Validate data source
    is_valid, file_info, data, error_msg = validate_data_source(npz_path, motion_idx)
    
    if not is_valid:
        print(f"❌ Validation failed: {error_msg}")
        return
    
    # Print file information
    print("📁 File Information:")
    print(f"  Path: {file_info['path']}")
    print(f"  Size: {file_info['size_mb']:.2f} MB ({file_info['size']} bytes)")
    print(f"  Modified: {file_info['modified_time']}")
    print(f"  Number of motions: {file_info['num_motions']}")
    print(f"  Motion shape: {file_info['motion_shape']}")
    print(f"  Number of joints: {file_info['num_joints']}")
    print(f"  Time steps: {file_info['time_steps']}")
    print()
    
    # Get joint names
    joint_names = None
    if 'joint_sequence' in data.files:
        joint_names = data['joint_sequence']
    
    # Get motion data
    real_positions = data['real_dof_positions']
    cmd_positions = data['real_dof_positions_cmd']
    
    # Get selected motion data
    real_pos = real_positions[motion_idx]  # shape: (time_steps, num_joints)
    cmd_pos = cmd_positions[motion_idx]     # shape: (time_steps, num_joints)
    
    time_steps, num_joints = real_pos.shape
    
    print(f"Motion data shape: {time_steps} steps x {num_joints} joints")
    print(f"Duration: {time_steps / sample_rate:.2f} seconds")
    print()
    
    # Create time axis (seconds)
    time_axis = np.arange(time_steps) / sample_rate
    
    # Calculate differences
    position_error = real_pos - cmd_pos  # shape: (time_steps, num_joints)
    
    # Calculate statistics
    mse_per_joint = np.mean(position_error ** 2, axis=0)  # Mean squared error per joint
    max_error_per_joint = np.max(np.abs(position_error), axis=0)  # Max absolute error per joint
    mean_error_per_joint = np.mean(np.abs(position_error), axis=0)  # Mean absolute error per joint
    
    print("📈 Error Statistics (per joint):")
    print(f"{'Joint':<30} {'MSE':<12} {'Max Error':<12} {'Mean Error':<12}")
    print("-" * 70)
    for i in range(num_joints):
        joint_name = joint_names[i] if joint_names is not None else f"Joint_{i}"
        print(f"{joint_name:<30} {mse_per_joint[i]:<12.6f} {max_error_per_joint[i]:<12.6f} {mean_error_per_joint[i]:<12.6f}")
    print()
    
    # 确定需要多少个图
    num_figures = (num_joints + max_joints_per_figure - 1) // max_joints_per_figure
    
    # 生成保存路径
    if save_path is None:
        base_dir = os.path.dirname(npz_path)
        base_name = os.path.splitext(os.path.basename(npz_path))[0]
        save_path = os.path.join(base_dir, f"{base_name}_motion{motion_idx}_position_comparison.png")
    
    # 创建多个图
    for fig_idx in range(num_figures):
        start_joint = fig_idx * max_joints_per_figure
        end_joint = min((fig_idx + 1) * max_joints_per_figure, num_joints)
        joints_in_figure = end_joint - start_joint
        
        # 创建子图
        fig, axes = plt.subplots(joints_in_figure, 1, figsize=(14, 2.5 * joints_in_figure))
        if joints_in_figure == 1:
            axes = [axes]
        
        fig.suptitle(f'Joint Position Comparison - Motion {motion_idx} (Figure {fig_idx + 1}/{num_figures})', 
                     fontsize=14, fontweight='bold')
        
        for i, joint_idx in enumerate(range(start_joint, end_joint)):
            ax = axes[i]
            joint_name = joint_names[joint_idx] if joint_names is not None else f"Joint_{joint_idx}"
            
            # Plot real and command positions
            ax.plot(time_axis, real_pos[:, joint_idx], 'b-', label='Real Position', 
                   linewidth=1.5, alpha=0.7)
            ax.plot(time_axis, cmd_pos[:, joint_idx], 'r--', label='Command Position', 
                   linewidth=1.5, alpha=0.7)
            
            # Fill difference area
            ax.fill_between(time_axis, real_pos[:, joint_idx], cmd_pos[:, joint_idx], 
                           alpha=0.2, color='gray', label='Difference Area')
            
            # Set labels and title
            ax.set_ylabel(f'{joint_name}\nPosition (rad)', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=8)
            
            # Display error statistics on plot
            mse_text = f'MSE: {mse_per_joint[joint_idx]:.6f}\n'
            max_err_text = f'Max Err: {max_error_per_joint[joint_idx]:.6f}'
            ax.text(0.02, 0.98, mse_text + max_err_text, 
                   transform=ax.transAxes, fontsize=8,
                   verticalalignment='top', bbox=dict(boxstyle='round', 
                   facecolor='wheat', alpha=0.5))
            
            # Only show x-axis label on last subplot
            if i == joints_in_figure - 1:
                ax.set_xlabel('Time (s)', fontsize=10)
            else:
                ax.set_xticklabels([])
        
        plt.tight_layout()
        
        # Save figure
        if num_figures > 1:
            figure_save_path = save_path.replace('.png', f'_part{fig_idx + 1}.png')
        else:
            figure_save_path = save_path
        
        plt.savefig(figure_save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved figure: {figure_save_path}")
        plt.close()
    
    # Create error summary plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Overview of all joint positions (first few time steps)
    ax1 = axes[0]
    # Only show first 1000 time steps (if data is too long)
    display_steps = min(1000, time_steps)
    display_time = time_axis[:display_steps]
    
    for joint_idx in range(num_joints):
        joint_name = joint_names[joint_idx] if joint_names is not None else f"J_{joint_idx}"
        ax1.plot(display_time, real_pos[:display_steps, joint_idx], 
                alpha=0.3, linewidth=0.5, label=f'{joint_name} (real)')
        ax1.plot(display_time, cmd_pos[:display_steps, joint_idx], 
                '--', alpha=0.3, linewidth=0.5, label=f'{joint_name} (cmd)')
    
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Joint Position (rad)', fontsize=12)
    ax1.set_title(f'All Joints Position Overview - Motion {motion_idx} (First {display_steps} steps)', 
                  fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(ncol=4, fontsize=6, loc='upper right')
    
    # Plot 2: Error statistics bar chart
    ax2 = axes[1]
    x_pos = np.arange(num_joints)
    width = 0.35
    
    bars1 = ax2.bar(x_pos - width/2, mse_per_joint, width, label='Mean Squared Error (MSE)', alpha=0.7)
    bars2 = ax2.bar(x_pos + width/2, mean_error_per_joint, width, label='Mean Absolute Error (MAE)', alpha=0.7)
    
    ax2.set_xlabel('Joint Index', fontsize=12)
    ax2.set_ylabel('Error Value', fontsize=12)
    ax2.set_title(f'Error Statistics per Joint - Motion {motion_idx}', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos)
    if joint_names is not None:
        ax2.set_xticklabels([name[:10] for name in joint_names], rotation=45, ha='right', fontsize=8)
    else:
        ax2.set_xticklabels([f'J{i}' for i in range(num_joints)], rotation=45, ha='right', fontsize=8)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save error summary plot
    error_summary_path = save_path.replace('.png', '_error_summary.png')
    plt.savefig(error_summary_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved error summary: {error_summary_path}")
    plt.close()
    
    print()
    print("=" * 80)
    print("✅ Plotting completed!")
    print("=" * 80)


def plot_all_motions(npz_path: str, base_save_dir: str = None,
                     sample_rate: float = 50.0, max_joints_per_figure: int = 8,
                     silent: bool = False):
    """
    Plot comparison for all motions in the NPZ file.
    
    Args:
        npz_path: Path to NPZ file
        base_save_dir: Base directory to save plots (default: /home/wanhe/GR-3/gaponet/data_process/plots)
        sample_rate: Sampling rate in Hz (default: 50.0)
        max_joints_per_figure: Maximum joints per figure (default: 8)
        silent: If True, suppress individual motion plotting output (default: False)
    """
    # Set default save directory
    if base_save_dir is None:
        base_save_dir = "/home/wanhe/GR-3/gaponet/data_process/plots"
    
    # Create base directory if it doesn't exist
    os.makedirs(base_save_dir, exist_ok=True)
    
    # Validate data source (check first motion to get file info)
    is_valid, file_info, data, error_msg = validate_data_source(npz_path, motion_idx=0)
    
    if not is_valid:
        print(f"❌ Validation failed: {error_msg}")
        return
    
    num_motions = file_info['num_motions']
    
    # Create subdirectory based on NPZ file name
    npz_basename = os.path.splitext(os.path.basename(npz_path))[0]
    save_dir = os.path.join(base_save_dir, npz_basename)
    os.makedirs(save_dir, exist_ok=True)
    
    if not silent:
        print("=" * 80)
        print(f"📊 Plotting All Motions")
        print("=" * 80)
        print(f"File: {npz_path}")
        print(f"Total motions: {num_motions}")
        print(f"Save directory: {save_dir}")
        print()
        
        # Print file information
        print("📁 File Information:")
        print(f"  Path: {file_info['path']}")
        print(f"  Size: {file_info['size_mb']:.2f} MB ({file_info['size']} bytes)")
        print(f"  Modified: {file_info['modified_time']}")
        print(f"  Number of joints: {file_info['num_joints']}")
        print()
    
    # Process each motion
    successful = 0
    failed = 0
    
    import sys
    from io import StringIO
    
    for motion_idx in range(num_motions):
        if not silent:
            print(f"\n{'=' * 80}")
            print(f"Processing Motion {motion_idx + 1}/{num_motions}")
            print(f"{'=' * 80}")
        
        old_stdout = None
        try:
            # Create save path for this motion
            motion_save_path = os.path.join(save_dir, f"motion_{motion_idx:04d}_position_comparison.png")
            
            # Temporarily redirect output if silent mode
            if silent:
                old_stdout = sys.stdout
                sys.stdout = StringIO()
            
            # Plot this motion
            plot_position_comparison(
                npz_path=npz_path,
                motion_idx=motion_idx,
                save_path=motion_save_path,
                sample_rate=sample_rate,
                max_joints_per_figure=max_joints_per_figure
            )
            
            if silent and old_stdout is not None:
                sys.stdout = old_stdout
            
            successful += 1
            if not silent:
                print(f"✅ Successfully processed motion {motion_idx}")
            
        except Exception as e:
            if silent and old_stdout is not None:
                sys.stdout = old_stdout
            failed += 1
            print(f"❌ Failed to process motion {motion_idx}: {e}")
            continue
    
    # Summary
    print()
    print("=" * 80)
    print("📊 Processing Summary")
    print("=" * 80)
    print(f"Total motions: {num_motions}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Save directory: {save_dir}")
    print("=" * 80)


def main():
    """Main function"""
    import argparse
    
    # Default file path
    default_path = "/home/wanhe/GR-3/gaponet/source/sim2real/sim2real/tasks/humanoid_operator/motions/motion_amass/fourior/merged_50Hz_31_10_payload.npz"
    
    parser = argparse.ArgumentParser(description='Analyze NPZ file data structure and plot position comparison')
    parser.add_argument('npz_path', nargs='?', default=default_path,
                       help='NPZ file path (default: %(default)s)')
    parser.add_argument('--motion-idx', type=int, default=None,
                       help='Motion index to plot (default: None, plot all motions)')
    parser.add_argument('--save-path', type=str, default=None,
                       help='Path to save figures (default: auto-generated in data_process/plots)')
    parser.add_argument('--save-dir', type=str, default=None,
                       help='Base directory to save plots (default: /home/wanhe/GR-3/gaponet/data_process/plots)')
    parser.add_argument('--sample-rate', type=float, default=50.0,
                       help='Sampling rate in Hz (default: 50.0)')
    parser.add_argument('--max-joints', type=int, default=8,
                       help='Maximum joints per figure (default: 8)')
    parser.add_argument('--analyze-only', action='store_true',
                       help='Only analyze, do not plot (default: plot all motions)')
    parser.add_argument('--silent', action='store_true',
                       help='Suppress individual motion plotting output when plotting all motions')
    
    args = parser.parse_args()
    
    npz_path = args.npz_path
    
    # Execute analysis
    if not args.analyze_only:
        data = analyze_npz_file(npz_path)
    else:
        data = None
    
    # Plotting logic
    if not args.analyze_only:
        if args.motion_idx is not None:
            # Plot single motion
            plot_position_comparison(
                npz_path=npz_path,
                motion_idx=args.motion_idx,
                save_path=args.save_path,
                sample_rate=args.sample_rate,
                max_joints_per_figure=args.max_joints
            )
        else:
            # Plot all motions (default behavior)
            plot_all_motions(
                npz_path=npz_path,
                base_save_dir=args.save_dir,
                sample_rate=args.sample_rate,
                max_joints_per_figure=args.max_joints,
                silent=args.silent
            )


if __name__ == "__main__":
    main()

