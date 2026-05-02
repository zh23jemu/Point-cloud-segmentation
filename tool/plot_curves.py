#!/usr/bin/env python
"""
训练曲线绘制脚本
可以在训练结束后单独运行，根据保存的训练历史数据绘制曲线图
也可以从TensorBoard日志中读取数据绘制曲线

使用方法:
    1. 从训练历史文件绘制:
       python tool/plot_curves.py --history exp/seg/pointtransformer_v3/plots/training_history.npy
    
    2. 从TensorBoard日志绘制:
       python tool/plot_curves.py --logdir exp/seg/pointtransformer_v3
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def plot_from_history(history_path, output_dir=None):
    """从训练历史文件绘制曲线"""
    print(f"Loading history from: {history_path}")
    history = np.load(history_path, allow_pickle=True).item()
    
    if output_dir is None:
        output_dir = os.path.dirname(history_path)
    os.makedirs(output_dir, exist_ok=True)
    
    epochs = history['epochs']
    if len(epochs) == 0:
        print("No training history found!")
        return
    
    print(f"Found {len(epochs)} epochs of training data")
    
    # 设置样式
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    
    # ============ 图1: 损失曲线 ============
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 总损失
    ax1 = axes[0]
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2, marker='o', markersize=3)
    if history['val_loss']:
        val_epochs = [epochs[i] for i in range(len(epochs)) if i < len(history['val_loss'])]
        ax1.plot(val_epochs, history['val_loss'][:len(val_epochs)], 'r-', label='Val Loss', linewidth=2, marker='s', markersize=3)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Total Loss')
    ax1.set_title('Total Loss Curve', fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # CE损失
    ax2 = axes[1]
    ax2.plot(epochs, history['train_loss_ce'], 'b-', label='Train CE Loss', linewidth=2, marker='o', markersize=3)
    if history['val_loss_ce']:
        val_epochs = [epochs[i] for i in range(len(epochs)) if i < len(history['val_loss_ce'])]
        ax2.plot(val_epochs, history['val_loss_ce'][:len(val_epochs)], 'r-', label='Val CE Loss', linewidth=2, marker='s', markersize=3)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('CE Loss')
    ax2.set_title('CrossEntropy Loss Curve', fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Lovasz损失
    ax3 = axes[2]
    ax3.plot(epochs, history['train_loss_lovasz'], 'b-', label='Train Lovasz Loss', linewidth=2, marker='o', markersize=3)
    if history['val_loss_lovasz']:
        val_epochs = [epochs[i] for i in range(len(epochs)) if i < len(history['val_loss_lovasz'])]
        ax3.plot(val_epochs, history['val_loss_lovasz'][:len(val_epochs)], 'r-', label='Val Lovasz Loss', linewidth=2, marker='s', markersize=3)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Lovasz Loss')
    ax3.set_title('Lovasz Loss Curve', fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    loss_fig_path = os.path.join(output_dir, 'loss_curves.png')
    plt.savefig(loss_fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {loss_fig_path}")
    
    # ============ 图2: 指标曲线 ============
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # mIoU
    ax1 = axes[0]
    ax1.plot(epochs, history['train_mIoU'], 'b-', label='Train mIoU', linewidth=2, marker='o', markersize=3)
    if history['val_mIoU']:
        val_epochs = [epochs[i] for i in range(len(epochs)) if i < len(history['val_mIoU'])]
        ax1.plot(val_epochs, history['val_mIoU'][:len(val_epochs)], 'r-', label='Val mIoU', linewidth=2, marker='s', markersize=3)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('mIoU')
    ax1.set_title('Mean IoU Curve', fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # mAcc
    ax2 = axes[1]
    ax2.plot(epochs, history['train_mAcc'], 'b-', label='Train mAcc', linewidth=2, marker='o', markersize=3)
    if history['val_mAcc']:
        val_epochs = [epochs[i] for i in range(len(epochs)) if i < len(history['val_mAcc'])]
        ax2.plot(val_epochs, history['val_mAcc'][:len(val_epochs)], 'r-', label='Val mAcc', linewidth=2, marker='s', markersize=3)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('mAcc')
    ax2.set_title('Mean Accuracy Curve', fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    # allAcc
    ax3 = axes[2]
    ax3.plot(epochs, history['train_allAcc'], 'b-', label='Train allAcc', linewidth=2, marker='o', markersize=3)
    if history['val_allAcc']:
        val_epochs = [epochs[i] for i in range(len(epochs)) if i < len(history['val_allAcc'])]
        ax3.plot(val_epochs, history['val_allAcc'][:len(val_epochs)], 'r-', label='Val allAcc', linewidth=2, marker='s', markersize=3)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Overall Accuracy')
    ax3.set_title('Overall Accuracy Curve', fontweight='bold')
    ax3.legend(loc='lower right')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1])
    
    plt.tight_layout()
    metrics_fig_path = os.path.join(output_dir, 'metrics_curves.png')
    plt.savefig(metrics_fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {metrics_fig_path}")
    
    # ============ 图3: 综合图 ============
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 总损失
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    if history['val_loss']:
        val_epochs = [epochs[i] for i in range(len(epochs)) if i < len(history['val_loss'])]
        ax.plot(val_epochs, history['val_loss'][:len(val_epochs)], 'r-', label='Val', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Total Loss', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # mIoU
    ax = axes[0, 1]
    ax.plot(epochs, history['train_mIoU'], 'b-', label='Train', linewidth=2)
    if history['val_mIoU']:
        val_epochs = [epochs[i] for i in range(len(epochs)) if i < len(history['val_mIoU'])]
        ax.plot(val_epochs, history['val_mIoU'][:len(val_epochs)], 'r-', label='Val', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('mIoU')
    ax.set_title('Mean IoU', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # CE & Lovasz Loss
    ax = axes[1, 0]
    ax.plot(epochs, history['train_loss_ce'], 'b-', label='Train CE', linewidth=2)
    ax.plot(epochs, history['train_loss_lovasz'], 'b--', label='Train Lovasz', linewidth=2)
    if history['val_loss_ce']:
        val_epochs = [epochs[i] for i in range(len(epochs)) if i < len(history['val_loss_ce'])]
        ax.plot(val_epochs, history['val_loss_ce'][:len(val_epochs)], 'r-', label='Val CE', linewidth=2)
    if history['val_loss_lovasz']:
        val_epochs = [epochs[i] for i in range(len(epochs)) if i < len(history['val_loss_lovasz'])]
        ax.plot(val_epochs, history['val_loss_lovasz'][:len(val_epochs)], 'r--', label='Val Lovasz', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('CE & Lovasz Loss', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Accuracy
    ax = axes[1, 1]
    ax.plot(epochs, history['train_mAcc'], 'b-', label='Train mAcc', linewidth=2)
    ax.plot(epochs, history['train_allAcc'], 'b--', label='Train allAcc', linewidth=2)
    if history['val_mAcc']:
        val_epochs = [epochs[i] for i in range(len(epochs)) if i < len(history['val_mAcc'])]
        ax.plot(val_epochs, history['val_mAcc'][:len(val_epochs)], 'r-', label='Val mAcc', linewidth=2)
    if history['val_allAcc']:
        val_epochs = [epochs[i] for i in range(len(epochs)) if i < len(history['val_allAcc'])]
        ax.plot(val_epochs, history['val_allAcc'][:len(val_epochs)], 'r--', label='Val allAcc', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy Metrics', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    summary_fig_path = os.path.join(output_dir, 'training_summary.png')
    plt.savefig(summary_fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {summary_fig_path}")
    
    print("\nAll plots saved successfully!")


def plot_from_tensorboard(logdir, output_dir=None):
    """从TensorBoard日志文件绘制曲线"""
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        print("Error: tensorboard package not installed. Install with: pip install tensorboard")
        return
    
    if output_dir is None:
        output_dir = os.path.join(logdir, 'plots')
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找event文件
    event_files = []
    for root, dirs, files in os.walk(logdir):
        for file in files:
            if file.startswith('events.out.tfevents'):
                event_files.append(os.path.join(root, file))
    
    if not event_files:
        print(f"No TensorBoard event files found in: {logdir}")
        return
    
    print(f"Found {len(event_files)} event file(s)")
    
    # 读取所有事件数据
    history = {
        'epochs': [],
        'train_loss': [],
        'train_mIoU': [],
        'train_mAcc': [],
        'train_allAcc': [],
        'val_loss': [],
        'val_mIoU': [],
        'val_mAcc': [],
        'val_allAcc': [],
    }
    
    for event_file in event_files:
        ea = event_accumulator.EventAccumulator(event_file)
        ea.Reload()
        
        # 获取可用的标签
        scalar_tags = ea.Tags().get('scalars', [])
        print(f"Available tags: {scalar_tags}")
        
        for tag in scalar_tags:
            events = ea.Scalars(tag)
            for event in events:
                step = event.step
                value = event.value
                
                if step not in history['epochs']:
                    history['epochs'].append(step)
                
                if tag == 'loss_train':
                    history['train_loss'].append(value)
                elif tag == 'mIoU_train':
                    history['train_mIoU'].append(value)
                elif tag == 'mAcc_train':
                    history['train_mAcc'].append(value)
                elif tag == 'allAcc_train':
                    history['train_allAcc'].append(value)
                elif tag == 'loss_val':
                    history['val_loss'].append(value)
                elif tag == 'mIoU_val':
                    history['val_mIoU'].append(value)
                elif tag == 'mAcc_val':
                    history['val_mAcc'].append(value)
                elif tag == 'allAcc_val':
                    history['val_allAcc'].append(value)
    
    # 排序
    if history['epochs']:
        sorted_idx = np.argsort(history['epochs'])
        history['epochs'] = [history['epochs'][i] for i in sorted_idx]
        for key in ['train_loss', 'train_mIoU', 'train_mAcc', 'train_allAcc']:
            if history[key]:
                history[key] = [history[key][i] for i in sorted_idx if i < len(history[key])]
    
    if len(history['epochs']) == 0:
        print("No valid training data found in TensorBoard logs!")
        return
    
    # 补充缺失的keys用于兼容
    history['train_loss_ce'] = history['train_loss']  # 简化处理
    history['train_loss_lovasz'] = [0] * len(history['train_loss'])
    history['val_loss_ce'] = history['val_loss']
    history['val_loss_lovasz'] = [0] * len(history['val_loss'])
    
    # 绘制简化版曲线
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    epochs = history['epochs']
    
    # 损失
    ax = axes[0, 0]
    if history['train_loss']:
        ax.plot(epochs[:len(history['train_loss'])], history['train_loss'], 'b-', label='Train', linewidth=2)
    if history['val_loss']:
        ax.plot(epochs[:len(history['val_loss'])], history['val_loss'], 'r-', label='Val', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Curve', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # mIoU
    ax = axes[0, 1]
    if history['train_mIoU']:
        ax.plot(epochs[:len(history['train_mIoU'])], history['train_mIoU'], 'b-', label='Train', linewidth=2)
    if history['val_mIoU']:
        ax.plot(epochs[:len(history['val_mIoU'])], history['val_mIoU'], 'r-', label='Val', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('mIoU')
    ax.set_title('Mean IoU', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # mAcc
    ax = axes[1, 0]
    if history['train_mAcc']:
        ax.plot(epochs[:len(history['train_mAcc'])], history['train_mAcc'], 'b-', label='Train', linewidth=2)
    if history['val_mAcc']:
        ax.plot(epochs[:len(history['val_mAcc'])], history['val_mAcc'], 'r-', label='Val', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('mAcc')
    ax.set_title('Mean Accuracy', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # allAcc
    ax = axes[1, 1]
    if history['train_allAcc']:
        ax.plot(epochs[:len(history['train_allAcc'])], history['train_allAcc'], 'b-', label='Train', linewidth=2)
    if history['val_allAcc']:
        ax.plot(epochs[:len(history['val_allAcc'])], history['val_allAcc'], 'r-', label='Val', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Overall Accuracy')
    ax.set_title('Overall Accuracy', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    summary_fig_path = os.path.join(output_dir, 'training_summary_from_tb.png')
    plt.savefig(summary_fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {summary_fig_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot training curves')
    parser.add_argument('--history', type=str, default='', help='Path to training_history.npy file')
    parser.add_argument('--logdir', type=str, default='', help='Path to TensorBoard log directory')
    parser.add_argument('--output', type=str, default='', help='Output directory for plots')
    args = parser.parse_args()
    
    if args.history:
        output_dir = args.output if args.output else None
        plot_from_history(args.history, output_dir)
    elif args.logdir:
        output_dir = args.output if args.output else None
        plot_from_tensorboard(args.logdir, output_dir)
    else:
        print("Please specify either --history or --logdir")
        print("\nExamples:")
        print("  python tool/plot_curves.py --history exp/seg/model/plots/training_history.npy")
        print("  python tool/plot_curves.py --logdir exp/seg/model/")


if __name__ == '__main__':
    main()

