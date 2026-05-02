import os
import time
import random
import numpy as np
import logging
import argparse
import shutil

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from util import config
from util.s3dis import S3DIS
from util.common_util import AverageMeter, intersectionAndUnionGPU, find_free_port
from util.data_util import collate_fn
from util import transform as t
from util.lovasz_loss import LovaszLoss

# 全局变量用于记录训练历史
train_history = {
    'epochs': [],
    'train_loss': [],
    'train_loss_ce': [],
    'train_loss_lovasz': [],
    'train_mIoU': [],
    'train_mRecall': [],
    'train_mPrecision': [],
    'val_loss': [],
    'val_loss_ce': [],
    'val_loss_lovasz': [],
    'val_mIoU': [],
    'val_mRecall': [],
    'val_mPrecision': [],
    'val_iou_per_class': [],
    'val_recall_per_class': [],
    'val_precision_per_class': [],
}


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Point Cloud Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/pointtransformer.yaml', help='config file')
    parser.add_argument('opts', help='see config/pointtransformer.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def worker_init_fn(worker_id):
    random.seed(args.manual_seed + worker_id)


def main_process():
    return not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)


def plot_training_curves(save_path, history):
    """
    绘制训练和验证的损失曲线及指标曲线
    """
    epochs = history['epochs']
    if len(epochs) == 0:
        return
    
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    plt.rcParams['axes.unicode_minus'] = False
    
    plot_dir = os.path.join(save_path, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    # ============ 图1: 损失曲线 ============
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    ax1 = axes[0]
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2, marker='o', markersize=3)
    if history['val_loss']:
        val_epochs = [epochs[i] for i in range(len(epochs)) if i < len(history['val_loss'])]
        ax1.plot(val_epochs, history['val_loss'][:len(val_epochs)], 'r-', label='Val Loss', linewidth=2, marker='s', markersize=3)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Total Loss', fontsize=12)
    ax1.set_title('Total Loss Curve', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    ax2.plot(epochs, history['train_loss_ce'], 'b-', label='Train CE Loss', linewidth=2, marker='o', markersize=3)
    if history['val_loss_ce']:
        val_epochs = [epochs[i] for i in range(len(epochs)) if i < len(history['val_loss_ce'])]
        ax2.plot(val_epochs, history['val_loss_ce'][:len(val_epochs)], 'r-', label='Val CE Loss', linewidth=2, marker='s', markersize=3)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('CE Loss', fontsize=12)
    ax2.set_title('CrossEntropy Loss Curve', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[2]
    ax3.plot(epochs, history['train_loss_lovasz'], 'b-', label='Train Lovasz Loss', linewidth=2, marker='o', markersize=3)
    if history['val_loss_lovasz']:
        val_epochs = [epochs[i] for i in range(len(epochs)) if i < len(history['val_loss_lovasz'])]
        ax3.plot(val_epochs, history['val_loss_lovasz'][:len(val_epochs)], 'r-', label='Val Lovasz Loss', linewidth=2, marker='s', markersize=3)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Lovasz Loss', fontsize=12)
    ax3.set_title('Lovasz Loss Curve', fontsize=14, fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    loss_fig_path = os.path.join(plot_dir, 'loss_curves.png')
    plt.savefig(loss_fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # ============ 图2: 指标曲线 (mIoU, mRecall, mPrecision) ============
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    ax1 = axes[0]
    ax1.plot(epochs, history['train_mIoU'], 'b-', label='Train mIoU', linewidth=2, marker='o', markersize=3)
    if history['val_mIoU']:
        val_epochs = [epochs[i] for i in range(len(epochs)) if i < len(history['val_mIoU'])]
        ax1.plot(val_epochs, history['val_mIoU'][:len(val_epochs)], 'r-', label='Val mIoU', linewidth=2, marker='s', markersize=3)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('mIoU', fontsize=12)
    ax1.set_title('Mean IoU Curve', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    ax2 = axes[1]
    ax2.plot(epochs, history['train_mRecall'], 'b-', label='Train mRecall', linewidth=2, marker='o', markersize=3)
    if history['val_mRecall']:
        val_epochs = [epochs[i] for i in range(len(epochs)) if i < len(history['val_mRecall'])]
        ax2.plot(val_epochs, history['val_mRecall'][:len(val_epochs)], 'r-', label='Val mRecall', linewidth=2, marker='s', markersize=3)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('mRecall', fontsize=12)
    ax2.set_title('Mean Recall Curve', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    ax3 = axes[2]
    ax3.plot(epochs, history['train_mPrecision'], 'b-', label='Train mPrecision', linewidth=2, marker='o', markersize=3)
    if history['val_mPrecision']:
        val_epochs = [epochs[i] for i in range(len(epochs)) if i < len(history['val_mPrecision'])]
        ax3.plot(val_epochs, history['val_mPrecision'][:len(val_epochs)], 'r-', label='Val mPrecision', linewidth=2, marker='s', markersize=3)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('mPrecision', fontsize=12)
    ax3.set_title('Mean Precision Curve', fontsize=14, fontweight='bold')
    ax3.legend(loc='lower right')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1])
    
    plt.tight_layout()
    metrics_fig_path = os.path.join(plot_dir, 'metrics_curves.png')
    plt.savefig(metrics_fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # ============ 图3: 综合图（2x2布局） ============
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
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
    
    ax = axes[1, 0]
    ax.plot(epochs, history['train_mRecall'], 'b-', label='Train', linewidth=2)
    if history['val_mRecall']:
        val_epochs = [epochs[i] for i in range(len(epochs)) if i < len(history['val_mRecall'])]
        ax.plot(val_epochs, history['val_mRecall'][:len(val_epochs)], 'r-', label='Val', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('mRecall')
    ax.set_title('Mean Recall', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    ax = axes[1, 1]
    ax.plot(epochs, history['train_mPrecision'], 'b-', label='Train', linewidth=2)
    if history['val_mPrecision']:
        val_epochs = [epochs[i] for i in range(len(epochs)) if i < len(history['val_mPrecision'])]
        ax.plot(val_epochs, history['val_mPrecision'][:len(val_epochs)], 'r-', label='Val', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('mPrecision')
    ax.set_title('Mean Precision', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    summary_fig_path = os.path.join(plot_dir, 'training_summary.png')
    plt.savefig(summary_fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    history_path = os.path.join(plot_dir, 'training_history.npy')
    np.save(history_path, history)
    
    csv_path = os.path.join(plot_dir, 'training_history.csv')
    save_history_to_csv(history, csv_path)
    
    return loss_fig_path, metrics_fig_path, summary_fig_path


def save_history_to_csv(history, csv_path):
    """
    将训练历史保存为CSV文件，包含每个类别的IoU、Recall、Precision
    """
    import csv
    
    epochs = history['epochs']
    if len(epochs) == 0:
        return
    
    rows = []
    for i, epoch in enumerate(epochs):
        row = {
            'epoch': epoch,
            'train_loss': history['train_loss'][i] if i < len(history['train_loss']) else '',
            'train_loss_ce': history['train_loss_ce'][i] if i < len(history['train_loss_ce']) else '',
            'train_loss_lovasz': history['train_loss_lovasz'][i] if i < len(history['train_loss_lovasz']) else '',
            'train_mIoU': history['train_mIoU'][i] if i < len(history['train_mIoU']) else '',
            'train_mRecall': history['train_mRecall'][i] if i < len(history['train_mRecall']) else '',
            'train_mPrecision': history['train_mPrecision'][i] if i < len(history['train_mPrecision']) else '',
            'val_loss': history['val_loss'][i] if i < len(history['val_loss']) else '',
            'val_loss_ce': history['val_loss_ce'][i] if i < len(history['val_loss_ce']) else '',
            'val_loss_lovasz': history['val_loss_lovasz'][i] if i < len(history['val_loss_lovasz']) else '',
            'val_mIoU': history['val_mIoU'][i] if i < len(history['val_mIoU']) else '',
            'val_mRecall': history['val_mRecall'][i] if i < len(history['val_mRecall']) else '',
            'val_mPrecision': history['val_mPrecision'][i] if i < len(history['val_mPrecision']) else '',
        }
        
        # 添加每个类别的IoU、Recall、Precision
        if 'val_iou_per_class' in history and i < len(history['val_iou_per_class']):
            iou_array = history['val_iou_per_class'][i]
            for class_idx, iou_val in enumerate(iou_array):
                row[f'val_iou_class_{class_idx}'] = float(iou_val)
        
        if 'val_recall_per_class' in history and i < len(history['val_recall_per_class']):
            recall_array = history['val_recall_per_class'][i]
            for class_idx, recall_val in enumerate(recall_array):
                row[f'val_recall_class_{class_idx}'] = float(recall_val)
        
        if 'val_precision_per_class' in history and i < len(history['val_precision_per_class']):
            precision_array = history['val_precision_per_class'][i]
            for class_idx, precision_val in enumerate(precision_array):
                row[f'val_precision_class_{class_idx}'] = float(precision_val)
        
        rows.append(row)
    
    fieldnames = ['epoch', 'train_loss', 'train_loss_ce', 'train_loss_lovasz', 
                  'train_mIoU', 'train_mRecall', 'train_mPrecision',
                  'val_loss', 'val_loss_ce', 'val_loss_lovasz',
                  'val_mIoU', 'val_mRecall', 'val_mPrecision']
    
    # 添加类别IoU、Recall、Precision的列名
    if 'val_iou_per_class' in history and len(history['val_iou_per_class']) > 0:
        num_classes = len(history['val_iou_per_class'][0])
        for i in range(num_classes):
            fieldnames.append(f'val_iou_class_{i}')
    
    if 'val_recall_per_class' in history and len(history['val_recall_per_class']) > 0:
        num_classes = len(history['val_recall_per_class'][0])
        for i in range(num_classes):
            fieldnames.append(f'val_recall_class_{i}')
    
    if 'val_precision_per_class' in history and len(history['val_precision_per_class']) > 0:
        num_classes = len(history['val_precision_per_class'][0])
        for i in range(num_classes):
            fieldnames.append(f'val_precision_class_{i}')
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval='')
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)

    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False

    if args.data_name == 's3dis':
        S3DIS(split='train', data_root=args.data_root, test_area=args.test_area)
        S3DIS(split='val', data_root=args.data_root, test_area=args.test_area)
    else:
        raise NotImplementedError()
    if args.multiprocessing_distributed:
        port = find_free_port()
        args.dist_url = f"tcp://localhost:{port}"
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, argss):
    global args, best_iou, best_results
    args, best_iou = argss, 0
    best_results = {
        'epoch': 0,
        'train_loss': 0, 'train_loss_ce': 0, 'train_loss_lovasz': 0,
        'train_mIoU': 0, 'train_mRecall': 0, 'train_mPrecision': 0,
        'val_loss': 0, 'val_loss_ce': 0, 'val_loss_lovasz': 0,
        'val_mIoU': 0, 'val_mRecall': 0, 'val_mPrecision': 0,
    }
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    if args.arch == 'pointtransformer_v3':
        from model.pointtransformer_v3 import PointTransformerV3 as Model
    else:
        raise Exception('architecture not supported yet'.format(args.arch))
    
    model_kwargs = {
        'in_channels': args.fea_dim,
        'n_cls': args.classes
    }
    if hasattr(args, 'enable_ema'):
        model_kwargs['enable_ema'] = args.enable_ema
    if hasattr(args, 'ema_factor'):
        model_kwargs['ema_factor'] = args.ema_factor
    if hasattr(args, 'ema_stages'):
        model_kwargs['ema_stages'] = args.ema_stages
    if hasattr(args, 'ema_fusion_weight'):
        model_kwargs['ema_fusion_weight'] = args.ema_fusion_weight
    model = Model(**model_kwargs)
    if args.sync_bn:
       model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    class_weights = torch.FloatTensor([10.0, 10.0, 1]).cuda()
    
    criterion_ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=args.ignore_label).cuda()
    criterion_lovasz = LovaszLoss(mode="multiclass", ignore_index=args.ignore_label).cuda()
    
    criteria_config = [
        {"criterion": criterion_ce, "loss_weight": 1.0},
        {"criterion": criterion_lovasz, "loss_weight": 1.0}
    ]

    optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.epochs*0.6), int(args.epochs*0.8)], gamma=0.1)

    if main_process():
        global logger, writer
        logger = get_logger()
        writer = SummaryWriter(args.save_path)
        logger.info(args)
        logger.info("=> creating model ...")
        logger.info("Classes: {}".format(args.classes))
        logger.info("Class weights: {}".format(class_weights))
        logger.info("Loss functions: CrossEntropyLoss (weight=1.0) + LovaszLoss (weight=1.0)")
        logger.info(model)
    if args.distributed:
        torch.cuda.set_device(gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.batch_size_val = int(args.batch_size_val / ngpus_per_node)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(
            model.cuda(),
            device_ids=[gpu],
            find_unused_parameters=True if "transformer" in args.arch else False
        )

    else:
        model = torch.nn.DataParallel(model.cuda())

    if args.weight:
        if os.path.isfile(args.weight):
            if main_process():
                logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight)
            model.load_state_dict(checkpoint['state_dict'])
            if main_process():
                logger.info("=> loaded weight '{}'".format(args.weight))
        else:
            logger.info("=> no weight found at '{}'".format(args.weight))

    if args.resume:
        if os.path.isfile(args.resume):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            best_iou = checkpoint['best_iou']
            if main_process():
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            if main_process():
                logger.info("=> no checkpoint found at '{}'".format(args.resume))

    train_transform = t.Compose([t.RandomScale([0.9, 1.1]), t.ChromaticAutoContrast(), t.ChromaticTranslation(), t.ChromaticJitter(), t.HueSaturationTranslation()])
    train_data = S3DIS(split='train', data_root=args.data_root, test_area=args.test_area, voxel_size=args.voxel_size, voxel_max=args.voxel_max, transform=train_transform, shuffle_index=True, loop=args.loop)
    if main_process():
            logger.info("train_data samples: '{}'".format(len(train_data)))
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True, collate_fn=collate_fn)

    val_loader = None
    if args.evaluate:
        val_transform = None
        val_data = S3DIS(split='val', data_root=args.data_root, test_area=args.test_area, voxel_size=args.voxel_size, voxel_max=800000, transform=val_transform)
        if args.distributed:
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
        else:
            val_sampler = None
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False, num_workers=args.workers, pin_memory=True, sampler=val_sampler, collate_fn=collate_fn)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        loss_train, loss_ce_train, loss_lovasz_train, mIoU_train, mRecall_train, mPrecision_train = train(train_loader, model, criteria_config, optimizer, epoch)
        scheduler.step()
        epoch_log = epoch + 1
        if main_process():
            writer.add_scalar('loss_train', loss_train, epoch_log)
            writer.add_scalar('mIoU_train', mIoU_train, epoch_log)
            writer.add_scalar('mRecall_train', mRecall_train, epoch_log)
            writer.add_scalar('mPrecision_train', mPrecision_train, epoch_log)
            
            train_history['epochs'].append(epoch_log)
            train_history['train_loss'].append(loss_train)
            train_history['train_loss_ce'].append(loss_ce_train)
            train_history['train_loss_lovasz'].append(loss_lovasz_train)
            train_history['train_mIoU'].append(mIoU_train)
            train_history['train_mRecall'].append(mRecall_train)
            train_history['train_mPrecision'].append(mPrecision_train)

        is_best = False
        if args.evaluate and (epoch_log % args.eval_freq == 0):
            if args.data_name == 'shapenet':
                raise NotImplementedError()
            else:
                loss_val, loss_ce_val, loss_lovasz_val, mIoU_val, mRecall_val, mPrecision_val, iou_class_val, recall_class_val, precision_class_val = validate(val_loader, model, criteria_config)
            if main_process():
                writer.add_scalar('loss_val', loss_val, epoch_log)
                writer.add_scalar('mIoU_val', mIoU_val, epoch_log)
                writer.add_scalar('mRecall_val', mRecall_val, epoch_log)
                writer.add_scalar('mPrecision_val', mPrecision_val, epoch_log)
                is_best = mIoU_val > best_iou
                best_iou = max(best_iou, mIoU_val)
                
                if is_best:
                    best_results['epoch'] = epoch_log
                    best_results['train_loss'] = loss_train
                    best_results['train_loss_ce'] = loss_ce_train
                    best_results['train_loss_lovasz'] = loss_lovasz_train
                    best_results['train_mIoU'] = mIoU_train
                    best_results['train_mRecall'] = mRecall_train
                    best_results['train_mPrecision'] = mPrecision_train
                    best_results['val_loss'] = loss_val
                    best_results['val_loss_ce'] = loss_ce_val
                    best_results['val_loss_lovasz'] = loss_lovasz_val
                    best_results['val_mIoU'] = mIoU_val
                    best_results['val_mRecall'] = mRecall_val
                    best_results['val_mPrecision'] = mPrecision_val
                
                train_history['val_loss'].append(loss_val)
                train_history['val_loss_ce'].append(loss_ce_val)
                train_history['val_loss_lovasz'].append(loss_lovasz_val)
                train_history['val_mIoU'].append(mIoU_val)
                train_history['val_mRecall'].append(mRecall_val)
                train_history['val_mPrecision'].append(mPrecision_val)
                train_history['val_iou_per_class'].append(iou_class_val)
                train_history['val_recall_per_class'].append(recall_class_val)
                train_history['val_precision_per_class'].append(precision_class_val)

        if (epoch_log % args.save_freq == 0) and main_process():
            filename = args.save_path + '/model/model_last.pth'
            logger.info('Saving checkpoint to: ' + filename)
            torch.save({'epoch': epoch_log, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(), 'best_iou': best_iou, 'is_best': is_best}, filename)
            if is_best:
                logger.info('Best validation mIoU updated to: {:.4f}'.format(best_iou))
                shutil.copyfile(filename, args.save_path + '/model/model_best.pth')
        
        if main_process() and len(train_history['epochs']) > 0:
            try:
                plot_training_curves(args.save_path, train_history)
            except Exception as e:
                logger.warning(f'Failed to plot training curves: {e}')

    if main_process():
        writer.close()
        
        logger.info('=' * 80)
        logger.info('Training Completed!')
        logger.info('=' * 80)
        logger.info('')
        logger.info('>>>>>>>>>>>>>> Best Results (Epoch {}) <<<<<<<<<<<<<<'.format(best_results['epoch']))
        logger.info('')
        logger.info('  [Train Metrics]')
        logger.info('    - Loss:       {:.4f} (CE: {:.4f}, Lovasz: {:.4f})'.format(
            best_results['train_loss'], best_results['train_loss_ce'], best_results['train_loss_lovasz']))
        logger.info('    - mIoU:       {:.4f}'.format(best_results['train_mIoU']))
        logger.info('    - mRecall:    {:.4f}'.format(best_results['train_mRecall']))
        logger.info('    - mPrecision: {:.4f}'.format(best_results['train_mPrecision']))
        logger.info('')
        logger.info('  [Validation Metrics]')
        logger.info('    - Loss:       {:.4f} (CE: {:.4f}, Lovasz: {:.4f})'.format(
            best_results['val_loss'], best_results['val_loss_ce'], best_results['val_loss_lovasz']))
        logger.info('    - mIoU:       {:.4f}  <-- Best'.format(best_results['val_mIoU']))
        logger.info('    - mRecall:    {:.4f}'.format(best_results['val_mRecall']))
        logger.info('    - mPrecision: {:.4f}'.format(best_results['val_mPrecision']))
        logger.info('')
        logger.info('  [Model Saved]')
        logger.info('    - Best model:  {}/model/model_best.pth'.format(args.save_path))
        logger.info('    - Last model:  {}/model/model_last.pth'.format(args.save_path))
        logger.info('')
        logger.info('=' * 80)
        
        try:
            loss_fig, metrics_fig, summary_fig = plot_training_curves(args.save_path, train_history)
            logger.info('Training curves saved to:')
            logger.info(f'  - Loss curves: {loss_fig}')
            logger.info(f'  - Metrics curves: {metrics_fig}')
            logger.info(f'  - Summary: {summary_fig}')
        except Exception as e:
            logger.warning(f'Failed to plot final training curves: {e}')
        
        best_results_path = os.path.join(args.save_path, 'best_results.txt')
        with open(best_results_path, 'w') as f:
            f.write('Best Results Summary\n')
            f.write('=' * 60 + '\n')
            f.write(f'Best Epoch: {best_results["epoch"]}\n\n')
            f.write('[Train Metrics]\n')
            f.write(f'  Loss:       {best_results["train_loss"]:.4f}\n')
            f.write(f'  Loss CE:    {best_results["train_loss_ce"]:.4f}\n')
            f.write(f'  Loss Lovasz: {best_results["train_loss_lovasz"]:.4f}\n')
            f.write(f'  mIoU:       {best_results["train_mIoU"]:.4f}\n')
            f.write(f'  mRecall:    {best_results["train_mRecall"]:.4f}\n')
            f.write(f'  mPrecision: {best_results["train_mPrecision"]:.4f}\n\n')
            f.write('[Validation Metrics]\n')
            f.write(f'  Loss:       {best_results["val_loss"]:.4f}\n')
            f.write(f'  Loss CE:    {best_results["val_loss_ce"]:.4f}\n')
            f.write(f'  Loss Lovasz: {best_results["val_loss_lovasz"]:.4f}\n')
            f.write(f'  mIoU:       {best_results["val_mIoU"]:.4f}\n')
            f.write(f'  mRecall:    {best_results["val_mRecall"]:.4f}\n')
            f.write(f'  mPrecision: {best_results["val_mPrecision"]:.4f}\n')
        logger.info(f'Best results saved to: {best_results_path}')


def train(train_loader, model, criteria_config, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    loss_ce_meter = AverageMeter()
    loss_lovasz_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)
    for i, (coord, feat, target, offset) in enumerate(train_loader):
        data_time.update(time.time() - end)
        coord, feat, target, offset = coord.cuda(non_blocking=True), feat.cuda(non_blocking=True), target.cuda(non_blocking=True), offset.cuda(non_blocking=True)
        output = model([coord, feat, offset])
        if target.shape[-1] == 1:
            target = target[:, 0]
        
        loss_ce = criteria_config[0]["criterion"](output, target) * criteria_config[0]["loss_weight"]
        loss_lovasz = criteria_config[1]["criterion"](output, target) * criteria_config[1]["loss_weight"]
        loss = loss_ce + loss_lovasz
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.max(1)[1]
        n = coord.size(0)
        if args.multiprocessing_distributed:
            loss *= n
            loss_ce *= n
            loss_lovasz *= n
            count = target.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss), dist.all_reduce(loss_ce), dist.all_reduce(loss_lovasz), dist.all_reduce(count)
            n = count.item()
            loss /= n
            loss_ce /= n
            loss_lovasz /= n
            
        intersection, union, target_tensor = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        if args.multiprocessing_distributed:
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target_tensor)
        intersection, union, target_tensor = intersection.cpu().numpy(), union.cpu().numpy(), target_tensor.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target_tensor)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), n)
        loss_ce_meter.update(loss_ce.item(), n)
        loss_lovasz_meter.update(loss_lovasz.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()

        current_iter = epoch * len(train_loader) + i + 1
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'Loss {loss_meter.val:.4f} (CE: {loss_ce:.4f}, Lovasz: {loss_lovasz:.4f}) '
                        'Accuracy {accuracy:.4f}.'.format(epoch+1, args.epochs, i + 1, len(train_loader),
                                                          batch_time=batch_time, data_time=data_time,
                                                          remain_time=remain_time,
                                                          loss_meter=loss_meter,
                                                          loss_ce=loss_ce_meter.val,
                                                          loss_lovasz=loss_lovasz_meter.val,
                                                          accuracy=accuracy))
        if main_process():
            writer.add_scalar('loss_train_batch', loss_meter.val, current_iter)
            writer.add_scalar('loss_ce_train_batch', loss_ce_meter.val, current_iter)
            writer.add_scalar('loss_lovasz_train_batch', loss_lovasz_meter.val, current_iter)

    # ============ 指标计算公式 ============
    # 数据含义：
    # intersection = TP（真正例：预测对的点数）
    # union = TP + FP + FN（预测为该类或实际为该类的所有点）
    # target = TP + FN（实际为该类的所有点）
    
    # IoU = TP / (TP + FP + FN)
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    
    # Recall = TP / (TP + FN)
    recall_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    
    # Precision = TP / (TP + FP)，其中 FP = union - target
    # 推导：FP = (TP + FP + FN) - (TP + FN) = union - target
    precision_class = intersection_meter.sum / (intersection_meter.sum + union_meter.sum - target_meter.sum + 1e-10)
    # ========================================
    
    mIoU = np.mean(iou_class)
    mRecall = np.mean(recall_class)
    mPrecision = np.mean(precision_class)
    
    if main_process():
        logger.info('Train result at epoch [{}/{}]: mIoU/mRecall/mPrecision {:.4f}/{:.4f}/{:.4f}.'.format(
            epoch+1, args.epochs, mIoU, mRecall, mPrecision))
        logger.info('Train losses: Total {:.4f}, CE {:.4f}, Lovasz {:.4f}'.format(loss_meter.avg, loss_ce_meter.avg, loss_lovasz_meter.avg))
    return loss_meter.avg, loss_ce_meter.avg, loss_lovasz_meter.avg, mIoU, mRecall, mPrecision


def validate(val_loader, model, criteria_config):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    loss_ce_meter = AverageMeter()
    loss_lovasz_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.eval()
    end = time.time()
    for i, (coord, feat, target, offset) in enumerate(val_loader):
        data_time.update(time.time() - end)
        coord, feat, target, offset = coord.cuda(non_blocking=True), feat.cuda(non_blocking=True), target.cuda(non_blocking=True), offset.cuda(non_blocking=True)
        if target.shape[-1] == 1:
            target = target[:, 0]
        with torch.no_grad():
            output = model([coord, feat, offset])
        
        loss_ce = criteria_config[0]["criterion"](output, target) * criteria_config[0]["loss_weight"]
        loss_lovasz = criteria_config[1]["criterion"](output, target) * criteria_config[1]["loss_weight"]
        loss = loss_ce + loss_lovasz

        output = output.max(1)[1]
        n = coord.size(0)
        if args.multiprocessing_distributed:
            loss *= n
            loss_ce *= n
            loss_lovasz *= n
            count = target.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss), dist.all_reduce(loss_ce), dist.all_reduce(loss_lovasz), dist.all_reduce(count)
            n = count.item()
            loss /= n
            loss_ce /= n
            loss_lovasz /= n

        intersection, union, target_tensor = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        if args.multiprocessing_distributed:
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target_tensor)
        intersection, union, target_tensor = intersection.cpu().numpy(), union.cpu().numpy(), target_tensor.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target_tensor)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), n)
        loss_ce_meter.update(loss_ce.item(), n)
        loss_lovasz_meter.update(loss_lovasz.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                        'Accuracy {accuracy:.4f}.'.format(i + 1, len(val_loader),
                                                          data_time=data_time,
                                                          batch_time=batch_time,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy))

    # ============ 指标计算公式 ============
    # 数据含义：
    # intersection = TP（真正例：预测对的点数）
    # union = TP + FP + FN（预测为该类或实际为该类的所有点）
    # target = TP + FN（实际为该类的所有点）
    
    # IoU = TP / (TP + FP + FN)
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    
    # Recall = TP / (TP + FN)
    recall_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    
    # Precision = TP / (TP + FP)，其中 FP = union - target
    # 推导：FP = (TP + FP + FN) - (TP + FN) = union - target
    precision_class = intersection_meter.sum / (intersection_meter.sum + union_meter.sum - target_meter.sum + 1e-10)
    # ========================================
    
    mIoU = np.mean(iou_class)
    mRecall = np.mean(recall_class)
    mPrecision = np.mean(precision_class)

    if main_process():
        logger.info('Val result: mIoU/mRecall/mPrecision {:.4f}/{:.4f}/{:.4f}.'.format(
            mIoU, mRecall, mPrecision))
        logger.info('Val losses: Total {:.4f}, CE {:.4f}, Lovasz {:.4f}'.format(loss_meter.avg, loss_ce_meter.avg, loss_lovasz_meter.avg))
        for i in range(args.classes):
            logger.info('Class_{} Result: iou/recall/precision {:.4f}/{:.4f}/{:.4f}.'.format(
                i, iou_class[i], recall_class[i], precision_class[i]))
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')

    return loss_meter.avg, loss_ce_meter.avg, loss_lovasz_meter.avg, mIoU, mRecall, mPrecision, iou_class, recall_class, precision_class


if __name__ == '__main__':
    import gc
    gc.collect()
    main()