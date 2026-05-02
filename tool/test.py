import os
import time
import random
import numpy as np
import logging
import pickle
import argparse
import collections

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端

from util import config
from util.common_util import AverageMeter, intersectionAndUnion, check_makedirs
from util.voxelize import voxelize
from util.lovasz_loss import LovaszLoss  # 导入Lovasz Loss


def plot_confusion_matrix(cm, class_names, save_path, normalize=True, title='Confusion Matrix'):
    """
    绘制并保存混淆矩阵
    Args:
        cm: 混淆矩阵 (n_classes, n_classes)
        class_names: 类别名称列表
        save_path: 保存路径
        normalize: 是否归一化
        title: 图表标题
    """
    if normalize:
        cm_normalized = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)
    else:
        cm_normalized = cm
    
    n_classes = len(class_names)
    
    # 设置图形大小
    fig_size = max(8, n_classes * 0.8)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    
    # 绘制热力图
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # 设置坐标轴
    ax.set(xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           xticklabels=class_names,
           yticklabels=class_names,
           title=title,
           ylabel='True Label',
           xlabel='Predicted Label')
    
    # 旋转x轴标签
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # 在每个格子中添加数值
    fmt = '.2f' if normalize else 'd'
    thresh = cm_normalized.max() / 2.
    for i in range(n_classes):
        for j in range(n_classes):
            value = cm_normalized[i, j] if normalize else cm[i, j]
            ax.text(j, i, format(value, fmt),
                   ha="center", va="center",
                   color="white" if cm_normalized[i, j] > thresh else "black",
                   fontsize=max(6, 10 - n_classes // 3))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path


def save_confusion_matrix_data(cm, class_names, save_dir):
    """
    保存混淆矩阵数据到文件
    """
    import csv
    
    # 保存为npy
    np.save(os.path.join(save_dir, 'confusion_matrix.npy'), cm)
    
    # 保存为csv（带类别名称）
    csv_path = os.path.join(save_dir, 'confusion_matrix.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # 写入表头
        writer.writerow([''] + class_names)
        # 写入数据
        for i, name in enumerate(class_names):
            writer.writerow([name] + list(cm[i]))
    
    # 计算并保存每类的精度、召回率、F1
    metrics_path = os.path.join(save_dir, 'class_metrics.csv')
    with open(metrics_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Class', 'Precision', 'Recall', 'F1-Score', 'Support'])
        
        for i, name in enumerate(class_names):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            support = cm[i, :].sum()
            
            precision = tp / (tp + fp + 1e-10)
            recall = tp / (tp + fn + 1e-10)
            f1 = 2 * precision * recall / (precision + recall + 1e-10)
            
            writer.writerow([name, f'{precision:.4f}', f'{recall:.4f}', f'{f1:.4f}', int(support)])
    
    return csv_path, metrics_path

random.seed(123)
np.random.seed(123)

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Point Cloud Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/s3dis/pointtransformer.yaml', help='config file')
    parser.add_argument('opts', help='see config/s3dis/pointtransformer.yaml for all options', default=None, nargs=argparse.REMAINDER)
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


def main():
    global args, logger
    args = get_parser()
    logger = get_logger()
    logger.info(args)
    assert args.classes > 1
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))
    


# if args.arch == 'pointtransformer_v3':
#     model = Model(in_channels=args.fea_dim, n_cls=args.classes).cuda()
# else:
#     model = Model(c=args.fea_dim, k=args.classes).cuda()


    
    # if args.arch == 'pointtransformer_seg_repro':
    #     from model.pointtransformer.pointtransformer_seg import pointtransformer_seg_repro as Model    
    # else:
    #     raise Exception('architecture not supported yet'.format(args.arch))
    

    if args.arch == 'pointtransformer_v3':
        from model.pointtransformer_v3 import PointTransformerV3 as Model
    elif args.arch == 'pointtransformer_seg_repro':
        from model.pointtransformer.pointtransformer_seg import pointtransformer_seg_repro as Model    
    else:
        raise Exception('architecture not supported yet'.format(args.arch))

    # 传递 EMA 相关参数（如果配置文件中存在）
    model_kwargs = {
        'in_channels': args.fea_dim,
        'n_cls': args.classes
    }
    
    # 检查并传递 EMA 参数
    enable_ema = getattr(args, 'enable_ema', False)
    logger.info("EMA enabled: {}".format(enable_ema))
    if enable_ema:
        model_kwargs['enable_ema'] = enable_ema
        if hasattr(args, 'ema_factor'):
            model_kwargs['ema_factor'] = args.ema_factor
            logger.info("EMA factor: {}".format(args.ema_factor))
        if hasattr(args, 'ema_stages'):
            model_kwargs['ema_stages'] = args.ema_stages
            logger.info("EMA stages: {}".format(args.ema_stages))
        if hasattr(args, 'ema_fusion_weight'):
            model_kwargs['ema_fusion_weight'] = args.ema_fusion_weight
            logger.info("EMA fusion weight: {}".format(args.ema_fusion_weight))
    
    logger.info("Model kwargs: {}".format(model_kwargs))
    model = Model(**model_kwargs).cuda()
    logger.info(model)
    
    # ============ 修改损失函数部分 ============
    # 定义类别权重 [2, 2, 0.5, ...]，根据你的类别数量调整
    # 假设有13个类别（S3DIS数据集）
    class_weights = torch.FloatTensor([10.0, 10.0, 1]).cuda()
    
    # 定义两个损失函数
    criterion_ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=args.ignore_label).cuda()
    criterion_lovasz = LovaszLoss(mode="multiclass", ignore_index=args.ignore_label).cuda()
    
    # 定义损失函数配置
    criteria_config = [
        {"criterion": criterion_ce, "loss_weight": 1.0},
        {"criterion": criterion_lovasz, "loss_weight": 1.0}
    ]
    
    logger.info("Class weights: {}".format(class_weights))
    logger.info("Loss functions: CrossEntropyLoss (weight=1.0) + LovaszLoss (weight=1.0)")
    # =========================================
    
    # 加载类别名称，如果names_path未设置则使用默认名称
    if hasattr(args, 'names_path') and args.names_path and os.path.isfile(args.names_path):
        names = [line.rstrip('\n') for line in open(args.names_path)]
        logger.info("Loaded class names from: {}".format(args.names_path))
    else:
        names = ['Class_{}'.format(i) for i in range(args.classes)]
        logger.warning("names_path not set or file not found, using default class names: {}".format(names))
    if os.path.isfile(args.model_path):
        logger.info("=> loading checkpoint '{}'".format(args.model_path))
        checkpoint = torch.load(args.model_path)
        state_dict = checkpoint['state_dict']
        new_state_dict = collections.OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=True)
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.model_path, checkpoint['epoch']))
        args.epoch = checkpoint['epoch']
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.model_path))
    test(model, criteria_config, names)


def data_prepare():
    if args.data_name == 's3dis':
        # 使用配置中的split来决定从哪个目录读取数据
        split = getattr(args, 'split', 'test')  # 默认使用test目录
        data_dir = os.path.join(args.data_root, split)
        if os.path.exists(data_dir):
            files = sorted(os.listdir(data_dir))
            # 查找包含Area_{test_area}的.npy文件
            data_list = [item[:-4] for item in files if item.endswith('.npy') and 'Area_{}'.format(args.test_area) in item]
        else:
            # 如果split目录不存在，回退到原来的逻辑
            data_list = sorted(os.listdir(args.data_root))
            data_list = [item[:-4] for item in data_list if 'Area_{}'.format(args.test_area) in item]
    else:
        raise Exception('dataset not supported yet'.format(args.data_name))
    logger.info("Totally {} samples in val set.".format(len(data_list)))
    return data_list


def data_load(data_name):
    # 使用配置中的split来决定从哪个目录读取数据
    split = getattr(args, 'split', 'test')  # 默认使用test目录
    data_dir = os.path.join(args.data_root, split)
    if os.path.exists(data_dir):
        data_path = os.path.join(data_dir, data_name + '.npy')
    else:
        # 如果split目录不存在，回退到原来的逻辑
        data_path = os.path.join(args.data_root, data_name + '.npy')
    data = np.load(data_path)  # xyzrgbl, N*7
    coord, feat, label = data[:, :3], data[:, 3:6], data[:, 6]
    # 将标签从 [1, 2, 3] 转换为 [0, 1, 2]，与训练时保持一致
    label = label - 1

    idx_data = []
    if args.voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        idx_sort, count = voxelize(coord, args.voxel_size, mode=1)
        for i in range(count.max()):
            idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
            idx_part = idx_sort[idx_select]
            idx_data.append(idx_part)
    else:
        idx_data.append(np.arange(label.shape[0]))
    return coord, feat, label, idx_data


def input_normalize(coord, feat):
    coord_min = np.min(coord, 0)
    coord -= coord_min
    feat = feat / 255.
    return coord, feat


def test(model, criteria_config, names):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    loss_meter = AverageMeter()
    loss_ce_meter = AverageMeter()
    loss_lovasz_meter = AverageMeter()
    args.batch_size_test = 10
    model.eval()

    check_makedirs(args.save_folder)
    pred_save, label_save = [], []
    data_list = data_prepare()
    
    # 检查数据列表是否为空
    if len(data_list) == 0:
        logger.error("No validation data found! Please check your data directory and test_area setting.")
        split = getattr(args, 'split', 'test')
        logger.error("Expected data files containing 'Area_{}' in directory: {}".format(args.test_area, os.path.join(args.data_root, split)))
        return
    
    for idx, item in enumerate(data_list):
        end = time.time()
        pred_save_path = os.path.join(args.save_folder, '{}_{}_pred.npy'.format(item, args.epoch))
        label_save_path = os.path.join(args.save_folder, '{}_{}_label.npy'.format(item, args.epoch))
        if os.path.isfile(pred_save_path) and os.path.isfile(label_save_path):
            logger.info('{}/{}: {}, loaded pred and label.'.format(idx + 1, len(data_list), item))
            pred, label = np.load(pred_save_path), np.load(label_save_path)
        else:
            coord, feat, label, idx_data = data_load(item)
            pred = torch.zeros((label.size, args.classes)).cuda()
            idx_size = len(idx_data)
            idx_list, coord_list, feat_list, offset_list  = [], [], [], []
            for i in range(idx_size):
                logger.info('{}/{}: {}/{}/{}, {}'.format(idx + 1, len(data_list), i + 1, idx_size, idx_data[0].shape[0], item))
                idx_part = idx_data[i]
                coord_part, feat_part = coord[idx_part], feat[idx_part]
                if args.voxel_max and coord_part.shape[0] > args.voxel_max:
                    coord_p, idx_uni, cnt = np.random.rand(coord_part.shape[0]) * 1e-3, np.array([]), 0
                    while idx_uni.size != idx_part.shape[0]:
                        init_idx = np.argmin(coord_p)
                        dist = np.sum(np.power(coord_part - coord_part[init_idx], 2), 1)
                        idx_crop = np.argsort(dist)[:args.voxel_max]
                        coord_sub, feat_sub, idx_sub = coord_part[idx_crop], feat_part[idx_crop], idx_part[idx_crop]
                        dist = dist[idx_crop]
                        delta = np.square(1 - dist / np.max(dist))
                        coord_p[idx_crop] += delta
                        coord_sub, feat_sub = input_normalize(coord_sub, feat_sub)
                        idx_list.append(idx_sub), coord_list.append(coord_sub), feat_list.append(feat_sub), offset_list.append(idx_sub.size)
                        idx_uni = np.unique(np.concatenate((idx_uni, idx_sub)))
                else:
                    coord_part, feat_part = input_normalize(coord_part, feat_part)
                    idx_list.append(idx_part), coord_list.append(coord_part), feat_list.append(feat_part), offset_list.append(idx_part.size)
            batch_num = int(np.ceil(len(idx_list) / args.batch_size_test))
            for i in range(batch_num):
                s_i, e_i = i * args.batch_size_test, min((i + 1) * args.batch_size_test, len(idx_list))
                idx_part, coord_part, feat_part, offset_part = idx_list[s_i:e_i], coord_list[s_i:e_i], feat_list[s_i:e_i], offset_list[s_i:e_i]
                idx_part = np.concatenate(idx_part)
                coord_part = torch.FloatTensor(np.concatenate(coord_part)).cuda(non_blocking=True)
                feat_part = torch.FloatTensor(np.concatenate(feat_part)).cuda(non_blocking=True)
                offset_part = torch.IntTensor(np.cumsum(offset_part)).cuda(non_blocking=True)
                with torch.no_grad():
                    pred_part = model([coord_part, feat_part, offset_part])  # (n, k)
                torch.cuda.empty_cache()
                pred[idx_part, :] += pred_part
                logger.info('Test: {}/{}, {}/{}, {}/{}'.format(idx + 1, len(data_list), e_i, len(idx_list), args.voxel_max, idx_part.shape[0]))
            
            # ============ 计算组合损失 ============
            label_tensor = torch.LongTensor(label).cuda(non_blocking=True)
            loss_ce = criteria_config[0]["criterion"](pred, label_tensor) * criteria_config[0]["loss_weight"]
            loss_lovasz = criteria_config[1]["criterion"](pred, label_tensor) * criteria_config[1]["loss_weight"]
            loss = loss_ce + loss_lovasz
            # ====================================
            
            loss_meter.update(loss.item())
            loss_ce_meter.update(loss_ce.item())
            loss_lovasz_meter.update(loss_lovasz.item())
            
            pred = pred.max(1)[1].data.cpu().numpy()

        # calculation 1: add per room predictions
        intersection, union, target = intersectionAndUnion(pred, label, args.classes, args.ignore_label)
        intersection_meter.update(intersection)
        union_meter.update(union)
        target_meter.update(target)

        accuracy = sum(intersection) / (sum(target) + 1e-10)
        batch_time.update(time.time() - end)
        logger.info('Test: [{}/{}]-{} '
                    'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                    'Loss {loss:.4f} (CE: {loss_ce:.4f}, Lovasz: {loss_lovasz:.4f}) '
                    'Accuracy {accuracy:.4f}.'.format(idx + 1, len(data_list), label.size, 
                                                      batch_time=batch_time,
                                                      loss=loss_meter.avg,
                                                      loss_ce=loss_ce_meter.avg,
                                                      loss_lovasz=loss_lovasz_meter.avg,
                                                      accuracy=accuracy))
        pred_save.append(pred); label_save.append(label)
        np.save(pred_save_path, pred); np.save(label_save_path, label)

    with open(os.path.join(args.save_folder, "pred.pickle"), 'wb') as handle:
        pickle.dump({'pred': pred_save}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(args.save_folder, "label.pickle"), 'wb') as handle:
        pickle.dump({'label': label_save}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # calculation 1
    # 检查是否有数据
    if intersection_meter.count == 0:
        logger.warning("No predictions were made. Skipping metrics calculation.")
        return
    
    # intersection_meter.sum应该是numpy数组（intersectionAndUnion返回数组）
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU1 = np.mean(iou_class)
    mAcc1 = np.mean(accuracy_class)
    allAcc1 = np.sum(intersection_meter.sum) / (np.sum(target_meter.sum) + 1e-10)

    # calculation 2
    if len(pred_save) == 0 or len(label_save) == 0:
        logger.warning("No predictions to concatenate. Skipping calculation 2.")
        return
    
    intersection, union, target = intersectionAndUnion(np.concatenate(pred_save), np.concatenate(label_save), args.classes, args.ignore_label)
    iou_class = intersection / (union + 1e-10)
    accuracy_class = intersection / (target + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection) / (sum(target) + 1e-10)
    
    logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    logger.info('Val1 result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU1, mAcc1, allAcc1))
    logger.info('Test losses: Total {:.4f}, CE {:.4f}, Lovasz {:.4f}'.format(loss_meter.avg, loss_ce_meter.avg, loss_lovasz_meter.avg))

    for i in range(args.classes):
        logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}, name: {}.'.format(i, iou_class[i], accuracy_class[i], names[i]))
    
    # ============ 计算并保存混淆矩阵 ============
    logger.info('Computing confusion matrix...')
    all_pred = np.concatenate(pred_save)
    all_label = np.concatenate(label_save)
    
    # 过滤掉ignore_label
    valid_mask = all_label != args.ignore_label
    all_pred_valid = all_pred[valid_mask]
    all_label_valid = all_label[valid_mask]
    
    # 计算混淆矩阵
    confusion_matrix = np.zeros((args.classes, args.classes), dtype=np.int64)
    for pred_cls, true_cls in zip(all_pred_valid, all_label_valid):
        if 0 <= pred_cls < args.classes and 0 <= true_cls < args.classes:
            confusion_matrix[int(true_cls), int(pred_cls)] += 1
    
    # 保存混淆矩阵数据
    cm_save_dir = os.path.join(args.save_folder, 'confusion_matrix')
    check_makedirs(cm_save_dir)
    csv_path, metrics_path = save_confusion_matrix_data(confusion_matrix, names, cm_save_dir)
    logger.info(f'Confusion matrix data saved to: {csv_path}')
    logger.info(f'Class metrics saved to: {metrics_path}')
    
    # 绘制混淆矩阵图
    # 归一化版本
    cm_fig_path = plot_confusion_matrix(
        confusion_matrix, names, 
        os.path.join(cm_save_dir, 'confusion_matrix_normalized.png'),
        normalize=True, 
        title=f'Confusion Matrix (Normalized) - Epoch {args.epoch}'
    )
    logger.info(f'Normalized confusion matrix saved to: {cm_fig_path}')
    
    # 非归一化版本
    cm_fig_path_raw = plot_confusion_matrix(
        confusion_matrix, names,
        os.path.join(cm_save_dir, 'confusion_matrix_counts.png'),
        normalize=False,
        title=f'Confusion Matrix (Counts) - Epoch {args.epoch}'
    )
    logger.info(f'Count confusion matrix saved to: {cm_fig_path_raw}')
    # =============================================
    
    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')


if __name__ == '__main__':
    main()
    