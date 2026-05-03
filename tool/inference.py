#!/usr/bin/env python
"""
Point Transformer V3 推理脚本
用于点云语义分割推理，按原始点云顺序保存预测标签
改进：消除随机性来源，支持 ensemble 多次推理取平均
"""
import os
import time
import random
import numpy as np
import logging
import argparse
import collections

import torch
import torch.nn as nn
from tqdm import tqdm

from util import config
from util.voxelize import voxelize


# ──────────────────────────────────────────────
#  全局种子固定（必须在所有其他 import 之后立刻调用）
# ──────────────────────────────────────────────
def set_seed(seed: int = 42):
    """固定所有随机性来源，确保推理结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


def get_parser():
    parser = argparse.ArgumentParser(description='Point Transformer V3 Inference')
    parser.add_argument('--config', type=str, default='config/seg/seg_pointtransformer_v3_EMA.yaml', help='config file')
    parser.add_argument('--input_file', type=str, required=True, help='path to input point cloud file (.txt or .npy)')
    parser.add_argument('--model_path', type=str, required=True, help='path to model checkpoint (.pth)')
    parser.add_argument('--output_file', type=str, default='', help='path to output file (default: input_pred.txt)')
    parser.add_argument('--batch_size', type=int, default=10, help='batch size for inference')
    # 新增：ensemble 推理次数，>1 时开启多次推理取 logit 平均
    parser.add_argument('--ensemble_runs', type=int, default=1,
                        help='number of ensemble runs (default: 1, i.e. single run). '
                             'Set >1 to average logits over multiple runs with different seeds.')
    parser.add_argument('opts', help='additional options', default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    cfg.input_file = args.input_file
    cfg.model_path = args.model_path
    cfg.output_file = (args.output_file if args.output_file
                       else args.input_file.replace('.txt', '_pred.txt').replace('.npy', '_pred.txt'))
    cfg.batch_size_infer = args.batch_size
    cfg.ensemble_runs = args.ensemble_runs
    return cfg


def get_logger():
    logger_name = "inference-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def load_point_cloud(file_path, logger):
    """
    加载点云文件，支持 .txt 和 .npy 格式
    最后一列为真实标签（形如1.0, 2.0, 3.0），转为0-based整数
    返回: coord (N, 3), feat (N, 3), original_data (N, ?), label_gt (N,)
    """
    logger.info(f"Loading point cloud from: {file_path}")

    if file_path.endswith('.npy'):
        data = np.load(file_path)
    elif file_path.endswith('.txt'):
        data = np.loadtxt(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}. Use .txt or .npy")

    logger.info(f"Loaded point cloud with shape: {data.shape}")

    coord = data[:, :3].astype(np.float32)

    if data.shape[1] >= 6:
        feat = data[:, 3:6].astype(np.float32)
    else:
        feat = np.ones((coord.shape[0], 3), dtype=np.float32) * 128

    label_gt = data[:, -1].astype(np.float32)
    label_gt = (label_gt - label_gt.min()).astype(np.int64)

    return coord, feat, data, label_gt


def input_normalize(coord, feat):
    """归一化输入数据"""
    coord_min = np.min(coord, 0)
    coord = coord - coord_min
    feat = feat / 255.0
    return coord, feat


def _build_idx_list(coord, feat, args):
    """
    将点云按 voxel 分块，并在超出 voxel_max 时做确定性 sub-crop 采样。
    
    【核心改动】原来使用 np.random.rand() 初始化 coord_p，导致每次分块结果不同。
    现改为基于"到重心的归一化距离"做确定性初始化，完全消除随机性。
    
    返回: (idx_list, coord_list, feat_list, offset_list)
    """
    n_points = coord.shape[0]
    idx_list, coord_list, feat_list, offset_list = [], [], [], []

    if args.voxel_size:
        coord_min = np.min(coord, 0)
        coord_centered = coord - coord_min
        idx_sort, count = voxelize(coord_centered, args.voxel_size, mode=1)
        idx_data = []
        for i in range(count.max()):
            idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
            idx_part = idx_sort[idx_select]
            idx_data.append(idx_part)
    else:
        idx_data = [np.arange(n_points)]

    for idx_part in idx_data:
        coord_part = coord[idx_part]
        feat_part = feat[idx_part]

        if args.voxel_max and coord_part.shape[0] > args.voxel_max:
            # ── 确定性初始化：用到重心的归一化距离替代 np.random.rand() ──
            center = coord_part.mean(axis=0)
            dist_to_center = np.sum((coord_part - center) ** 2, axis=1)
            max_dist = dist_to_center.max()
            # 归一化到 [0, 1e-3]，与原始量级一致；若所有点重合则退化为全零（不影响逻辑）
            coord_p = dist_to_center / (max_dist + 1e-10) * 1e-3
            # ─────────────────────────────────────────────────────────────

            idx_uni = np.array([])

            while idx_uni.size != idx_part.shape[0]:
                init_idx = np.argmin(coord_p)
                dist = np.sum(np.power(coord_part - coord_part[init_idx], 2), 1)
                idx_crop = np.argsort(dist)[:args.voxel_max]

                coord_sub = coord_part[idx_crop]
                feat_sub = feat_part[idx_crop]
                idx_sub = idx_part[idx_crop]

                dist = dist[idx_crop]
                delta = np.square(1 - dist / np.max(dist))
                coord_p[idx_crop] += delta

                coord_sub, feat_sub = input_normalize(coord_sub, feat_sub)
                idx_list.append(idx_sub)
                coord_list.append(coord_sub)
                feat_list.append(feat_sub)
                offset_list.append(idx_sub.size)

                idx_uni = np.unique(np.concatenate((idx_uni, idx_sub)))
        else:
            coord_part_norm, feat_part_norm = input_normalize(coord_part.copy(), feat_part.copy())
            idx_list.append(idx_part)
            coord_list.append(coord_part_norm)
            feat_list.append(feat_part_norm)
            offset_list.append(idx_part.size)

    return idx_list, coord_list, feat_list, offset_list


def inference(model, args, coord, feat, logger, run_seed: int = 42):
    """
    对点云进行推理。
    run_seed 用于 ensemble 模式：每次使用不同但固定的种子，保证单次内部确定，
    同时各次之间有轻微的随机差异（来自 voxelize 内部），从而做有效的 ensemble。
    返回: pred_logit (N, C) — 原始 logit，供外部累加平均
    """
    # 每次 run 使用固定种子，保证可复现
    set_seed(run_seed)

    model.eval()
    n_points = coord.shape[0]
    pred = torch.zeros((n_points, args.classes)).cuda()

    logger.info(f"  [run seed={run_seed}] Building index list ...")
    idx_list, coord_list, feat_list, offset_list = _build_idx_list(coord, feat, args)
    logger.info(f"  [run seed={run_seed}] Total batches: {len(idx_list)}")

    batch_num = int(np.ceil(len(idx_list) / args.batch_size_infer))
    pbar = tqdm(total=len(idx_list), desc=f"Inference (seed={run_seed})", unit="batch")

    for i in range(batch_num):
        s_i = i * args.batch_size_infer
        e_i = min((i + 1) * args.batch_size_infer, len(idx_list))

        idx_part = np.concatenate(idx_list[s_i:e_i])
        coord_part = torch.FloatTensor(np.concatenate(coord_list[s_i:e_i])).cuda(non_blocking=True)
        feat_part = torch.FloatTensor(np.concatenate(feat_list[s_i:e_i])).cuda(non_blocking=True)
        offset_part = torch.IntTensor(np.cumsum(offset_list[s_i:e_i])).cuda(non_blocking=True)

        with torch.no_grad():
            pred_part = model([coord_part, feat_part, offset_part])

        torch.cuda.empty_cache()
        pred[idx_part, :] += pred_part

        pbar.update(e_i - s_i)
        gpu_mem = torch.cuda.memory_allocated() / 1024 ** 3
        pbar.set_postfix({
            'points': idx_part.shape[0],
            'batch': f'{e_i}/{len(idx_list)}',
            'gpu_mem': f'{gpu_mem:.1f}GB'
        })

    pbar.close()
    return pred  # 返回 logit tensor，不在此处 argmax


def inference_ensemble(model, args, coord, feat, logger):
    """
    多次推理 + logit 平均（ensemble）。
    n_runs=1 时等价于普通单次推理，完全向后兼容。
    """
    n_runs = args.ensemble_runs
    n_points = coord.shape[0]
    pred_sum = torch.zeros((n_points, args.classes)).cuda()

    logger.info(f"Ensemble mode: {n_runs} run(s)")
    for run_idx in range(n_runs):
        run_seed = 42 + run_idx  # 每次使用不同但固定的种子
        logger.info(f"=> Run {run_idx + 1}/{n_runs} (seed={run_seed})")
        pred_sum += inference(model, args, coord, feat, logger, run_seed=run_seed)

    pred_class = pred_sum.max(1)[1].data.cpu().numpy()
    return pred_class


def compute_metrics(pred_class, label_gt, n_classes, ignore_label, logger):
    """计算每个类别的 IoU / Recall / Precision 以及均值"""
    intersection = np.zeros(n_classes)
    union = np.zeros(n_classes)
    target = np.zeros(n_classes)

    for cls in range(n_classes):
        mask_ignore = (label_gt != ignore_label)
        pred_cls = (pred_class == cls) & mask_ignore
        gt_cls = (label_gt == cls) & mask_ignore

        tp = np.logical_and(pred_cls, gt_cls).sum()
        fp = np.logical_and(pred_cls, ~gt_cls).sum()
        fn = np.logical_and(~pred_cls, gt_cls).sum()

        intersection[cls] = tp
        union[cls] = tp + fp + fn
        target[cls] = tp + fn

    iou_class = intersection / (union + 1e-10)
    recall_class = intersection / (target + 1e-10)
    precision_class = intersection / (intersection + union - target + 1e-10)

    miou = np.mean(iou_class)
    mrecall = np.mean(recall_class)
    mprecision = np.mean(precision_class)

    logger.info("=" * 60)
    logger.info("Evaluation Results")
    logger.info("=" * 60)
    logger.info(f"  mIoU:       {miou:.4f}")
    logger.info(f"  mRecall:    {mrecall:.4f}")
    logger.info(f"  mPrecision: {mprecision:.4f}")
    logger.info("-" * 60)
    for cls in range(n_classes):
        logger.info(f"  Class {cls}: IoU {iou_class[cls]:.4f} | "
                    f"Recall {recall_class[cls]:.4f} | "
                    f"Precision {precision_class[cls]:.4f}")
    logger.info("=" * 60)


def save_results(output_file, original_data, pred_class, logger):
    """保存预测结果，按原始点云顺序"""
    logger.info(f"Saving results to: {output_file}")

    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    n_cols = original_data.shape[1]
    output_data = np.hstack((original_data, pred_class.reshape(-1, 1)))

    fmt_str = ' '.join(['%.6f'] * 3)
    if n_cols >= 6:
        fmt_str += ' ' + ' '.join(['%d'] * 3)
    if n_cols > 6:
        fmt_str += ' ' + ' '.join(['%.6f'] * (n_cols - 6))
    fmt_str += ' %d'

    np.savetxt(output_file, output_data, fmt=fmt_str)

    label_only_file = output_file.replace('.txt', '_labels.txt')
    np.savetxt(label_only_file, pred_class, fmt='%d')

    logger.info("Results saved!")
    logger.info(f"  - Full data with predictions: {output_file}")
    logger.info(f"  - Labels only: {label_only_file}")

    unique, counts = np.unique(pred_class, return_counts=True)
    logger.info("Prediction statistics:")
    for cls, cnt in zip(unique, counts):
        logger.info(f"  Class {cls}: {cnt} points ({100.0 * cnt / len(pred_class):.2f}%)")


def main():
    global args, logger
    args = get_parser()
    logger = get_logger()

    logger.info("=" * 60)
    logger.info("Point Transformer V3 Inference")
    logger.info("=" * 60)
    logger.info(f"Config: {args}")

    logger.info("=> Creating model ...")

    if args.arch == 'pointtransformer_v3':
        from model.pointtransformer_v3 import PointTransformerV3 as Model
    elif args.arch == 'pointtransformer_seg_repro':
        from model.pointtransformer.pointtransformer_seg import pointtransformer_seg_repro as Model
    else:
        raise Exception(f'Architecture {args.arch} not supported yet')

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

    model = Model(**model_kwargs).cuda()
    logger.info(f"Model: {args.arch}, in_channels={args.fea_dim}, classes={args.classes}")
    if hasattr(args, 'enable_ema') and args.enable_ema:
        logger.info(f"EMA enabled: factor={args.ema_factor}, stages={args.ema_stages}, "
                    f"fusion_weight={args.ema_fusion_weight}")

    if os.path.isfile(args.model_path):
        logger.info(f"=> Loading checkpoint '{args.model_path}'")
        checkpoint = torch.load(args.model_path)
        state_dict = checkpoint['state_dict']

        new_state_dict = collections.OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict, strict=True)
        epoch = checkpoint.get('epoch', 'unknown')
        logger.info(f"=> Loaded checkpoint (epoch {epoch})")
    else:
        raise RuntimeError(f"=> No checkpoint found at '{args.model_path}'")

    coord, feat, original_data, label_gt = load_point_cloud(args.input_file, logger)
    logger.info(f"Point cloud: {coord.shape[0]} points")

    logger.info("=> Starting inference ...")
    start_time = time.time()
    pred_class = inference_ensemble(model, args, coord, feat, logger)
    inference_time = time.time() - start_time
    logger.info(f"=> Inference completed in {inference_time:.2f} seconds")
    logger.info(f"=> Speed: {coord.shape[0] / inference_time:.0f} points/sec")

    ignore_label = args.ignore_label if hasattr(args, 'ignore_label') else 255
    compute_metrics(pred_class, label_gt, args.classes, ignore_label, logger)

    save_results(args.output_file, original_data, pred_class, logger)

    logger.info("=" * 60)
    logger.info("All done!")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()