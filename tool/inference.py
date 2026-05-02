#!/usr/bin/env python
"""
Point Transformer V3 推理脚本
用于点云语义分割推理，按原始点云顺序保存预测标签
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

random.seed(123)
np.random.seed(123)


def get_parser():
    parser = argparse.ArgumentParser(description='Point Transformer V3 Inference')
    parser.add_argument('--config', type=str, default='config/seg/seg_pointtransformer_v3_EMA.yaml', help='config file')
    parser.add_argument('--input_file', type=str, required=True, help='path to input point cloud file (.txt or .npy)')
    parser.add_argument('--model_path', type=str, required=True, help='path to model checkpoint (.pth)')
    parser.add_argument('--output_file', type=str, default='', help='path to output file (default: input_pred.txt)')
    parser.add_argument('--batch_size', type=int, default=10, help='batch size for inference')
    parser.add_argument('opts', help='additional options', default=None, nargs=argparse.REMAINDER)
    
    args = parser.parse_args()
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    cfg.input_file = args.input_file
    cfg.model_path = args.model_path
    cfg.output_file = args.output_file if args.output_file else args.input_file.replace('.txt', '_pred.txt').replace('.npy', '_pred.txt')
    cfg.batch_size_infer = args.batch_size
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
    返回: coord (N, 3), feat (N, 3), original_data (N, ?)
    """
    logger.info(f"Loading point cloud from: {file_path}")
    
    if file_path.endswith('.npy'):
        data = np.load(file_path)
    elif file_path.endswith('.txt'):
        data = np.loadtxt(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}. Use .txt or .npy")
    
    logger.info(f"Loaded point cloud with shape: {data.shape}")
    
    # 提取坐标和特征
    coord = data[:, :3].astype(np.float32)  # xyz
    
    if data.shape[1] >= 6:
        feat = data[:, 3:6].astype(np.float32)  # RGB
    else:
        feat = np.ones((coord.shape[0], 3), dtype=np.float32) * 128  # 默认灰色
    
    return coord, feat, data


def input_normalize(coord, feat):
    """归一化输入数据"""
    coord_min = np.min(coord, 0)
    coord = coord - coord_min
    feat = feat / 255.0
    return coord, feat


def inference(model, args, coord, feat, logger):
    """
    对点云进行推理
    使用与test.py相似的处理逻辑，确保所有点都被预测
    返回: pred (N,) 每个点的预测类别
    """
    model.eval()
    n_points = coord.shape[0]
    pred = torch.zeros((n_points, args.classes)).cuda()
    
    # 准备数据索引
    idx_data = []
    if args.voxel_size:
        coord_min = np.min(coord, 0)
        coord_centered = coord - coord_min
        idx_sort, count = voxelize(coord_centered, args.voxel_size, mode=1)
        for i in range(count.max()):
            idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
            idx_part = idx_sort[idx_select]
            idx_data.append(idx_part)
    else:
        idx_data.append(np.arange(n_points))
    
    logger.info(f"Total voxel passes: {len(idx_data)}")
    
    # 准备批次数据
    idx_list, coord_list, feat_list, offset_list = [], [], [], []
    
    logger.info("Preparing batch data...")
    for i, idx_part in tqdm(enumerate(idx_data), total=len(idx_data), desc="Preparing batches"):
        coord_part, feat_part = coord[idx_part], feat[idx_part]
        
        if args.voxel_max and coord_part.shape[0] > args.voxel_max:
            # 如果点数超过voxel_max，需要分块处理
            coord_p = np.random.rand(coord_part.shape[0]) * 1e-3
            idx_uni = np.array([])
            cnt = 0
            
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
                cnt += 1
        else:
            coord_part_norm, feat_part_norm = input_normalize(coord_part.copy(), feat_part.copy())
            idx_list.append(idx_part)
            coord_list.append(coord_part_norm)
            feat_list.append(feat_part_norm)
            offset_list.append(idx_part.size)
    
    logger.info(f"Total batches to process: {len(idx_list)}")
    
    # 批次推理
    batch_num = int(np.ceil(len(idx_list) / args.batch_size_infer))
    
    # 使用进度条显示推理进度
    pbar = tqdm(total=len(idx_list), desc="Inference", unit="batch")
    
    for i in range(batch_num):
        s_i = i * args.batch_size_infer
        e_i = min((i + 1) * args.batch_size_infer, len(idx_list))
        
        idx_part = idx_list[s_i:e_i]
        coord_part = coord_list[s_i:e_i]
        feat_part = feat_list[s_i:e_i]
        offset_part = offset_list[s_i:e_i]
        
        idx_part = np.concatenate(idx_part)
        coord_part = torch.FloatTensor(np.concatenate(coord_part)).cuda(non_blocking=True)
        feat_part = torch.FloatTensor(np.concatenate(feat_part)).cuda(non_blocking=True)
        offset_part = torch.IntTensor(np.cumsum(offset_part)).cuda(non_blocking=True)
        
        with torch.no_grad():
            pred_part = model([coord_part, feat_part, offset_part])
        
        torch.cuda.empty_cache()
        pred[idx_part, :] += pred_part
        
        # 更新进度条，显示GPU内存使用
        pbar.update(e_i - s_i)
        gpu_mem = torch.cuda.memory_allocated() / 1024**3
        pbar.set_postfix({
            'points': idx_part.shape[0], 
            'batch': f'{e_i}/{len(idx_list)}',
            'gpu_mem': f'{gpu_mem:.1f}GB'
        })
    
    pbar.close()
    
    # 获取最终预测类别
    pred_class = pred.max(1)[1].data.cpu().numpy()
    
    return pred_class


def save_results(output_file, original_data, pred_class, logger):
    """
    保存预测结果，按原始点云顺序
    """
    logger.info(f"Saving results to: {output_file}")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 将预测标签追加到原始数据
    # 格式: x y z r g b ... pred_label
    n_cols = original_data.shape[1]
    
    # 创建输出数据
    output_data = np.hstack((original_data, pred_class.reshape(-1, 1)))
    
    # 根据列数生成格式字符串
    fmt_str = ' '.join(['%.6f'] * 3)  # xyz
    if n_cols >= 6:
        fmt_str += ' ' + ' '.join(['%d'] * 3)  # RGB
    if n_cols > 6:
        fmt_str += ' ' + ' '.join(['%.6f'] * (n_cols - 6))  # 其他列
    fmt_str += ' %d'  # 预测标签
    
    np.savetxt(output_file, output_data, fmt=fmt_str)
    
    # 同时保存一个只有标签的文件（便于可视化）
    label_only_file = output_file.replace('.txt', '_labels.txt')
    np.savetxt(label_only_file, pred_class, fmt='%d')
    
    logger.info(f"Results saved!")
    logger.info(f"  - Full data with predictions: {output_file}")
    logger.info(f"  - Labels only: {label_only_file}")
    
    # 统计每个类别的点数
    unique, counts = np.unique(pred_class, return_counts=True)
    logger.info("Prediction statistics:")
    for cls, cnt in zip(unique, counts):
        logger.info(f"  Class {cls}: {cnt} points ({100.0*cnt/len(pred_class):.2f}%)")


def main():
    global args, logger
    args = get_parser()
    logger = get_logger()
    
    logger.info("=" * 60)
    logger.info("Point Transformer V3 Inference")
    logger.info("=" * 60)
    logger.info(f"Config: {args}")
    
    # 加载模型
    logger.info("=> Creating model ...")
    
    if args.arch == 'pointtransformer_v3':
        from model.pointtransformer_v3 import PointTransformerV3 as Model
    elif args.arch == 'pointtransformer_seg_repro':
        from model.pointtransformer.pointtransformer_seg import pointtransformer_seg_repro as Model
    else:
        raise Exception(f'Architecture {args.arch} not supported yet')
    
    # 传递 EMA 相关参数（如果配置文件中存在）
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
        logger.info(f"EMA enabled: factor={args.ema_factor}, stages={args.ema_stages}, fusion_weight={args.ema_fusion_weight}")
    
    # 加载权重
    if os.path.isfile(args.model_path):
        logger.info(f"=> Loading checkpoint '{args.model_path}'")
        checkpoint = torch.load(args.model_path)
        state_dict = checkpoint['state_dict']
        
        # 处理 DataParallel 保存的模型
        new_state_dict = collections.OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                name = k[7:]
            else:
                name = k
            new_state_dict[name] = v
        
        model.load_state_dict(new_state_dict, strict=True)
        epoch = checkpoint.get('epoch', 'unknown')
        logger.info(f"=> Loaded checkpoint (epoch {epoch})")
    else:
        raise RuntimeError(f"=> No checkpoint found at '{args.model_path}'")
    
    # 加载点云数据
    coord, feat, original_data = load_point_cloud(args.input_file, logger)
    logger.info(f"Point cloud: {coord.shape[0]} points")
    
    # 推理
    logger.info("=> Starting inference ...")
    start_time = time.time()
    
    pred_class = inference(model, args, coord, feat, logger)
    
    inference_time = time.time() - start_time
    logger.info(f"=> Inference completed in {inference_time:.2f} seconds")
    logger.info(f"=> Speed: {coord.shape[0] / inference_time:.0f} points/sec")
    
    # 保存结果
    save_results(args.output_file, original_data, pred_class, logger)
    
    logger.info("=" * 60)
    logger.info("All done!")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
