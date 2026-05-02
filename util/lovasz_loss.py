"""
Lovasz-Softmax Loss for Semantic Segmentation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax_flat(probas, labels, classes='present', ignore_index=-1):
    """
    Multi-class Lovasz-Softmax loss
    """
    if probas.numel() == 0:
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float()
        if classes == 'present' and fg.sum() == 0:
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))
    return torch.mean(torch.stack(losses))


class LovaszLoss(nn.Module):
    """
    Lovasz Loss for multiclass segmentation
    """
    def __init__(self, mode='multiclass', ignore_index=-1, per_image=False):
        super(LovaszLoss, self).__init__()
        self.mode = mode
        self.ignore_index = ignore_index
        self.per_image = per_image

    def forward(self, logits, labels):
        """
        Args:
            logits: [N, C] tensor of logits
            labels: [N] tensor of ground truth labels
        """
        if self.mode == 'multiclass':
            probas = F.softmax(logits, dim=1)
            
            # Handle ignore index
            if self.ignore_index is not None:
                valid = (labels != self.ignore_index)
                probas = probas[valid]
                labels = labels[valid]
            
            loss = lovasz_softmax_flat(probas, labels, classes='present', ignore_index=self.ignore_index)
            return loss
        else:
            raise NotImplementedError(f"Mode {self.mode} not implemented")