"""
Point Transformer - V3 Mode1
Pointcept detached version

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import sys
from functools import partial
from addict import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv
import torch_scatter
from timm.models.layers import DropPath
from collections import OrderedDict

try:
    import flash_attn
except ImportError:
    flash_attn = None

from .serialization import encode


@torch.inference_mode()
def offset2bincount(offset):
    return torch.diff(
        offset, prepend=torch.tensor([0], device=offset.device, dtype=torch.long)
    )


@torch.inference_mode()
def offset2batch(offset):
    bincount = offset2bincount(offset)
    return torch.arange(
        len(bincount), device=offset.device, dtype=torch.long
    ).repeat_interleave(bincount)


@torch.inference_mode()
def batch2offset(batch):
    return torch.cumsum(batch.bincount(), dim=0).long()


# =============================================================================
#  [已修改] EMA 模块: 从 2D 适配为 1D 结构
# =============================================================================
class EMA(nn.Module):
    def __init__(self, channels, c2=None, factor=32):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        
        # 修改: 使用 1D 全局池化
        self.agp = nn.AdaptiveAvgPool1d(1)
        
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        
        # 修改: 使用 1D 卷积
        self.conv1x1 = nn.Conv1d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv1d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # x shape: (Batch, Channels, Length)
        b, c, n = x.size()
        group_x = x.reshape(b * self.groups, -1, n)  # (B*G, C//G, N)
        
        # --- 分支 1: 全局上下文 (Global Context) ---
        x_g = self.agp(group_x) # (B*G, C//G, 1)
        x_g = self.conv1x1(x_g) 
        x1 = self.gn(group_x * x_g.sigmoid()) 
        
        # --- 分支 2: 局部上下文 (Local Context) ---
        x2 = self.conv3x3(group_x)
        
        # --- 跨空间聚合 (Cross Aggregation) ---
        # Branch 1 Global -> Branch 2 Features
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        
        # Branch 2 Global -> Branch 1 Features
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)
        
        # Matrix Multiplication
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, n)
        
        return (group_x * weights.sigmoid()).reshape(b, c, n)
# =============================================================================


class Point(Dict):
    """
    Point Structure of Pointcept
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # If one of "offset" or "batch" do not exist, generate by the existing one
        if "batch" not in self.keys() and "offset" in self.keys():
            self["batch"] = offset2batch(self.offset)
        elif "offset" not in self.keys() and "batch" in self.keys():
            self["offset"] = batch2offset(self.batch)

    def serialization(self, order="z", depth=None, shuffle_orders=False):
        """
        Point Cloud Serialization
        """
        assert "batch" in self.keys()
        if "grid_coord" not in self.keys():
            assert {"grid_size", "coord"}.issubset(self.keys())
            self["grid_coord"] = torch.div(
                self.coord - self.coord.min(0)[0], self.grid_size, rounding_mode="trunc"
            ).int()

        if depth is None:
            depth = int(self.grid_coord.max()).bit_length()
        self["serialized_depth"] = depth
        assert depth * 3 + len(self.offset).bit_length() <= 63
        assert depth <= 16

        code = [
            encode(self.grid_coord, self.batch, depth, order=order_) for order_ in order
        ]
        code = torch.stack(code)
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(
                code.shape[0], 1
            ),
        )

        if shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code = code[perm]
            order = order[perm]
            inverse = inverse[perm]

        self["serialized_code"] = code
        self["serialized_order"] = order
        self["serialized_inverse"] = inverse

    def sparsify(self, pad=96):
        """
        Point Cloud Serialization
        """
        assert {"feat", "batch"}.issubset(self.keys())
        if "grid_coord" not in self.keys():
            assert {"grid_size", "coord"}.issubset(self.keys())
            self["grid_coord"] = torch.div(
                self.coord - self.coord.min(0)[0], self.grid_size, rounding_mode="trunc"
            ).int()
        if "sparse_shape" in self.keys():
            sparse_shape = self.sparse_shape
        else:
            sparse_shape = torch.add(
                torch.max(self.grid_coord, dim=0).values, pad
            ).tolist()
        sparse_conv_feat = spconv.SparseConvTensor(
            features=self.feat,
            indices=torch.cat(
                [self.batch.unsqueeze(-1).int(), self.grid_coord.int()], dim=1
            ).contiguous(),
            spatial_shape=sparse_shape,
            batch_size=self.batch[-1].tolist() + 1,
        )
        self["sparse_shape"] = sparse_shape
        self["sparse_conv_feat"] = sparse_conv_feat


class PointModule(nn.Module):
    r"""PointModule
    placeholder, all module subclass from this will take Point in PointSequential.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PointSequential(PointModule):
    r"""A sequential container.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        for name, module in kwargs.items():
            if sys.version_info < (3, 6):
                raise ValueError("kwargs only supported in py36+")
            if name in self._modules:
                raise ValueError("name exists.")
            self.add_module(name, module)

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError("index {} is out of range".format(idx))
        if idx < 0:
            idx += len(self)
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        return len(self._modules)

    def add(self, module, name=None):
        if name is None:
            name = str(len(self._modules))
            if name in self._modules:
                raise KeyError("name exists")
        self.add_module(name, module)

    def forward(self, input):
        for k, module in self._modules.items():
            # Point module
            if isinstance(module, PointModule):
                input = module(input)
            # Spconv module
            elif spconv.modules.is_spconv_module(module):
                if isinstance(input, Point):
                    input.sparse_conv_feat = module(input.sparse_conv_feat)
                    input.feat = input.sparse_conv_feat.features
                else:
                    input = module(input)
            # PyTorch module
            else:
                if isinstance(input, Point):
                    input.feat = module(input.feat)
                    if "sparse_conv_feat" in input.keys():
                        input.sparse_conv_feat = input.sparse_conv_feat.replace_feature(
                            input.feat
                        )
                elif isinstance(input, spconv.SparseConvTensor):
                    if input.indices.shape[0] != 0:
                        input = input.replace_feature(module(input.features))
                else:
                    input = module(input)
        return input


class PDNorm(PointModule):
    def __init__(
        self,
        num_features,
        norm_layer,
        context_channels=256,
        conditions=("ScanNet", "S3DIS", "Structured3D"),
        decouple=True,
        adaptive=False,
    ):
        super().__init__()
        self.conditions = conditions
        self.decouple = decouple
        self.adaptive = adaptive
        if self.decouple:
            self.norm = nn.ModuleList([norm_layer(num_features) for _ in conditions])
        else:
            self.norm = norm_layer
        if self.adaptive:
            self.modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(context_channels, 2 * num_features, bias=True)
            )

    def forward(self, point):
        assert {"feat", "condition"}.issubset(point.keys())
        if isinstance(point.condition, str):
            condition = point.condition
        else:
            condition = point.condition[0]
        if self.decouple:
            assert condition in self.conditions
            norm = self.norm[self.conditions.index(condition)]
        else:
            norm = self.norm
        point.feat = norm(point.feat)
        if self.adaptive:
            assert "context" in point.keys()
            shift, scale = self.modulation(point.context).chunk(2, dim=1)
            point.feat = point.feat * (1.0 + scale) + shift
        return point


class RPE(torch.nn.Module):
    def __init__(self, patch_size, num_heads):
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.pos_bnd = int((4 * patch_size) ** (1 / 3) * 2)
        self.rpe_num = 2 * self.pos_bnd + 1
        self.rpe_table = torch.nn.Parameter(torch.zeros(3 * self.rpe_num, num_heads))
        torch.nn.init.trunc_normal_(self.rpe_table, std=0.02)

    def forward(self, coord):
        idx = (
            coord.clamp(-self.pos_bnd, self.pos_bnd)  # clamp into bnd
            + self.pos_bnd  # relative position to positive index
            + torch.arange(3, device=coord.device) * self.rpe_num  # x, y, z stride
        )
        out = self.rpe_table.index_select(0, idx.reshape(-1).long())
        out = out.view(idx.shape + (-1,)).sum(3)
        out = out.permute(0, 3, 1, 2)  # (N, K, K, H) -> (N, H, K, K)
        return out


class SerializedAttention(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        order_index=0,
        enable_rpe=False,
        enable_flash=False,  # V100服务器不稳定支持flash-attn，默认走已有的普通attention分支
        upcast_attention=True,
        upcast_softmax=True,
    ):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        self.order_index = order_index
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.enable_rpe = enable_rpe
        self.enable_flash = enable_flash
        if enable_flash:
            assert (
                enable_rpe is False
            ), "Set enable_rpe to False when enable Flash Attention"
            assert (
                upcast_attention is False
            ), "Set upcast_attention to False when enable Flash Attention"
            assert (
                upcast_softmax is False
            ), "Set upcast_softmax to False when enable Flash Attention"
            assert flash_attn is not None, "Make sure flash_attn is installed."
            self.patch_size = patch_size
            self.attn_drop = attn_drop
        else:
            # when disable flash attention, we still don't want to use mask
            # consequently, patch size will auto set to the
            # min number of patch_size_max and number of points
            self.patch_size_max = patch_size
            self.patch_size = 0
            self.attn_drop = torch.nn.Dropout(attn_drop)

        self.qkv = torch.nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = torch.nn.Linear(channels, channels)
        self.proj_drop = torch.nn.Dropout(proj_drop)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.rpe = RPE(patch_size, num_heads) if self.enable_rpe else None

    @torch.no_grad()
    def get_rel_pos(self, point, order):
        K = self.patch_size
        rel_pos_key = f"rel_pos_{self.order_index}"
        if rel_pos_key not in point.keys():
            grid_coord = point.grid_coord[order]
            grid_coord = grid_coord.reshape(-1, K, 3)
            point[rel_pos_key] = grid_coord.unsqueeze(2) - grid_coord.unsqueeze(1)
        return point[rel_pos_key]

    @torch.no_grad()
    def get_padding_and_inverse(self, point):
        pad_key = "pad"
        unpad_key = "unpad"
        cu_seqlens_key = "cu_seqlens_key"
        if (
            pad_key not in point.keys()
            or unpad_key not in point.keys()
            or cu_seqlens_key not in point.keys()
        ):
            offset = point.offset
            bincount = offset2bincount(offset)
            bincount_pad = (
                torch.div(
                    bincount + self.patch_size - 1,
                    self.patch_size,
                    rounding_mode="trunc",
                )
                * self.patch_size
            )
            # only pad point when num of points larger than patch_size
            mask_pad = bincount > self.patch_size
            bincount_pad = ~mask_pad * bincount + mask_pad * bincount_pad
            _offset = nn.functional.pad(offset, (1, 0))
            _offset_pad = nn.functional.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))
            pad = torch.arange(_offset_pad[-1], device=offset.device)
            unpad = torch.arange(_offset[-1], device=offset.device)
            cu_seqlens = []
            for i in range(len(offset)):
                unpad[_offset[i] : _offset[i + 1]] += _offset_pad[i] - _offset[i]
                if bincount[i] != bincount_pad[i]:
                    pad[
                        _offset_pad[i + 1]
                        - self.patch_size
                        + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
                    ] = pad[
                        _offset_pad[i + 1]
                        - 2 * self.patch_size
                        + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
                        - self.patch_size
                    ]
                pad[_offset_pad[i] : _offset_pad[i + 1]] -= _offset_pad[i] - _offset[i]
                cu_seqlens.append(
                    torch.arange(
                        _offset_pad[i],
                        _offset_pad[i + 1],
                        step=self.patch_size,
                        dtype=torch.int32,
                        device=offset.device,
                    )
                )
            point[pad_key] = pad
            point[unpad_key] = unpad
            point[cu_seqlens_key] = nn.functional.pad(
                torch.concat(cu_seqlens), (0, 1), value=_offset_pad[-1]
            )
        return point[pad_key], point[unpad_key], point[cu_seqlens_key]

    def forward(self, point):
        if not self.enable_flash:
            self.patch_size = min(
                offset2bincount(point.offset).min().tolist(), self.patch_size_max
            )

        H = self.num_heads
        K = self.patch_size
        C = self.channels

        pad, unpad, cu_seqlens = self.get_padding_and_inverse(point)

        order = point.serialized_order[self.order_index][pad]
        inverse = unpad[point.serialized_inverse[self.order_index]]

        # padding and reshape feat and batch for serialized point patch
        qkv = self.qkv(point.feat)[order]

        if not self.enable_flash:
            # encode and reshape qkv: (N', K, 3, H, C') => (3, N', H, K, C')
            q, k, v = (
                qkv.reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0)
            )
            # attn
            if self.upcast_attention:
                q = q.float()
                k = k.float()
            attn = (q * self.scale) @ k.transpose(-2, -1)  # (N', H, K, K)
            if self.enable_rpe:
                attn = attn + self.rpe(self.get_rel_pos(point, order))
            if self.upcast_softmax:
                attn = attn.float()
            attn = self.softmax(attn)
            attn = self.attn_drop(attn).to(qkv.dtype)
            feat = (attn @ v).transpose(1, 2).reshape(-1, C)
        else:
            feat = flash_attn.flash_attn_varlen_qkvpacked_func(
                qkv.half().reshape(-1, 3, H, C // H),
                cu_seqlens,
                max_seqlen=self.patch_size,
                dropout_p=self.attn_drop if self.training else 0,
                softmax_scale=self.scale,
            ).reshape(-1, C)
            feat = feat.to(qkv.dtype)
        feat = feat[inverse]

        # ffn
        feat = self.proj(feat)
        feat = self.proj_drop(feat)
        point.feat = feat
        return point


class MLP(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_channels=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PointGRN(PointModule):
    """
    Point-wise Global Response Normalization for point-cloud sequences.

    该模块参考 ConvNeXt V2 中的 GRN（Global Response Normalization）思想，
    但针对点云语义分割的变长 batch 做了适配：每个样本使用自己的 offset
    独立统计通道响应，避免不同场景点数不一致时互相污染统计量。

    与现有 EMA 的区别：
    - EMA 主要做 1D 序列上的全局/局部空间上下文重标定；
    - GRN 只在通道维度建立响应竞争，不生成空间注意力图，因此功能不重复。
    """

    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.channels = channels
        self.eps = eps

        # gamma / beta 采用零初始化，使新模块在训练初期近似恒等映射。
        # 这样可以保留原 PTv3 + EMA 的稳定收敛路径，再由训练逐步学习通道响应校准。
        self.gamma = nn.Parameter(torch.zeros(1, channels))
        self.beta = nn.Parameter(torch.zeros(1, channels))

    def forward(self, point: Point):
        assert {"feat", "offset"}.issubset(point.keys())

        feat = point.feat
        offset = point.offset
        bincount = offset2bincount(offset)
        start = 0
        output_list = []

        for cnt in bincount:
            cnt = int(cnt)
            if cnt > 0:
                sample_feat = feat[start : start + cnt]

                # 对当前样本的每个通道计算 L2 全局响应，形状为 (1, C)。
                # 点云 batch 中每个场景点数不同，逐样本统计能避免长场景支配短场景。
                response = torch.norm(sample_feat, p=2, dim=0, keepdim=True)

                # 用通道响应的均值归一化，形成通道间的竞争系数。
                # eps 只用于数值稳定，不改变类别或数据相关策略。
                response_norm = response / (response.mean(dim=1, keepdim=True) + self.eps)

                # 残差式 GRN：零初始化时输出等于输入，训练后学习通道响应增强/抑制。
                sample_feat = sample_feat + self.gamma * (sample_feat * response_norm) + self.beta
                output_list.append(sample_feat)

            start += cnt

        point.feat = torch.cat(output_list, dim=0) if output_list else feat
        if "sparse_conv_feat" in point.keys():
            point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


class GRNMLP(PointModule):
    """
    带 GRN 的 PTv3 MLP 分支。

    顺序保持为 Linear -> GELU -> Dropout -> PointGRN -> Linear -> Dropout。
    GRN 插在 MLP 扩展通道之后，与 ConvNeXt V2 的接入位置一致；它属于
    Block 内部特征校准，不改分类头、不改损失函数、不改数据处理。
    """

    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_channels=None,
        act_layer=nn.GELU,
        drop=0.0,
        enable_grn=True,
        grn_eps=1e-6,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.grn = PointGRN(hidden_channels, eps=grn_eps) if enable_grn else None
        self.fc2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, point: Point):
        assert {"feat", "offset"}.issubset(point.keys())

        point.feat = self.fc1(point.feat)
        point.feat = self.act(point.feat)
        point.feat = self.drop(point.feat)

        if self.grn is not None:
            point = self.grn(point)

        point.feat = self.fc2(point.feat)
        point.feat = self.drop(point.feat)
        if "sparse_conv_feat" in point.keys():
            point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


class PointStateMixer(PointModule):
    """
    轻量级点云序列混合模块。

    模块类型说明：
    - 属于“状态空间 / Mamba 类长序列建模”方向的轻量化实现，参考
      Mamba(2023)、Vision Mamba(2024) 和 PointMamba(2024) 中利用线性复杂度
      序列建模增强长程依赖的思路，目标是补足 EMA 之外的顺序依赖建模能力。
    - 不改分类头、不改损失函数、不改数据增强，也不引入新的序列化规则，只在
      当前 PTv3 已有的点特征流上做线性复杂度的双向序列混合。

    设计动机：
    - EMA 更偏向通道与局部上下文重标定；
    - 这里增加的是沿点序列方向的前向 / 反向信息传播，两者功能互补而不重复。
    """

    def __init__(
        self,
        channels,
        kernel_size=5,
        expansion=2,
        proj_drop=0.0,
        fusion_weight=0.1,
    ):
        super().__init__()
        hidden_channels = channels * expansion
        padding = kernel_size // 2

        self.norm = nn.GroupNorm(1, channels)
        self.in_proj = nn.Conv1d(channels, hidden_channels, kernel_size=1, bias=False)
        self.forward_mixer = nn.Conv1d(
            hidden_channels,
            hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=hidden_channels,
            bias=False,
        )
        self.backward_mixer = nn.Conv1d(
            hidden_channels,
            hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=hidden_channels,
            bias=False,
        )
        self.out_proj = nn.Conv1d(hidden_channels, channels, kernel_size=1, bias=False)
        self.channel_gate = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.drop = nn.Dropout(proj_drop)

        # 用较小的初始融合权重保持保守起步，尽量降低新模块对基线的扰动。
        self.fusion_weight = nn.Parameter(torch.tensor(fusion_weight))

        # 输出投影零初始化，使模块初始状态近似恒等映射；训练时再逐步学习
        # 沿点序列方向的双向状态传播，降低对已稳定 EMA 分支的早期扰动。
        nn.init.zeros_(self.out_proj.weight)

    def _mix_single_sample(self, sample_feat):
        """
        输入形状：(N, C)
        输出形状：(N, C)
        """
        x = sample_feat.transpose(0, 1).unsqueeze(0)  # (1, C, N)
        x_norm = self.norm(x)

        # 全局池化生成通道门控，控制状态混合的注入强度。
        gate = torch.sigmoid(self.channel_gate(x_norm.mean(dim=-1, keepdim=True)))

        state = self.in_proj(x_norm)
        state = F.gelu(state)

        forward_state = self.forward_mixer(state)
        backward_state = self.backward_mixer(state.flip(-1)).flip(-1)
        mixed_state = 0.5 * (forward_state + backward_state)
        mixed_state = self.out_proj(self.drop(F.gelu(mixed_state)))

        x = x + self.fusion_weight * gate * mixed_state
        return x.squeeze(0).transpose(0, 1)

    def forward(self, point: Point):
        assert {"feat", "offset"}.issubset(point.keys())
        feat = point.feat
        offset = point.offset

        bincount = offset2bincount(offset)
        start = 0
        output_list = []
        for cnt in bincount:
            cnt = int(cnt)
            if cnt > 0:
                sample_feat = feat[start : start + cnt]
                output_list.append(self._mix_single_sample(sample_feat))
            start += cnt

        point.feat = torch.cat(output_list, dim=0) if output_list else feat
        if "sparse_conv_feat" in point.keys():
            point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


class PointResidualFeatureGate(PointModule):
    """
    残差式点云特征门控模块。

    模块类型说明：
    - 属于“特征重标定 / 残差门控”方向，只根据每个样本自身的通道统计生成
      channel-wise gate，不做类别相关调整，也不改变点云序列化、采样策略或损失函数。
    - 与 EMA 的区别是：EMA 侧重多尺度空间上下文建模，这里只做解码特征的通道可靠性
      重标定，不生成空间注意力图，因此功能上不重复。

    稳定性设计：
    - 每个 batch 样本按 offset 独立统计均值和方差，避免不同文件点数差异互相污染；
    - 最后一层零初始化，初始输出严格接近恒等映射，先保留 PTv3 + EMA 的原有能力；
    - 通过较小的可学习 fusion_weight 逐步放大有效门控，减少对 class0/class2 的扰动。
    """

    def __init__(self, channels, reduction=4, fusion_weight=0.05, eps=1e-6):
        super().__init__()
        self.channels = channels
        self.eps = eps
        hidden_channels = max(channels // reduction, 8)

        # 输入拼接每个样本的通道均值和标准差，既保留响应强度，也保留波动信息。
        # 这里只输出通道门控，不接触类别 logits，因此不会形成针对 class1 的硬编码偏置。
        self.gate_mlp = nn.Sequential(
            nn.Linear(channels * 2, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, channels),
        )
        self.fusion_weight = nn.Parameter(torch.tensor(fusion_weight))

        # 零初始化让模块初始为恒等残差，避免新模块一开始破坏 EMA-only 的稳定基线。
        nn.init.zeros_(self.gate_mlp[-1].weight)
        nn.init.zeros_(self.gate_mlp[-1].bias)

    def forward(self, point: Point):
        assert {"feat", "offset"}.issubset(point.keys())

        feat = point.feat
        offset = point.offset
        bincount = offset2bincount(offset)

        output_list = []
        start = 0
        for cnt in bincount:
            end = start + cnt
            sample_feat = feat[start:end]

            if cnt > 0:
                # 每个样本独立统计，避免大场景样本主导小场景样本的通道门控。
                mean = sample_feat.mean(dim=0, keepdim=True)
                centered = sample_feat - mean
                std = torch.sqrt(centered.pow(2).mean(dim=0, keepdim=True) + self.eps)
                context = torch.cat([mean, std], dim=1)

                # tanh 将门控幅度限制在 [-1, 1]，再乘较小 fusion_weight，控制改动幅度。
                gate = torch.tanh(self.gate_mlp(context))
                sample_feat = sample_feat + self.fusion_weight * sample_feat * gate

            output_list.append(sample_feat)
            start = end

        point.feat = torch.cat(output_list, dim=0) if output_list else feat
        if "sparse_conv_feat" in point.keys():
            point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


class Block(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size=48,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        pre_norm=True,
        order_index=0,
        cpe_indice_key=None,
        enable_rpe=False,
        enable_flash=False,  # V100服务器不稳定支持flash-attn，默认走已有的普通attention分支
        upcast_attention=True,
        upcast_softmax=True,
        enable_grn=True,
        grn_eps=1e-6,
    ):
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm

        self.cpe = PointSequential(
            spconv.SubMConv3d(
                channels,
                channels,
                kernel_size=3,
                # 显式设置 padding=1，保证 3x3x3 子流形卷积按中心对齐生成邻域。
                # 服务器 V100 + CUDA 12.8 + spconv-cu120 环境下，默认 padding 在 CPE
                # 第一层会触发底层 FPE；该设置不改变数据或训练策略，只稳定 spconv 索引构建。
                padding=1,
                bias=True,
                indice_key=cpe_indice_key,
                # V100(sm70) 上 spconv 的默认卷积算法可能走到不稳定的隐式 GEMM 路径。
                # CPE 是局部位置编码分支，显式使用 Native 算法更保守，优先保证训练可运行。
                algo=spconv.ConvAlgo.Native,
            ),
            nn.Linear(channels, channels),
            norm_layer(channels),
        )

        self.norm1 = PointSequential(norm_layer(channels))
        self.attn = SerializedAttention(
            channels=channels,
            patch_size=patch_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            order_index=order_index,
            enable_rpe=enable_rpe,
            enable_flash=enable_flash,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
        )
        self.norm2 = PointSequential(norm_layer(channels))
        self.mlp = PointSequential(
            GRNMLP(
                in_channels=channels,
                hidden_channels=int(channels * mlp_ratio),
                out_channels=channels,
                act_layer=act_layer,
                drop=proj_drop,
                enable_grn=enable_grn,
                grn_eps=grn_eps,
            )
        )
        self.drop_path = PointSequential(
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

    def forward(self, point: Point):
        shortcut = point.feat
        point = self.cpe(point)
        point.feat = shortcut + point.feat
        shortcut = point.feat
        if self.pre_norm:
            point = self.norm1(point)
        point = self.drop_path(self.attn(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm1(point)

        shortcut = point.feat
        if self.pre_norm:
            point = self.norm2(point)
        point = self.drop_path(self.mlp(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm2(point)
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


class SerializedPooling(PointModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        norm_layer=None,
        act_layer=None,
        reduce="max",
        shuffle_orders=True,
        traceable=True,  # record parent and cluster
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert stride == 2 ** (math.ceil(stride) - 1).bit_length()  # 2, 4, 8
        # TODO: add support to grid pool (any stride)
        self.stride = stride
        assert reduce in ["sum", "mean", "min", "max"]
        self.reduce = reduce
        self.shuffle_orders = shuffle_orders
        self.traceable = traceable

        self.proj = nn.Linear(in_channels, out_channels)
        if norm_layer is not None:
            self.norm = PointSequential(norm_layer(out_channels))
        if act_layer is not None:
            self.act = PointSequential(act_layer())

    def forward(self, point: Point):
        pooling_depth = (math.ceil(self.stride) - 1).bit_length()
        if pooling_depth > point.serialized_depth:
            pooling_depth = 0
        assert {
            "serialized_code",
            "serialized_order",
            "serialized_inverse",
            "serialized_depth",
        }.issubset(
            point.keys()
        ), "Run point.serialization() point cloud before SerializedPooling"

        code = point.serialized_code >> pooling_depth * 3
        code_, cluster, counts = torch.unique(
            code[0],
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        # indices of point sorted by cluster, for torch_scatter.segment_csr
        _, indices = torch.sort(cluster)
        # index pointer for sorted point, for torch_scatter.segment_csr
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        # head_indices of each cluster, for reduce attr e.g. code, batch
        head_indices = indices[idx_ptr[:-1]]
        # generate down code, order, inverse
        code = code[:, head_indices]
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(
                code.shape[0], 1
            ),
        )

        if self.shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code = code[perm]
            order = order[perm]
            inverse = inverse[perm]

        # collect information
        point_dict = Dict(
            feat=torch_scatter.segment_csr(
                self.proj(point.feat)[indices], idx_ptr, reduce=self.reduce
            ),
            coord=torch_scatter.segment_csr(
                point.coord[indices], idx_ptr, reduce="mean"
            ),
            grid_coord=point.grid_coord[head_indices] >> pooling_depth,
            serialized_code=code,
            serialized_order=order,
            serialized_inverse=inverse,
            serialized_depth=point.serialized_depth - pooling_depth,
            batch=point.batch[head_indices],
        )

        if "condition" in point.keys():
            point_dict["condition"] = point.condition
        if "context" in point.keys():
            point_dict["context"] = point.context

        if self.traceable:
            point_dict["pooling_inverse"] = cluster
            point_dict["pooling_parent"] = point
        point = Point(point_dict)
        if self.norm is not None:
            point = self.norm(point)
        if self.act is not None:
            point = self.act(point)
        point.sparsify()
        return point


class SerializedUnpooling(PointModule):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        norm_layer=None,
        act_layer=None,
        traceable=False,  # record parent and cluster
    ):
        super().__init__()
        self.proj = PointSequential(nn.Linear(in_channels, out_channels))
        self.proj_skip = PointSequential(nn.Linear(skip_channels, out_channels))

        if norm_layer is not None:
            self.proj.add(norm_layer(out_channels))
            self.proj_skip.add(norm_layer(out_channels))

        if act_layer is not None:
            self.proj.add(act_layer())
            self.proj_skip.add(act_layer())

        self.traceable = traceable

    def forward(self, point):
        assert "pooling_parent" in point.keys()
        assert "pooling_inverse" in point.keys()
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        point = self.proj(point)
        parent = self.proj_skip(parent)
        parent.feat = parent.feat + point.feat[inverse]

        if self.traceable:
            parent["unpooling_parent"] = point
        return parent


# =============================================================================
#  [已修改] EMAPointAdapter: 逐 Batch 处理，避免 Padding 补零导致均值错误
#  添加残差连接和可学习的融合权重，提高稳定性
# =============================================================================
class EMAPointAdapter(PointModule):
    """
    Revised Adapter: Processing point cloud by batch offsets
    to avoid padding zeros which corrupts Global Average Pooling in EMA.
    Now with residual connection and learnable fusion weight.
    """
    def __init__(self, channels, factor=32, use_residual=True, fusion_weight=0.1):
        super().__init__()
        self.ema = EMA(channels, factor=factor)
        self.use_residual = use_residual
        
        # 可学习的融合权重，初始值较小，让EMA的影响逐渐增强
        if use_residual:
            # 使用可学习的权重参数
            self.fusion_weight = nn.Parameter(torch.tensor(fusion_weight))
        else:
            self.fusion_weight = None

    def forward(self, point: Point):
        assert {"feat", "offset"}.issubset(point.keys())
        feat = point.feat
        offset = point.offset
        
        bincount = offset2bincount(offset)
        start = 0
        output_list = []
        
        # Iterate over each sample in the batch
        for cnt in bincount:
            cnt = int(cnt)
            if cnt > 0:
                # Extract sample: (N, C) -> (1, C, N)
                # 1D Conv expects (Batch, Channels, Length)
                x = feat[start : start + cnt].transpose(0, 1).unsqueeze(0)
                
                # Apply EMA
                x_ema = self.ema(x)
                
                # 残差连接：保留原始特征，让EMA作为增强
                if self.use_residual:
                    # 使用可学习的权重融合原始特征和EMA特征
                    # 初始时权重较小，EMA影响较小，随着训练逐渐增强
                    x = x + self.fusion_weight * (x_ema - x)
                else:
                    x = x_ema
                
                # Restore shape: (1, C, N) -> (N, C)
                output_list.append(x.squeeze(0).transpose(0, 1))
            else:
                pass 
            start += cnt
            
        point.feat = torch.cat(output_list, dim=0) if output_list else feat
        return point
# =============================================================================


class Embedding(PointModule):
    def __init__(
        self,
        in_channels,
        embed_channels,
        norm_layer=None,
        act_layer=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels

        # TODO: check remove spconv
        self.stem = PointSequential(
            conv=spconv.SubMConv3d(
                in_channels,
                embed_channels,
                kernel_size=5,
                padding=1,
                bias=False,
                indice_key="stem",
            )
        )
        if norm_layer is not None:
            self.stem.add(norm_layer(embed_channels), name="norm")
        if act_layer is not None:
            self.stem.add(act_layer(), name="act")

    def forward(self, point: Point):
        point = self.stem(point)
        return point


class PointTransformerV3(PointModule):
    def __init__(
        self,
        in_channels=6,
        n_cls = 3, ####################
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2), 
        enc_depths=(2, 2, 2, 6, 2), 
        enc_channels=(32, 64, 128, 256, 512), 
        enc_num_head=(2, 4, 8, 16, 32), 
        enc_patch_size=(32, 32, 32, 32, 32),
        dec_depths=(2, 2, 2, 2), 
        dec_channels=(64, 64, 128, 256), 
        dec_num_head=(4, 4, 8, 16), 
        dec_patch_size=(32, 32, 32, 32), 
        mlp_ratio=4,  
        qkv_bias=True, 
        qk_scale=None, 
        attn_drop=0.0,
        proj_drop=0.0, 
        drop_path=0.3, 
        pre_norm=True,
        shuffle_orders=True,  
        enable_rpe=False,
        enable_flash=False,  # V100服务器不稳定支持flash-attn，默认走已有的普通attention分支
        upcast_attention=False ,
        upcast_softmax=False ,

        enable_ema=False,
        ema_factor=32,
        ema_stages=None,  # 指定哪些decoder stage使用EMA，None表示所有stage都使用
        ema_fusion_weight=0.1,  # EMA融合权重，初始值较小
        enable_grn=False,  # GRN在run3/run4测试文件上不稳定，默认关闭
        grn_eps=1e-6,  # GRN归一化的数值稳定项
        enable_ssm=False,  # SSM在run5-run8四文件推理上不稳定，默认关闭
        ssm_kernel_size=5,  # 使用较小卷积核做稳定的局部状态传播
        ssm_expansion=2,  # 轻量扩展倍率，避免显著增加显存
        ssm_stages=(1, 0),  # 默认只作用于高分辨率decoder stage，更贴近分割细节
        ssm_fusion_weight=0.1,  # 与残差分支融合的初始权重
        enable_rfg=True,  # 默认启用残差特征门控，作为下一轮更保守的模型主体改进
        rfg_reduction=4,  # 门控MLP压缩比例，保持参数量轻量
        rfg_stages=(0,),  # 先只放在最高分辨率decoder stage，优先稳定边界和细节
        rfg_fusion_weight=0.05,  # 初始融合权重更小，减少对原模型输出的扰动

        cls_mode=False, 
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
    ):
        super().__init__()
        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else order
        self.cls_mode = cls_mode
        self.shuffle_orders = shuffle_orders

        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(enc_num_head)
        assert self.num_stages == len(enc_patch_size)
        assert self.cls_mode or self.num_stages == len(dec_depths) + 1
        assert self.cls_mode or self.num_stages == len(dec_channels) + 1
        assert self.cls_mode or self.num_stages == len(dec_num_head) + 1
        assert self.cls_mode or self.num_stages == len(dec_patch_size) + 1

        # norm layers
        if pdnorm_bn:
            bn_layer = partial(
                PDNorm,
                norm_layer=partial(
                    nn.BatchNorm1d, eps=1e-3, momentum=0.01, affine=pdnorm_affine
                ),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        if pdnorm_ln:
            ln_layer = partial(
                PDNorm,
                norm_layer=partial(nn.LayerNorm, elementwise_affine=pdnorm_affine),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            ln_layer = nn.LayerNorm
        # activation layers
        act_layer = nn.GELU

        self.embedding = Embedding(
            in_channels=in_channels,
            embed_channels=enc_channels[0],
            norm_layer=bn_layer,
            act_layer=act_layer,
        )
        self.enable_ema = enable_ema
        self.ema_stages = ema_stages  # 如果为None，则所有stage都使用
        self.ema_fusion_weight = ema_fusion_weight
        self.enable_ssm = enable_ssm
        self.ssm_stages = set(ssm_stages) if ssm_stages is not None else None
        self.ssm_kernel_size = ssm_kernel_size
        self.ssm_expansion = ssm_expansion
        self.ssm_fusion_weight = ssm_fusion_weight
        self.enable_rfg = enable_rfg
        self.rfg_stages = set(rfg_stages) if rfg_stages is not None else None
        self.rfg_reduction = rfg_reduction
        self.rfg_fusion_weight = rfg_fusion_weight

        # encoder
        enc_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))
        ]
        self.enc = PointSequential()
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[
                sum(enc_depths[:s]) : sum(enc_depths[: s + 1])
            ]
            enc = PointSequential()
            if s > 0:
                enc.add(
                    SerializedPooling(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        stride=stride[s - 1],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="down",
                )
            for i in range(enc_depths[s]):
                enc.add(
                    Block(
                        channels=enc_channels[s],
                        num_heads=enc_num_head[s],
                        patch_size=enc_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=enc_drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        enable_rpe=enable_rpe,
                        enable_flash=enable_flash,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                        enable_grn=enable_grn,
                        grn_eps=grn_eps,
                    ),
                    name=f"block{i}",
                )
            if len(enc) != 0:
                self.enc.add(module=enc, name=f"enc{s}")

        # decoder
        if not self.cls_mode:
            dec_drop_path = [
                x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))
            ]
            self.dec = PointSequential()
            dec_channels = list(dec_channels) + [enc_channels[-1]]
            for s in reversed(range(self.num_stages - 1)):
                dec_drop_path_ = dec_drop_path[
                    sum(dec_depths[:s]) : sum(dec_depths[: s + 1])
                ]
                dec_drop_path_.reverse()
                dec = PointSequential()
                dec.add(
                    SerializedUnpooling(
                        in_channels=dec_channels[s + 1],
                        skip_channels=enc_channels[s],
                        out_channels=dec_channels[s],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="up",
                )
                for i in range(dec_depths[s]):
                    dec.add(
                        Block(
                            channels=dec_channels[s],
                            num_heads=dec_num_head[s],
                            patch_size=dec_patch_size[s],
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            drop_path=dec_drop_path_[i],
                            norm_layer=ln_layer,
                            act_layer=act_layer,
                            pre_norm=pre_norm,
                            order_index=i % len(self.order),
                            cpe_indice_key=f"stage{s}",
                            enable_rpe=enable_rpe,
                            enable_flash=enable_flash,
                            upcast_attention=upcast_attention,
                            upcast_softmax=upcast_softmax,
                            enable_grn=enable_grn,
                            grn_eps=grn_eps,
                        ),
                        name=f"block{i}",
                    )
                
                # [已修改] 将 EMA 模块插入到 Decoder Stage 之后，增强多尺度特征
                # 使用残差连接和可学习的融合权重，提高稳定性
                if self.enable_ema:
                    # 如果指定了ema_stages，只在指定stage使用；否则所有stage都使用
                    use_ema_here = (self.ema_stages is None) or (s in self.ema_stages)
                    if use_ema_here:
                        dec.add(
                            EMAPointAdapter(
                                channels=dec_channels[s], 
                                factor=ema_factor,
                                use_residual=True,  # 使用残差连接，保留原始特征
                                fusion_weight=self.ema_fusion_weight  # 可学习的融合权重
                            ),
                            name=f"ema{s}"
                        )

                if self.enable_rfg:
                    # RFG放在EMA之后，只对已融合的解码特征做轻量通道门控；
                    # 默认只作用于最高分辨率stage，避免像SSM一样对多个stage产生较强扰动。
                    use_rfg_here = (self.rfg_stages is None) or (s in self.rfg_stages)
                    if use_rfg_here:
                        dec.add(
                            PointResidualFeatureGate(
                                channels=dec_channels[s],
                                reduction=self.rfg_reduction,
                                fusion_weight=self.rfg_fusion_weight,
                            ),
                            name=f"rfg{s}"
                        )

                if self.enable_ssm:
                    # 只在高分辨率decoder stage注入序列混合，尽量保守地增强细粒度类别边界。
                    use_ssm_here = (self.ssm_stages is None) or (s in self.ssm_stages)
                    if use_ssm_here:
                        dec.add(
                            PointStateMixer(
                                channels=dec_channels[s],
                                kernel_size=self.ssm_kernel_size,
                                expansion=self.ssm_expansion,
                                proj_drop=proj_drop,
                                fusion_weight=self.ssm_fusion_weight,
                            ),
                            name=f"ssm{s}"
                        )
                
                self.dec.add(module=dec, name=f"dec{s}")

            # [已修改] 将 seg_head 移到循环外，避免重复定义
            self.seg_head = nn.Sequential(
                nn.Linear(dec_channels[0], dec_channels[0]),
                nn.BatchNorm1d(dec_channels[0]),
                nn.ReLU(inplace=True),
                nn.Linear(dec_channels[0], n_cls)
            )

    def forward(self, pxo):
        """
        A data_dict is a dictionary containing properties of a batched point cloud.
        """
        p0, x0, o0 = pxo  # (n, 3), (n, c), (b)
        data_dict = {}
        data_dict["feat"] = x0
        data_dict["coord"] = p0
        data_dict["offset"] = o0
        data_dict["grid_size"] = 0.04

        point = Point(data_dict)
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()

        point = self.embedding(point)
        point = self.enc(point)
        if not self.cls_mode:
            point = self.dec(point)

            
        x = self.seg_head(point.feat)
        return x
