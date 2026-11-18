"""
    11-15 oracle diagnosis: how far does the student's decoder (vector head + seg head) get when its input BEV
    is a cached teacher BEV instead of its own?  Image backbone / BEV encoder are bypassed (MapTracker.bev_override).
    bev_key = 'bev_refined'      -> F_T,    future-aware teacher head-input BEV
              'cur_bev_refined'  -> F_Tcur, same teacher with empty memory (current frame only)
              'f0_bev'           -> F_0,    the KD-free bs64 student's own BEV (control)
    train BEVs: teacher_cache (train tokens), val BEVs: val_cache (val tokens, tools/oracle_val_cache.py).
    Not deployable (needs future frames for F_T); only answers "is the BEV or the decoder the bottleneck".

    11-17 predictability ceiling: an optional completion module G sits between the cached input BEV and the decoder,
        BEV_in (bev_key, e.g. F_Tcur) -> BEV_in + G(BEV_in) -> seg head + vector head   (G zero-init: starts as the plain oracle)
    completer = 'attn'      : 2x downsample -> global self-attention (every cell sees the whole 60 x 30 m map) -> upsample
                'local'     : the old FuturePredictor of branch_kd.py (3x3 convs, +-7 cells = +-4.2 m), capacity control
                'conv_long' : conv-only hourglass, 4x downsample + dilated blocks (+-40 cells = +-24 m at the bottleneck)
    loss_kd_weight > 0 adds  lambda * MSE(BEV_in + G(BEV_in), F_T)  (the future teacher's BEV of the same sample);
    kd_delta_raw (logged always, no gradient unless lambda > 0) is that per-element MSE; its value at iteration 1
    (G = 0, about 0.18 with the training-time batch) = "G predicts nothing of Delta".
"""
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F

from .amap_distiller import AMapDistiller
from .base_mapper import MAPPERS
from .branch_kd import FuturePredictor, ResBlock


class _AttnLayer(nn.Module):
    """Pre-norm transformer encoder layer with a hand-written attention (matmul + softmax): on this server torch 2.0.1's
    nn.MultiheadAttention / scaled_dot_product_attention die in backward with 'CUDA error: an illegal instruction'."""

    def __init__(self, dim, num_heads, ffn_dim):
        super().__init__()
        self.h = num_heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, dim))

    def forward(self, x):
        b, n, c = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(b, n, 3, self.h, c // self.h).permute(2, 0, 3, 1, 4)
        a = (torch.matmul(q, k.transpose(-1, -2)) * (c // self.h) ** -0.5).softmax(-1)
        x = x + self.proj(torch.matmul(a, v).transpose(1, 2).reshape(b, n, c))
        return x + self.ffn(self.norm2(x))


class AttnCompleter(nn.Module):
    """Global-context completion: stride-2 conv -> (H/2 * W/2) tokens + learned positions -> pre-norm transformer
    encoder layers -> bilinear upsample + 3x3 conv (zero-init). Output is the residual added to the input BEV."""

    def __init__(self, channels=256, num_layers=3, num_heads=8, ffn_dim=512, bev_hw=(50, 100)):
        super().__init__()
        self.down = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
        self.pos = nn.Parameter(torch.zeros(1, channels, bev_hw[0] // 2, bev_hw[1] // 2))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.layers = nn.Sequential(*[_AttnLayer(channels, num_heads, ffn_dim) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(channels)
        self.out = nn.Conv2d(channels, channels, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        h = self.down(x) + self.pos
        b, c, hh, ww = h.shape
        h = self.norm(self.layers(h.flatten(2).transpose(1, 2))).transpose(1, 2).reshape(b, c, hh, ww)
        return self.out(F.interpolate(h, size=x.shape[-2:], mode='bilinear', align_corners=False))


class ConvLongCompleter(nn.Module):
    """Conv-only long-range completion (hourglass): 3x3 in -> two stride-2 convs (4x down, 2.4 m cells) -> dilated
    residual blocks (1, 2, 4) -> two upsample+conv steps with skip sums -> 3x3 out (zero-init)."""

    def __init__(self, channels=256, hidden=128, dilations=(1, 2, 4)):
        super().__init__()
        self.proj_in = nn.Conv2d(channels, hidden, 3, padding=1)
        self.down1 = nn.Conv2d(hidden, hidden, 3, stride=2, padding=1)
        self.down2 = nn.Conv2d(hidden, hidden, 3, stride=2, padding=1)
        self.blocks = nn.Sequential(*[ResBlock(hidden, d, zero_init=False) for d in dilations])
        self.up2 = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.up1 = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.norm_out = nn.GroupNorm(32, hidden)
        self.proj_out = nn.Conv2d(hidden, channels, 3, padding=1)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x):
        h0 = self.proj_in(x)
        h1 = self.down1(F.silu(h0))
        h2 = self.blocks(self.down2(F.silu(h1)))
        u1 = h1 + self.up2(F.interpolate(h2, size=h1.shape[-2:], mode='bilinear', align_corners=False))
        u0 = h0 + self.up1(F.interpolate(F.silu(u1), size=h0.shape[-2:], mode='bilinear', align_corners=False))
        return self.proj_out(F.silu(self.norm_out(u0)))


@MAPPERS.register_module()
class OracleBEVMapper(AMapDistiller):

    def __init__(self, bev_key='bev_refined', val_cache=None, completer=None, loss_kd_weight=0.0,
                 kd_key='bev_refined', **kwargs):
        kwargs.setdefault('loss_feat_basic_weight', 0.0)
        kwargs.setdefault('loss_feat_refined_weight', 0.0)
        kwargs.setdefault('loss_logits_weight', 0.0)
        super().__init__(**kwargs)
        assert self.teacher_cache is not None and val_cache is not None
        self.bev_key = bev_key
        self.val_cache = val_cache
        assert completer in (None, 'attn', 'local', 'conv_long')
        self.completer = None
        if completer == 'attn':
            self.completer = AttnCompleter(256, bev_hw=(self.student.bev_h, self.student.bev_w))
        elif completer == 'local':
            self.completer = FuturePredictor(256, 256)
        elif completer == 'conv_long':
            self.completer = ConvLongCompleter(256)
        self.loss_kd_weight = loss_kd_weight
        self.kd_key = kd_key

    def _load_bev(self, folder, img_metas, device, keys=None):
        keys = keys or (self.bev_key,)
        cached = [torch.load(osp.join(folder, f"{m['token']}.pt"), map_location='cpu') for m in img_metas]
        outs = [torch.stack([c[k] for c in cached]).to(device).float() for k in keys]
        return outs[0] if len(outs) == 1 else outs

    def _complete(self, bev):
        return bev if self.completer is None else bev + self.completer(bev)

    def forward_train(self, img, vectors, semantic_mask, points=None, img_metas=None, all_prev_data=None,
                      all_local2global_info=None, **kwargs):
        if self.completer is None:
            bev, target = self._load_bev(self.teacher_cache, img_metas, img.device), None
        else:
            bev, target = self._load_bev(self.teacher_cache, img_metas, img.device, (self.bev_key, self.kd_key))
        bev = self._complete(bev)
        self.student.bev_override = bev
        try:
            loss, log_vars, num_sample = self.student.forward_train(
                img, vectors, semantic_mask, points=points, img_metas=img_metas)
        finally:
            self.student.bev_override = None
        if target is not None:
            if self.loss_kd_weight > 0:
                mse = F.mse_loss(bev, target)
                loss = loss + self.loss_kd_weight * mse
                log_vars['kd_delta'] = self.loss_kd_weight * mse.item()
                log_vars['total'] = loss.item()
            else:
                with torch.no_grad():
                    mse = F.mse_loss(bev, target)
            log_vars['kd_delta_raw'] = mse.item()
        return loss, log_vars, num_sample

    def forward_test(self, img, points=None, img_metas=None, seq_info=None, **kwargs):
        self.student.bev_override = self._complete(self._load_bev(self.val_cache, img_metas, img.device))
        try:
            return self.student.forward_test(img, points=points, img_metas=img_metas, seq_info=seq_info, **kwargs)
        finally:
            self.student.bev_override = None
