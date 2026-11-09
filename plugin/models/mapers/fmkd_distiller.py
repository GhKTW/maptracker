"""
    FM-KD: FTKD with the BEV reconstruction replaced by conditional flow matching.
    FTKD's deterministic generator (masked student BEV -> teacher BEV, MSE) becomes a
    velocity field v(x_t, t | masked student BEV) on the linear path (rectified flow)
        x_t = (1 - t) * x_0 + t * x_1,  x_0 ~ N(0, I),  x_1 = teacher refined BEV
    trained with || v - (x_1 - x_0) ||^2. The student gets the KD signal through the
    condition. The query reconstruction and the logit distillation of FTKD are unchanged.
    BEV mask modes:
        'random': FTKD's position mask, every position dropped with bev_mask_ratio
        'ahead':  same mean ratio, but the positions ahead of the ego (x > 0, the A-mAP
                  half) are dropped with ahead_mask_ratio and the rear ones with
                  2 * bev_mask_ratio - ahead_mask_ratio
    Refine modes (E4 / E5), the flow net also runs inside the student, right after the BEV
    neck, and the decoder / seg head read the refined BEV in training and at inference:
        None:     training-only flow net (E1-E3)
        'noise':  x_0 ~ N(0, I), refined BEV = Euler integration to t=1 conditioned on the
                  student BEV
        'bridge': x_0 = the student BEV itself (current -> future bridge), the flow loss
                  uses x_t = (1 - t) * student BEV + t * teacher BEV, v* = teacher - student
    Training feeds the decoder a 1-step refinement with gradients, refine_steps applies at
    inference.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_mapper import MAPPERS
from .ftkd_distiller import FTKDDistiller


class TimeEmbedding(nn.Module):

    def __init__(self, dims, freq_dims=128):
        super().__init__()
        self.freq_dims = freq_dims
        self.mlp = nn.Sequential(nn.Linear(freq_dims, dims), nn.SiLU(), nn.Linear(dims, dims))

    def forward(self, t):
        half = self.freq_dims // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=t.dtype) / half)
        args = 1000 * t[:, None] * freqs[None]
        return self.mlp(torch.cat([args.cos(), args.sin()], dim=-1))


class FlowResBlock(nn.Module):
    """conv3x3 residual block, the time embedding gives a per-channel scale and shift."""

    def __init__(self, channels, time_dims, groups=32):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels, affine=False)
        self.film = nn.Linear(time_dims, 2 * channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.film(temb)[:, :, None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        return x + self.conv2(F.silu(h))


class BEVFlowNet(nn.Module):
    """v(x_t, t | cond) on the BEV map: concat(x_t, cond) -> conv -> res blocks -> conv."""

    def __init__(self, channels=256, hidden=256, num_blocks=2):
        super().__init__()
        self.time_embed = TimeEmbedding(hidden)
        self.proj_in = nn.Conv2d(2 * channels, hidden, 3, padding=1)
        self.blocks = nn.ModuleList([FlowResBlock(hidden, hidden) for _ in range(num_blocks)])
        self.norm_out = nn.GroupNorm(32, hidden)
        self.proj_out = nn.Conv2d(hidden, channels, 3, padding=1)

    def forward(self, x_t, t, cond):
        temb = self.time_embed(t)
        h = self.proj_in(torch.cat([x_t, cond], dim=1))
        for block in self.blocks:
            h = block(h, temb)
        return self.proj_out(F.silu(self.norm_out(h)))

    @torch.no_grad()
    def sample(self, cond, num_steps=1, x_0=None):
        """Euler integration from x_0 (default N(0, I)) at t=0 to t=1."""
        x = torch.randn_like(cond) if x_0 is None else x_0
        ts = torch.linspace(0, 1, num_steps + 1, device=cond.device, dtype=cond.dtype)
        for i in range(num_steps):
            t = ts[i].expand(len(x))
            x = x + (ts[i + 1] - ts[i]) * self(x, t, cond)
        return x


@MAPPERS.register_module()
class FMKDDistiller(FTKDDistiller):

    def __init__(self,
                 flow_hidden=256,
                 flow_num_blocks=2,
                 bev_mask_mode='random',
                 ahead_mask_ratio=0.75,
                 t_logit_mean=0.0,
                 t_logit_std=1.0,
                 refine_mode=None,
                 refine_steps=1,
                 **kwargs):
        """
        Args:
            bev_mask_mode: 'random' or 'ahead' (see the module docstring)
            ahead_mask_ratio: mask ratio of the ahead half in the 'ahead' mode
            t_logit_mean / t_logit_std: t = sigmoid(N(mean, std)) (logit-normal sampling)
            refine_mode: None, 'noise' or 'bridge' (see the module docstring)
            refine_steps: Euler steps of the inference-time refinement
            kwargs: FTKDDistiller arguments, loss_bev_recon_weight weights the flow loss
        """
        super().__init__(**kwargs)
        assert bev_mask_mode in ('random', 'ahead')
        if bev_mask_mode == 'ahead':
            assert 0 <= 2 * self.bev_mask_ratio - ahead_mask_ratio <= 1
        self.bev_mask_mode = bev_mask_mode
        self.ahead_mask_ratio = ahead_mask_ratio
        self.t_logit_mean = t_logit_mean
        self.t_logit_std = t_logit_std

        bev_channels = self.bev_generator[0].in_channels
        del self.bev_generator
        self.bev_flow = BEVFlowNet(bev_channels, flow_hidden, flow_num_blocks)

        assert refine_mode in (None, 'noise', 'bridge')
        self.refine_mode = refine_mode
        self.refine_steps = refine_steps
        if refine_mode is not None:
            # the bridge starts as the identity, the decoder first sees the plain student BEV
            nn.init.zeros_(self.bev_flow.proj_out.weight)
            nn.init.zeros_(self.bev_flow.proj_out.bias)
            self.student.bev_refine_fn = self.refine_bev

    def refine_bev(self, bev):
        """Called by the student after the BEV neck, keeps the unrefined BEV for the flow loss."""
        self._student_bev = bev
        x_0 = bev if self.refine_mode == 'bridge' else torch.randn_like(bev)
        if not self.training:
            return self.bev_flow.sample(bev, self.refine_steps, x_0=x_0)
        t = bev.new_zeros(len(bev))
        return x_0 + self.bev_flow(x_0, t, bev)

    def bev_keep_mask(self, feat):
        bs, _, h, w = feat.shape
        if self.bev_mask_mode == 'random':
            ratio = feat.new_full((1, 1, 1, w), self.bev_mask_ratio)
        else:
            # BEV columns follow x from -roi/2 to roi/2 (MapTracker.plane), x > 0 is ahead
            x = torch.linspace(-1, 1, w, device=feat.device)
            ratio = torch.where(x > 0, feat.new_tensor(self.ahead_mask_ratio),
                                feat.new_tensor(2 * self.bev_mask_ratio - self.ahead_mask_ratio)).view(1, 1, 1, w)
        return (torch.rand(bs, 1, h, w, device=feat.device) >= ratio).to(feat.dtype)

    def bev_recon_loss(self, feat_student, feat_teacher):
        """Flow matching loss towards the teacher BEV, conditioned on the masked student BEV."""
        if self.refine_mode is not None:
            # feat_student is the refined BEV, the flow is conditioned on the student's own
            feat_student = self._student_bev
        x_1 = feat_teacher.detach()
        x_0 = feat_student.detach() if self.refine_mode == 'bridge' else torch.randn_like(x_1)
        t = torch.sigmoid(self.t_logit_mean + self.t_logit_std * torch.randn(len(x_1), device=x_1.device))
        t_ = t.view(-1, 1, 1, 1)
        x_t = (1 - t_) * x_0 + t_ * x_1
        cond = feat_student * self.bev_keep_mask(feat_student)
        v = self.bev_flow(x_t, t, cond)
        return F.mse_loss(v, x_1 - x_0)
