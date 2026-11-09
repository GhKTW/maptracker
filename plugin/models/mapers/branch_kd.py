"""
    Branched current / future KD (2026-11-05 design).
    The frozen future teacher's BEV is split as F_T = F_Tcur + Delta, F_Tcur being the same
    teacher run on the current frame only (empty memory). The student gets
        B_cur  : its own neck BEV (current branch)
        B_fut  : P(B_cur), a light future-prediction module (future branch)
        fused  : B_cur + B_fut, read by the seg head and the vector decoder
    and the losses (all optional, see BranchKDDistiller):
        loss_cur_weight   * MSE(B_cur, F_Tcur)
        loss_fut_weight   * mean_x w(x) * MSE_c(B_fut(x), target(x)),
                            target = Delta or the frozen probe's mean of Delta
        loss_fused_weight * MSE(fused, F_T)                  (single-branch control)
        loss_smg_weight   * SMG InfoNCE(fused, F_T) on the G.T. boxes
        loss_logits_weight* Hungarian-matched logit KL (AMap)
    w(x): 'none' | 'gt' (G.T. fg mask) | 'inv_var' (clip(1/sigma^2) of the probe) |
          'reducible' (max(e_S - sigma^2, 0), student error above the probe's floor),
          normalized to mean 1 over the batch.
    BEVProbe: Stage 0 predictability probe, standardized F_0 -> (mu, log sigma^2) of the
    standardized Delta, trained by tools/train_probe.py with the heteroscedastic NLL.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .amap_distiller import AMapDistiller
from .base_mapper import MAPPERS
from .labelkd_distiller import LabelKDDistiller


class ResBlock(nn.Module):

    def __init__(self, channels, dilation=1, zero_init=True):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.norm2 = nn.GroupNorm(32, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        if zero_init:
            nn.init.zeros_(self.conv2.weight)
            nn.init.zeros_(self.conv2.bias)

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        return x + self.conv2(F.silu(self.norm2(h)))


class FuturePredictor(nn.Module):
    """B_cur -> B_fut, zero output at init (the decoder first sees the plain student BEV)."""

    def __init__(self, channels=256, hidden=256, dilations=(1, 2)):
        super().__init__()
        self.proj_in = nn.Conv2d(channels, hidden, 3, padding=1)
        self.blocks = nn.Sequential(*[ResBlock(hidden, d) for d in dilations])
        self.norm_out = nn.GroupNorm(32, hidden)
        self.proj_out = nn.Conv2d(hidden, channels, 3, padding=1)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x):
        return self.proj_out(F.silu(self.norm_out(self.blocks(self.proj_in(x)))))


class BEVProbe(nn.Module):
    """Standardized F_0 -> (mu: C x H x W, logvar: 1 x H x W) of the standardized target."""

    def __init__(self, in_channels=256, out_channels=256, hidden=256, dilations=(1, 2, 4, 8)):
        super().__init__()
        self.proj_in = nn.Conv2d(in_channels, hidden, 3, padding=1)
        self.blocks = nn.Sequential(*[ResBlock(hidden, d, zero_init=False) for d in dilations])
        self.norm_out = nn.GroupNorm(32, hidden)
        self.mu = nn.Conv2d(hidden, out_channels, 1)
        self.logvar = nn.Conv2d(hidden, 1, 1)
        nn.init.zeros_(self.logvar.weight)
        nn.init.zeros_(self.logvar.bias)

    def forward(self, x):
        h = F.silu(self.norm_out(self.blocks(self.proj_in(x))))
        return self.mu(h), self.logvar(h).clamp(-10, 10)


@MAPPERS.register_module()
class BranchKDDistiller(AMapDistiller):

    element_boxes = LabelKDDistiller.element_boxes
    contrastive_loss = LabelKDDistiller.contrastive_loss

    def __init__(self,
                 branch=True,
                 pred_hidden=256,
                 pred_detach_input=False,
                 loss_cur_weight=1.0,
                 loss_fut_weight=1.0,
                 loss_fused_weight=0.0,
                 fut_target='delta',
                 fut_weight='none',
                 probe_ckpt=None,
                 w_clip=(0.1, 10.0),
                 weight_warmup_iters=0,
                 loss_smg_weight=0.0,
                 contrast_tau=0.1,
                 boundary_label=2,
                 eval_fut_off=False,
                 log_fut_split=False,
                 **kwargs):
        kwargs.setdefault('loss_feat_basic_weight', 0.0)
        kwargs.setdefault('loss_feat_refined_weight', 0.0)
        kwargs.setdefault('loss_logits_weight', 0.0)
        super().__init__(**kwargs)
        assert self.teacher_cache is not None, 'the branched KD reads the joint cache'
        assert fut_target in ('delta', 'probe_mu')
        assert fut_weight in ('none', 'gt', 'inv_var', 'reducible')
        self.branch = branch
        self.pred_detach_input = pred_detach_input
        self.loss_cur_weight = loss_cur_weight
        self.loss_fut_weight = loss_fut_weight if branch else 0.0
        self.loss_fused_weight = loss_fused_weight
        self.fut_target = fut_target
        self.fut_weight = fut_weight
        self.w_clip = w_clip
        self.weight_warmup_iters = weight_warmup_iters
        self.loss_smg_weight = loss_smg_weight
        self.contrast_tau = contrast_tau
        self.boundary_label = boundary_label
        self.contrast_classes = None
        self.eval_fut_off = eval_fut_off
        # monitoring only (no gradient): split the future-branch error with the frozen probe mean mu
        #   |B_fut - Delta|^2 = |B_fut - mu|^2 (predictable part) + |Delta - mu|^2 (floor) + cross term
        self.log_fut_split = log_fut_split
        self._split = None

        if branch:
            self.predictor = FuturePredictor(256, pred_hidden)
            self.student.bev_refine_fn = self.fuse_bev

        self.probe = None
        if fut_target == 'probe_mu' or fut_weight in ('inv_var', 'reducible') or log_fut_split:
            assert probe_ckpt is not None
            ckpt = torch.load(probe_ckpt, map_location='cpu')
            self.probe = BEVProbe(**ckpt['probe_args'])
            self.probe.load_state_dict(ckpt['state_dict'])
            for p in self.probe.parameters():
                p.requires_grad = False
            for k in ('f0_mean', 'f0_std', 'tgt_mean', 'tgt_std'):
                self.register_buffer(k, ckpt[k].view(1, -1, 1, 1).float(), persistent=False)
        self._b_cur = self._b_fut = self._gts = None

    def train(self, mode=True):
        super().train(mode)
        if self.probe is not None:
            self.probe.eval()
        return self

    def fuse_bev(self, bev):
        self._b_cur = bev
        b_fut = self.predictor(bev.detach() if self.pred_detach_input else bev)
        self._b_fut = b_fut
        if not self.training and self.eval_fut_off:
            return bev
        return bev + b_fut

    def _load_teacher_cache(self, img_metas, device):
        cached = [torch.load(f"{self.teacher_cache}/{meta['token']}.pt", map_location='cpu')
                  for meta in img_metas]
        outs = {k: torch.stack([c[k] for c in cached]).to(device).float()
                for k in ('bev_refined', 'cur_bev_refined', 'gt_fg')}
        if self.probe is not None:
            outs['f0_bev'] = torch.stack([c['f0_bev'] for c in cached]).to(device).float()
        for k in ('lines', 'scores'):
            outs[k] = [c[k].to(device).float() for c in cached]
        return outs

    @staticmethod
    def _mean_one(w):
        return w / w.mean().clamp(min=1e-6)

    def fut_loss(self, b_fut, t):
        delta = t['bev_refined'] - t['cur_bev_refined']
        sigma2 = None
        if self.probe is not None:
            with torch.no_grad():
                mu, logvar = self.probe((t['f0_bev'] - self.f0_mean) / self.f0_std)
            sigma2 = logvar.exp()[:, 0] # (bs, H, W), standardized units, mean over channels
            mu_raw = mu * self.tgt_std + self.tgt_mean
            if self.log_fut_split:
                with torch.no_grad():
                    self._split = (((b_fut.detach() - mu_raw) ** 2).mean().item(),
                                   ((delta - mu_raw) ** 2).mean().item())
            if self.fut_target == 'probe_mu':
                delta = mu_raw
        err = (b_fut - delta.detach()) ** 2
        err = err.mean(1) # (bs, H, W)

        if self.fut_weight == 'none' or self.student.num_iter < self.weight_warmup_iters:
            w = torch.ones_like(err)
        elif self.fut_weight == 'gt':
            w = t['gt_fg'][:, 0]
            w = w if w.sum() > 0 else torch.ones_like(err)
        elif self.fut_weight == 'inv_var':
            w = (1.0 / sigma2).clamp(*self.w_clip)
        else:
            # student error and the probe floor in the same standardized units
            e_s = (((b_fut.detach() - delta) / self.tgt_std) ** 2).mean(1)
            w = (e_s - sigma2).clamp(min=0)
            w = w if w.sum() > 0 else torch.ones_like(err)
        w = self._mean_one(w.detach())
        return (w * err).mean(), w

    def forward_train(self, img, vectors, semantic_mask, points=None, img_metas=None, all_prev_data=None,
                      all_local2global_info=None, **kwargs):
        t = self._load_teacher_cache(img_metas, img.device)
        if self.loss_smg_weight > 0:
            self._gts = self.student.batch_data(vectors, None, None, img.device)[0]
        try:
            loss, log_vars, num_sample, s = self.student.forward_train(
                img, vectors, semantic_mask, points=points, img_metas=img_metas, return_kd_outputs=True)
            fused = s['bev_refined']
            b_cur = self._b_cur if self.branch else fused

            kd = {}
            if self.loss_cur_weight > 0:
                kd['bkd_cur'] = self.loss_cur_weight * F.mse_loss(b_cur, t['cur_bev_refined'])
            if self.loss_fut_weight > 0:
                l_fut, w = self.fut_loss(self._b_fut, t)
                kd['bkd_fut'] = self.loss_fut_weight * l_fut
                log_vars['bkd_w_max'] = w.max().item()
                if self._split is not None:
                    # same scale as bkd_fut (x loss_fut_weight), unweighted by w(x)
                    log_vars['bkd_fut_pred'] = self.loss_fut_weight * self._split[0]
                    log_vars['bkd_fut_floor'] = self.loss_fut_weight * self._split[1]
                    self._split = None
            if self.loss_fused_weight > 0:
                kd['bkd_fused'] = self.loss_fused_weight * F.mse_loss(fused, t['bev_refined'])
            if self.loss_smg_weight > 0:
                kd['bkd_smg'] = self.loss_smg_weight * self.contrastive_loss(fused, t['bev_refined'], self._gts)
            if self.loss_logits_weight > 0:
                l_logits, nq = self.matched_logits_loss(s, t)
                kd['bkd_logits'] = self.loss_logits_weight * l_logits
            if self.branch:
                log_vars['bkd_fut_rms'] = self._b_fut.detach().pow(2).mean().sqrt().item()
        finally:
            self._gts = None
            self._b_cur = self._b_fut = None

        for name, v in kd.items():
            loss = loss + v
            log_vars[name] = v.item()
        log_vars['total'] = loss.item()
        return loss, log_vars, num_sample
