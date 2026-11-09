"""
    FTKD (Zheng et al., "Distilling Future Temporal Knowledge with Masked Feature
    Reconstruction for 3D Object Detection", AAAI 2026) adapted to MapTracker, with the
    same frozen future-aware teacher and current-frame student as AMap.
    - Future-aware feature reconstruction (FFR): random masks on the student features, an
      auxiliary generator recovers them, supervised by the teacher features (MSE)
        * dense BEV map (the paper's PV features): mask BEV positions, 2x conv3x3 generator,
          target = the teacher's refined BEV, which already fuses its future BEV memory
        * queries (the paper's sparse queries): mask whole queries, FFN + LayerNorm generator,
          target = the Hungarian-matched teacher query of the last decoder layer
    - Future-guided logit distillation (FLD): focal loss towards the teacher's class
      probabilities + L1 towards the teacher's lines, on all matched queries (foreground and
      background)
    The generators are training-only, the student is unchanged at inference.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .amap_distiller import AMapDistiller
from .base_mapper import MAPPERS


@MAPPERS.register_module()
class FTKDDistiller(AMapDistiller):

    def __init__(self,
                 bev_channels=256,
                 query_channels=512,
                 bev_mask_ratio=0.5,
                 query_mask_ratio=0.5,
                 loss_bev_recon_weight=1.0,
                 loss_query_recon_weight=16.0,
                 loss_fld_weight=1.0,
                 fld_cls_weight=2.0,
                 fld_reg_weight=0.25,
                 **kwargs):
        """
        Args:
            loss_bev_recon_weight / loss_query_recon_weight / loss_fld_weight: lambda 1-3 of
                the paper's Eq. 7
            fld_cls_weight / fld_reg_weight: alpha / beta of Eq. 6
            kwargs: AMapDistiller arguments (student, teacher, cache, matching costs). Its
                AMap loss weights are not used.
        """
        super().__init__(**kwargs)
        self.bev_mask_ratio = bev_mask_ratio
        self.query_mask_ratio = query_mask_ratio
        self.loss_bev_recon_weight = loss_bev_recon_weight
        self.loss_query_recon_weight = loss_query_recon_weight
        self.loss_fld_weight = loss_fld_weight
        self.fld_cls_weight = fld_cls_weight
        self.fld_reg_weight = fld_reg_weight

        self.bev_generator = nn.Sequential(
            nn.Conv2d(bev_channels, bev_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(bev_channels, bev_channels, 3, padding=1))
        self.query_generator = nn.Sequential(
            nn.Linear(query_channels, query_channels * 2),
            nn.ReLU(inplace=True),
            nn.Linear(query_channels * 2, query_channels))
        self.query_generator_norm = nn.LayerNorm(query_channels)

    def forward_train(self, img, vectors, semantic_mask, points=None, img_metas=None, all_prev_data=None,
                      all_local2global_info=None, **kwargs):
        assert self.teacher is None, 'FTKD reads the teacher outputs from the cache'
        teacher_outs = self._load_teacher_cache(img_metas, img.device)
        assert 'hs_embeds' in teacher_outs, 'the teacher cache has no query features'

        loss, log_vars, num_sample, student_outs = self.student.forward_train(
            img, vectors, semantic_mask, points=points, img_metas=img_metas, return_kd_outputs=True)

        kd_losses = {}
        if self.loss_bev_recon_weight > 0:
            kd_losses['ftkd_bev_recon'] = self.loss_bev_recon_weight * self.bev_recon_loss(
                student_outs['bev_refined'], teacher_outs['bev_refined'])
        if self.loss_query_recon_weight > 0 or self.loss_fld_weight > 0:
            query_recon, fld_cls, fld_reg = self.query_losses(student_outs, teacher_outs)
            if self.loss_query_recon_weight > 0:
                kd_losses['ftkd_query_recon'] = self.loss_query_recon_weight * query_recon
            if self.loss_fld_weight > 0:
                kd_losses['ftkd_fld_cls'] = self.loss_fld_weight * self.fld_cls_weight * fld_cls
                kd_losses['ftkd_fld_reg'] = self.loss_fld_weight * self.fld_reg_weight * fld_reg

        for name, kd_loss in kd_losses.items():
            loss = loss + kd_loss
            log_vars[name] = kd_loss.item()
        log_vars['total'] = loss.item()

        return loss, log_vars, num_sample

    def bev_recon_loss(self, feat_student, feat_teacher):
        """Eq. 2-4 on the dense BEV map: the mask drops BEV positions (all channels), the
        MSE is averaged over every position and channel."""
        bs, _, h, w = feat_student.shape
        keep = (torch.rand(bs, 1, h, w, device=feat_student.device) >= self.bev_mask_ratio).to(feat_student.dtype)
        recon = self.bev_generator(feat_student * keep)
        return F.mse_loss(recon, feat_teacher.detach())

    def query_losses(self, student_outs, teacher_outs):
        """Eq. 5 (masked query reconstruction) and Eq. 6 (logit distillation), both on the
        one-to-one Hungarian matching between the student and teacher predictions."""
        num_pts = student_outs['lines'][0].shape[-1] // 2
        recon_losses, cls_losses, reg_losses = [], [], []
        for b_i, (lines_s, scores_s, lines_t, scores_t) in enumerate(zip(
                student_outs['lines'], student_outs['scores'], teacher_outs['lines'], teacher_outs['scores'])):
            student_ids, teacher_ids = self.match_queries(lines_s.detach(), scores_s.detach(),
                                                          lines_t, scores_t)
            # masked query reconstruction
            hs_s = student_outs['hs_embeds'][b_i][student_ids]
            hs_t = teacher_outs['hs_embeds'][b_i][teacher_ids].detach()
            keep = (torch.rand(len(hs_s), 1, device=hs_s.device) >= self.query_mask_ratio).to(hs_s.dtype)
            masked = hs_s * keep
            recon = self.query_generator_norm(masked + self.query_generator(masked))
            recon_losses.append(F.mse_loss(recon, hs_t))

            # focal loss with the teacher probabilities as soft targets (sigmoid classifier)
            logits_s = scores_s[student_ids]
            p_t = scores_t[teacher_ids].detach().sigmoid()
            p_s = logits_s.sigmoid()
            focal = F.binary_cross_entropy_with_logits(logits_s, p_t, reduction='none') * \
                (p_s - p_t).abs().pow(2)
            cls_losses.append(focal.sum(-1).mean())

            # L1 towards the teacher lines, whichever direction is closer
            pred = lines_s[student_ids]
            target = lines_t[teacher_ids].detach()
            target_rev = target.view(-1, num_pts, 2).flip(1).reshape(len(target), -1)
            l1 = torch.min((pred - target).abs().mean(-1), (pred - target_rev).abs().mean(-1))
            reg_losses.append(l1.mean())

        return torch.stack(recon_losses).mean(), torch.stack(cls_losses).mean(), torch.stack(reg_losses).mean()
