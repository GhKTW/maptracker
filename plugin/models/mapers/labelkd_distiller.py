"""
    Label distillation: the frozen label teacher (LabelMapTeacher, G.T. map -> label BEV)
    guides the current-frame R18 student. The teacher runs online, it only needs the G.T.
    BEV distillation (loss_type, one of them or a list):
        'fg_mse':      LabelDistill Eq. 1, 1x1 conv adapter on the student BEV, MSE on the
                       G.T. foreground cells (the painted polylines)
        'mgd':         MGD / FTKD masked reconstruction, the student BEV is masked
                       ('random' or 'ahead', see FMKDDistiller), a conv generator recovers
                       the label BEV, MSE on every cell
        'contrastive': SMG (Ren et al., WACV 2026), symmetric InfoNCE between each G.T.
                       element's pooled student BEV and its pooled label BEV, pooled over the
                       element's box (two boxes for the road boundaries, "DualBox-bou");
                       with a BEV mask, a conv generator first recovers the masked student
                       BEV (MGD-style) and the generated features are aligned
    Optionally the FTKD query losses (masked query reconstruction + logit distillation)
    towards the teacher decoder's outputs on the label BEV.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet.utils import get_root_logger
from mmdet3d.models.builder import build_model

from .amap_distiller import AMapDistiller
from .base_mapper import BaseMapper, MAPPERS
from .ftkd_distiller import FTKDDistiller


@MAPPERS.register_module()
class LabelKDDistiller(BaseMapper):

    # query matching and FTKD query losses, shared with the cached-teacher distillers
    match_queries = AMapDistiller.match_queries
    query_losses = FTKDDistiller.query_losses

    def __init__(self,
                 student_cfg,
                 teacher_config,
                 teacher_ckpt,
                 loss_type='mgd',
                 loss_bev_weight=1.0,
                 bev_channels=256,
                 bev_mask_mode='random',
                 bev_mask_ratio=0.5,
                 ahead_mask_ratio=0.75,
                 contrast_tau=0.1,
                 boundary_label=2,
                 query_channels=512,
                 query_mask_ratio=0.5,
                 loss_query_recon_weight=0.0,
                 loss_fld_weight=0.0,
                 fld_cls_weight=2.0,
                 fld_reg_weight=0.25,
                 match_cls_weight=5.0,
                 match_reg_weight=50.0,
                 **kwargs):
        """
        Args:
            teacher_config / teacher_ckpt: the trained LabelMapTeacher
            loss_type: 'fg_mse', 'mgd', 'contrastive' or a list of them, each weighted by
                loss_bev_weight (see the module docstring)
            bev_mask_mode: 'none', 'random' or 'ahead' ('mgd' and 'contrastive')
            contrast_tau: InfoNCE temperature ('contrastive' only)
            boundary_label: class id of the road boundaries, pooled over two boxes
            loss_query_recon_weight / loss_fld_weight: FTKD lambda 2 / 3, 0 disables
        """
        super().__init__()
        self.student = build_model(student_cfg)
        assert not self.student.temporal
        self.teacher = build_model(Config.fromfile(teacher_config).model)
        self.teacher_ckpt = teacher_ckpt

        loss_types = [loss_type] if isinstance(loss_type, str) else list(loss_type)
        assert loss_types and all(t in ('fg_mse', 'mgd', 'contrastive') for t in loss_types)
        assert bev_mask_mode in ('none', 'random', 'ahead')
        if bev_mask_mode == 'ahead':
            assert 0 <= 2 * bev_mask_ratio - ahead_mask_ratio <= 1
        self.loss_types = loss_types
        self.loss_bev_weight = loss_bev_weight
        self.bev_mask_mode = bev_mask_mode
        self.bev_mask_ratio = bev_mask_ratio
        self.ahead_mask_ratio = ahead_mask_ratio
        self.contrast_tau = contrast_tau
        self.boundary_label = boundary_label
        self.query_mask_ratio = query_mask_ratio
        self.loss_query_recon_weight = loss_query_recon_weight
        self.loss_fld_weight = loss_fld_weight
        self.fld_cls_weight = fld_cls_weight
        self.fld_reg_weight = fld_reg_weight
        self.match_cls_weight = match_cls_weight
        self.match_reg_weight = match_reg_weight

        if 'fg_mse' in loss_types:
            self.bev_adapter = nn.Conv2d(bev_channels, bev_channels, 1)
        if 'mgd' in loss_types or ('contrastive' in loss_types and bev_mask_mode != 'none'):
            self.bev_generator = nn.Sequential(
                nn.Conv2d(bev_channels, bev_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(bev_channels, bev_channels, 3, padding=1))
        if loss_query_recon_weight > 0:
            self.query_generator = nn.Sequential(
                nn.Linear(query_channels, query_channels * 2),
                nn.ReLU(inplace=True),
                nn.Linear(query_channels * 2, query_channels))
            self.query_generator_norm = nn.LayerNorm(query_channels)
        self._freeze_teacher()

    def init_weights(self, pretrained=None):
        self.student.init_weights()
        load_checkpoint(self.teacher, self.teacher_ckpt, map_location='cpu', strict=True,
                        logger=get_root_logger())
        self._freeze_teacher()

    def _freeze_teacher(self):
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        # the teacher's train() refreezes its decoder, the encoder is frozen here
        self._freeze_teacher()
        return self

    @property
    def num_iter(self):
        return self.student.num_iter

    @num_iter.setter
    def num_iter(self, value):
        self.student.num_iter = value
        self.teacher.num_iter = value

    def forward_train(self, img, vectors, semantic_mask, points=None, img_metas=None, all_prev_data=None,
                      all_local2global_info=None, **kwargs):
        loss, log_vars, num_sample, student_outs = self.student.forward_train(
            img, vectors, semantic_mask, points=points, img_metas=img_metas, return_kd_outputs=True)

        with torch.no_grad():
            label_bev, fg, gts = self.teacher.encode(vectors, img.device)
        bev_s = student_outs['bev_refined']

        kd_losses = {}
        for loss_type in self.loss_types:
            if loss_type == 'fg_mse':
                bev_loss = self.fg_mse_loss(bev_s, label_bev, fg)
            elif loss_type == 'mgd':
                bev_loss = self.mgd_loss(bev_s, label_bev)
            else:
                feat_s = bev_s
                if self.bev_mask_mode != 'none':
                    feat_s = self.bev_generator(bev_s * self.bev_keep_mask(bev_s))
                bev_loss = self.contrastive_loss(feat_s, label_bev, gts)
            # the single-loss runs keep their 'lkd_bev' log name
            name = 'lkd_bev' if len(self.loss_types) == 1 else f'lkd_{loss_type}'
            kd_losses[name] = self.loss_bev_weight * bev_loss

        if self.loss_query_recon_weight > 0 or self.loss_fld_weight > 0:
            with torch.no_grad():
                preds_list = self.teacher.mapper.head(
                    bev_features=label_bev, img_metas=img_metas, gts=gts, track_query_info=None,
                    memory_bank=None, return_loss=True)[0]
            teacher_outs = {k: preds_list[-1][k] for k in ('lines', 'scores', 'hs_embeds')}
            query_recon, fld_cls, fld_reg = self.query_losses(student_outs, teacher_outs)
            if self.loss_query_recon_weight > 0:
                kd_losses['lkd_query_recon'] = self.loss_query_recon_weight * query_recon
            if self.loss_fld_weight > 0:
                kd_losses['lkd_fld_cls'] = self.loss_fld_weight * self.fld_cls_weight * fld_cls
                kd_losses['lkd_fld_reg'] = self.loss_fld_weight * self.fld_reg_weight * fld_reg

        for name, kd_loss in kd_losses.items():
            loss = loss + kd_loss
            log_vars[name] = kd_loss.item()
        log_vars['total'] = loss.item()
        return loss, log_vars, num_sample

    def bev_keep_mask(self, feat):
        bs, _, h, w = feat.shape
        if self.bev_mask_mode == 'none':
            return feat.new_ones(bs, 1, h, w)
        if self.bev_mask_mode == 'random':
            ratio = feat.new_full((1, 1, 1, w), self.bev_mask_ratio)
        else:
            # BEV columns follow x from -roi/2 to roi/2 (MapTracker.plane), x > 0 is ahead
            x = torch.linspace(-1, 1, w, device=feat.device)
            ratio = torch.where(x > 0, feat.new_tensor(self.ahead_mask_ratio),
                                feat.new_tensor(2 * self.bev_mask_ratio - self.ahead_mask_ratio)).view(1, 1, 1, w)
        return (torch.rand(bs, 1, h, w, device=feat.device) >= ratio).to(feat.dtype)

    def fg_mse_loss(self, bev_s, label_bev, fg):
        """MSE (channel mean) on the G.T. foreground cells, averaged over those cells."""
        fg = fg[:, None]
        sq = (self.bev_adapter(bev_s) - label_bev).pow(2).mean(1, keepdim=True)
        return (sq * fg).sum() / fg.sum().clamp(min=1.0)

    def mgd_loss(self, bev_s, label_bev):
        recon = self.bev_generator(bev_s * self.bev_keep_mask(bev_s))
        return F.mse_loss(recon, label_bev)

    def element_boxes(self, labels, lines, H, W):
        """(num_boxes, H, W) box masks on the BEV grid and their element index; the road
        boundaries are split into the first and the second half of their points."""
        pts = lines.view(len(lines), -1, 2)
        half = pts.shape[1] // 2
        groups, owners = [], []
        for i in range(len(pts)):
            if labels[i] == self.boundary_label:
                groups += [pts[i, :half + 1], pts[i, half:]]
                owners += [i, i]
            else:
                groups.append(pts[i])
                owners.append(i)
        p = torch.stack([torch.cat([g.min(0)[0], g.max(0)[0]]) for g in groups]) # x0 y0 x1 y1
        c0 = (p[:, 0] * (W - 1)).round().long()
        c1 = (p[:, 2] * (W - 1)).round().long()
        r0 = ((1 - p[:, 3]) * (H - 1)).round().long()
        r1 = ((1 - p[:, 1]) * (H - 1)).round().long()
        rows = torch.arange(H, device=lines.device).view(1, H, 1)
        cols = torch.arange(W, device=lines.device).view(1, 1, W)
        boxes = ((rows >= r0.view(-1, 1, 1)) & (rows <= r1.view(-1, 1, 1)) &
                 (cols >= c0.view(-1, 1, 1)) & (cols <= c1.view(-1, 1, 1))).float()
        return boxes, owners

    def contrastive_loss(self, bev_s, label_bev, gts):
        """Eq. 7 of SMG with the label BEV as the G.T. feature: box-pooled student / teacher
        features, cosine similarity, symmetric InfoNCE over all the boxes of the batch.
        Only the elements of contrast_classes (if set) are pooled."""
        feats_s, feats_t = [], []
        for b, (labels, lines) in enumerate(zip(gts[0]['labels'], gts[0]['lines'])):
            if lines.dim() == 3:
                lines = lines[:, 0]
            contrast_classes = getattr(self, 'contrast_classes', None)
            if contrast_classes is not None:
                keep = sum(labels == c for c in contrast_classes).bool()
                labels, lines = labels[keep], lines[keep]
                if not len(labels):
                    continue
            boxes, _ = self.element_boxes(labels, lines, *bev_s.shape[-2:])
            area = boxes.sum((1, 2)).clamp(min=1.0)[:, None]
            feats_s.append(torch.einsum('nhw,chw->nc', boxes, bev_s[b]) / area)
            feats_t.append(torch.einsum('nhw,chw->nc', boxes, label_bev[b]) / area)
        o = F.normalize(torch.cat(feats_s), dim=-1)
        g = F.normalize(torch.cat(feats_t), dim=-1)
        logits = g @ o.t() / self.contrast_tau
        target = torch.arange(len(o), device=o.device)
        return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target))

    def forward_test(self, *args, **kwargs):
        return self.student.forward_test(*args, **kwargs)
