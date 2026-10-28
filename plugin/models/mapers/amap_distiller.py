"""
    AMap: distill the priors of a frozen (future-aware) temporal teacher into a
    current-frame-only student.
    - Multi-level BEV distillation: masked L2 on the basic and refined BEV features
    - Asymmetric query distillation: Hungarian matching between the student queries and
      the teacher (tracked + new) queries, then KL on the classification logits
"""
import os.path as osp

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet.utils import get_root_logger
from mmdet3d.models.builder import build_model

from .base_mapper import BaseMapper, MAPPERS


@MAPPERS.register_module()
class AMapDistiller(BaseMapper):

    def __init__(self,
                 student_cfg,
                 teacher_config,
                 teacher_ckpt,
                 student_img_norm=None,
                 loss_feat_basic_weight=1.0,
                 loss_feat_refined_weight=1.0,
                 feat_mask=True,
                 loss_logits_weight=1.0,
                 kd_temperature=1.0,
                 logits_reduction='sum',
                 match_cls_weight=5.0,
                 match_reg_weight=50.0,
                 teacher_cache=None,
                 **kwargs):
        """
        Args:
            student_cfg: model config of the current-frame-only MapTracker
            teacher_config: path of the teacher's config file, its `model` and `img_norm_cfg`
                are used
            teacher_ckpt: trained teacher weights, frozen during the distillation
            student_img_norm: img_norm_cfg of the student. The pipeline normalizes the images
                for the student, they are re-normalized for the teacher if it differs.
            feat_mask: only distill the BEV positions covered by the G.T. semantic mask
            logits_reduction: 'sum' over the matched student queries (as the paper's Eq. 3)
                or 'mean'
            match_cls_weight / match_reg_weight: Hungarian cost weights, same scale as the
                MapTracker assigner (FocalLossCost 5.0 / LinesL1Cost 50.0)
            teacher_cache: folder of the teacher outputs precomputed per sample token by
                tools/amap_cache_teacher.py. The teacher is then not built and the train data
                needs no clip.
        """
        super().__init__()
        teacher_file_cfg = Config.fromfile(teacher_config)
        self.student = build_model(student_cfg)
        assert not self.student.temporal, 'the AMap student only sees the current frame'
        self.teacher_cache = teacher_cache
        if teacher_cache is None:
            self.teacher = build_model(teacher_file_cfg.model)
            assert self.teacher.temporal
        else:
            self.teacher = None
        self.teacher_ckpt = teacher_ckpt

        teacher_img_norm = teacher_file_cfg.img_norm_cfg
        if student_img_norm is None or student_img_norm == teacher_img_norm:
            # same normalization, no conversion
            self.student_img_norm = self.teacher_img_norm = None
        else:
            self.student_img_norm = student_img_norm
            self.teacher_img_norm = teacher_img_norm

        self.loss_feat_basic_weight = loss_feat_basic_weight
        self.loss_feat_refined_weight = loss_feat_refined_weight
        self.feat_mask = feat_mask
        self.loss_logits_weight = loss_logits_weight
        self.kd_temperature = kd_temperature
        assert logits_reduction in ['sum', 'mean']
        self.logits_reduction = logits_reduction
        self.match_cls_weight = match_cls_weight
        self.match_reg_weight = match_reg_weight

        self._freeze_teacher()

    def init_weights(self, pretrained=None):
        self.student.init_weights()
        logger = get_root_logger()
        if self.teacher is None:
            logger.info(f'AMap teacher outputs read from {self.teacher_cache}')
            return
        load_checkpoint(self.teacher, self.teacher_ckpt, map_location='cpu', strict=True, logger=logger)
        logger.info(f'AMap teacher loaded from {self.teacher_ckpt}')
        self._freeze_teacher()

    def _freeze_teacher(self):
        if self.teacher is None:
            return
        # MapTracker.train() toggles requires_grad of the BEV modules, freeze_bev keeps them off
        self.teacher.freeze_bev = True
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

    def train(self, mode=True):
        super().train(mode)
        self._freeze_teacher()
        return self

    # the runner sets num_iter / num_epoch on the wrapped model, MapTracker reads them
    @property
    def num_iter(self):
        return self.student.num_iter

    @num_iter.setter
    def num_iter(self, value):
        self.student.num_iter = value
        if self.teacher is not None:
            self.teacher.num_iter = value

    def _renorm_for_teacher(self, img):
        """img: (..., 3, H, W) normalized with student_img_norm."""
        if self.student_img_norm is None:
            return img
        src, dst = self.student_img_norm, self.teacher_img_norm
        shape = [1] * (img.dim() - 3) + [3, 1, 1]
        # mean/std follow the channel order after the optional BGR->RGB conversion
        raw = img * img.new_tensor(src['std']).view(shape) + img.new_tensor(src['mean']).view(shape)
        if src['to_rgb'] != dst['to_rgb']:
            raw = raw.flip(-3)
        return (raw - img.new_tensor(dst['mean']).view(shape)) / img.new_tensor(dst['std']).view(shape)

    def forward_train(self, img, vectors, semantic_mask, points=None, img_metas=None, all_prev_data=None,
                      all_local2global_info=None, **kwargs):
        # 1. frozen teacher: sequential inference over the clip, current-frame outputs
        if self.teacher is None:
            teacher_outs = self._load_teacher_cache(img_metas, img.device)
        else:
            assert all_prev_data is not None, 'the teacher needs the temporal clip'
            teacher_prev_data = [dict(prev_data, img=self._renorm_for_teacher(prev_data['img']))
                                 for prev_data in all_prev_data]
            teacher_outs = self.teacher.forward_kd_teacher(self._renorm_for_teacher(img), img_metas,
                                                           teacher_prev_data)

        # 2. student: current frame only, with its original MapTracker losses
        loss, log_vars, num_sample, student_outs = self.student.forward_train(
            img, vectors, semantic_mask, points=points, img_metas=img_metas, return_kd_outputs=True)

        # 3. distillation losses
        if self.feat_mask:
            # same flip as the seg G.T. in MapTracker, so that the mask aligns with the BEV grid
            fg_mask = torch.flip(semantic_mask, [2,]).float().amax(1, keepdim=True)
            fg_mask = F.adaptive_max_pool2d(fg_mask, student_outs['bev_refined'].shape[-2:])
        else:
            fg_mask = None

        kd_losses = {}
        if self.loss_feat_basic_weight > 0:
            kd_losses['kd_feat_basic'] = self.loss_feat_basic_weight * self.masked_feat_loss(
                student_outs['bev_basic'], teacher_outs['bev_basic'], fg_mask)
        if self.loss_feat_refined_weight > 0:
            kd_losses['kd_feat_refined'] = self.loss_feat_refined_weight * self.masked_feat_loss(
                student_outs['bev_refined'], teacher_outs['bev_refined'], fg_mask)
        if self.loss_logits_weight > 0:
            loss_logits, num_teacher_queries = self.matched_logits_loss(student_outs, teacher_outs)
            kd_losses['kd_logits'] = self.loss_logits_weight * loss_logits
            log_vars['kd_num_teacher_queries'] = num_teacher_queries

        for name, kd_loss in kd_losses.items():
            loss = loss + kd_loss
            log_vars[name] = kd_loss.item()
        log_vars['total'] = loss.item()

        return loss, log_vars, num_sample

    def _load_teacher_cache(self, img_metas, device):
        """Same structure as MapTracker.forward_kd_teacher, the BEV features are stored in fp16."""
        cached = [torch.load(osp.join(self.teacher_cache, f"{meta['token']}.pt"), map_location='cpu')
                  for meta in img_metas]
        teacher_outs = {k: torch.stack([c[k] for c in cached]).to(device).float()
                        for k in ('bev_basic', 'bev_refined')}
        for k in ('lines', 'scores'):
            teacher_outs[k] = [c[k].to(device) for c in cached]
        return teacher_outs

    def masked_feat_loss(self, feat_student, feat_teacher, mask=None):
        """Eq. 2: per-position L2 distance averaged over the masked positions.
        feats: (bs, C, H, W), mask: (bs, 1, H, W) in {0, 1}"""
        feat_teacher = feat_teacher.detach()
        # eps keeps the gradient of the sqrt finite at zero distance
        dist = ((feat_student - feat_teacher) ** 2).sum(1, keepdim=True).add(1e-12).sqrt()
        if mask is None:
            return dist.mean()
        return (dist * mask).sum() / mask.sum().clamp(min=1.0)

    @torch.no_grad()
    def match_queries(self, lines_s, scores_s, lines_t, scores_t):
        """One-to-one Hungarian matching from the student queries to the teacher queries,
        the teacher predictions serve as the targets of the matching cost."""
        cls_cost = torch.cdist(scores_s.sigmoid(), scores_t.sigmoid(), p=1)
        num_pts = lines_s.shape[-1] // 2
        lines_t_reversed = lines_t.view(len(lines_t), num_pts, 2).flip(1).reshape(len(lines_t), -1)
        # direction-invariant line distance, normalized as the LinesL1Cost
        reg_cost = torch.min(torch.cdist(lines_s, lines_t, p=1),
                             torch.cdist(lines_s, lines_t_reversed, p=1)) / num_pts
        cost = self.match_cls_weight * cls_cost + self.match_reg_weight * reg_cost
        student_ids, teacher_ids = linear_sum_assignment(cost.cpu().numpy())
        return (torch.as_tensor(student_ids, device=lines_s.device),
                torch.as_tensor(teacher_ids, device=lines_s.device))

    def matched_logits_loss(self, student_outs, teacher_outs):
        """Eq. 3: KL between the logits of the matched queries. The classifier is a sigmoid
        (focal loss) one, so the KL is computed per class between Bernoulli distributions."""
        T = self.kd_temperature
        losses = []
        num_teacher_queries = 0
        for lines_s, scores_s, lines_t, scores_t in zip(student_outs['lines'], student_outs['scores'],
                                                        teacher_outs['lines'], teacher_outs['scores']):
            assert len(scores_t) >= len(scores_s), 'every student query needs a teacher query'
            num_teacher_queries += len(scores_t)
            student_ids, teacher_ids = self.match_queries(lines_s.detach(), scores_s.detach(),
                                                          lines_t, scores_t)
            logits_s = scores_s[student_ids] / T
            logits_t = scores_t[teacher_ids].detach() / T
            p_t = logits_t.sigmoid()
            kl = p_t * (F.logsigmoid(logits_t) - F.logsigmoid(logits_s)) + \
                (1 - p_t) * (F.logsigmoid(-logits_t) - F.logsigmoid(-logits_s))
            kl = kl.sum(-1) # sum over classes
            kl = kl.sum() if self.logits_reduction == 'sum' else kl.mean()
            losses.append(kl * T * T)
        return torch.stack(losses).mean(), num_teacher_queries / len(losses)

    def forward_test(self, *args, **kwargs):
        return self.student.forward_test(*args, **kwargs)
