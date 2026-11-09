"""
    FTKD with the SMG contrastive BEV alignment: the masked BEV reconstruction (MSE towards
    the future teacher BEV) is replaced by the symmetric InfoNCE between each G.T. element's
    box-pooled student BEV and box-pooled future teacher BEV (LabelKDDistiller 'contrastive',
    the label BEV swapped for the cached teacher BEV). The query reconstruction and the logit
    distillation are FTKD's.
    Variants of FTKDContrastDistiller:
        contrast_pool='line': pool each element over its rasterized polyline (dilated by
            one cell) instead of its bounding box(es), less background inside the pool
        keep_bev_mse: keep FTKD's masked BEV MSE reconstruction next to the SMG loss
        loss_fg_cos_weight: add a dense per-cell cosine alignment towards the teacher BEV
            on the polyline cells of all the elements (keeps the spatial detail that the
            pooling averages away)
    FMKDContrastDistiller: FM-KD (e.g. the E5 bridge) with the SMG contrastive loss added
    to its flow matching loss, on the BEV the student's head reads.
"""
import torch
import torch.nn.functional as F

from .base_mapper import MAPPERS
from .fmkd_distiller import FMKDDistiller
from .ftkd_distiller import FTKDDistiller
from .labelkd_distiller import LabelKDDistiller


@MAPPERS.register_module()
class FTKDContrastDistiller(FTKDDistiller):

    contrastive_loss = LabelKDDistiller.contrastive_loss

    def __init__(self, contrast_tau=0.1, boundary_label=2, contrast_classes=None, contrast_mask=False,
                 contrast_pool='box', keep_bev_mse=False, loss_fg_cos_weight=0.0, **kwargs):
        """
        Args:
            contrast_tau: InfoNCE temperature
            boundary_label: class id of the road boundaries, pooled over two boxes
            contrast_classes: class ids whose elements are aligned, None for all
            contrast_mask: align the generator's output on the masked student BEV (FTKD's
                random mask) instead of the student BEV itself
            contrast_pool: 'box' (SMG) or 'line' (rasterized polyline cells)
            keep_bev_mse: add FTKD's masked BEV MSE reconstruction to the SMG loss
            loss_fg_cos_weight: weight of the dense cosine alignment on the polyline cells
            kwargs: FTKDDistiller arguments
        """
        super().__init__(**kwargs)
        assert contrast_pool in ('box', 'line')
        assert not (contrast_mask and keep_bev_mse), 'both would use the BEV generator'
        self.contrast_tau = contrast_tau
        self.boundary_label = boundary_label
        self.contrast_classes = contrast_classes
        self.contrast_mask = contrast_mask
        self.contrast_pool = contrast_pool
        self.keep_bev_mse = keep_bev_mse
        self.loss_fg_cos_weight = loss_fg_cos_weight
        if not (contrast_mask or keep_bev_mse):
            del self.bev_generator
        self._gts = None

    def element_boxes(self, labels, lines, H, W):
        if self.contrast_pool == 'box':
            return LabelKDDistiller.element_boxes(self, labels, lines, H, W)
        return self.element_line_masks(lines, H, W), list(range(len(lines)))

    @staticmethod
    def element_line_masks(lines, H, W, substeps=32):
        """(num_elements, H, W) masks of the polyline cells (densified, dilated by one cell)."""
        pts = lines.view(len(lines), -1, 2)
        w = torch.linspace(0, 1, substeps + 1, device=pts.device)[:-1].view(1, 1, -1, 1)
        dense = pts[:, :-1, None] * (1 - w) + pts[:, 1:, None] * w
        dense = torch.cat([dense.flatten(1, 2), pts[:, -1:]], dim=1).clamp(0, 1)
        cols = (dense[..., 0] * (W - 1)).round().long()
        rows = ((1 - dense[..., 1]) * (H - 1)).round().long()
        masks = pts.new_zeros(len(pts), H, W)
        idx = torch.arange(len(pts), device=pts.device)[:, None].expand_as(rows)
        masks[idx, rows, cols] = 1.0
        return F.max_pool2d(masks[:, None], 3, stride=1, padding=1)[:, 0]

    def fg_cos_loss(self, feat_student, feat_teacher):
        """1 - cos(student, teacher) per BEV cell, averaged over the polyline cells."""
        H, W = feat_student.shape[-2:]
        fg = []
        for lines in self._gts[0]['lines']:
            if lines.dim() == 3:
                lines = lines[:, 0]
            fg.append(self.element_line_masks(lines, H, W).amax(0) if len(lines) else feat_student.new_zeros(H, W))
        fg = torch.stack(fg)
        cos = F.cosine_similarity(feat_student, feat_teacher, dim=1)
        return ((1 - cos) * fg).sum() / fg.sum().clamp(min=1.0)

    def forward_train(self, img, vectors, semantic_mask, **kwargs):
        # the G.T. elements give the pooling boxes
        self._gts = self.student.batch_data(vectors, None, None, img.device)[0]
        try:
            return super().forward_train(img, vectors, semantic_mask, **kwargs)
        finally:
            self._gts = None

    def bev_recon_loss(self, feat_student, feat_teacher):
        if self.contrast_mask:
            bs, _, h, w = feat_student.shape
            keep = (torch.rand(bs, 1, h, w, device=feat_student.device) >= self.bev_mask_ratio).to(feat_student.dtype)
            feat_student = self.bev_generator(feat_student * keep)
        loss = self.contrastive_loss(feat_student, feat_teacher.detach(), self._gts)
        if self.keep_bev_mse:
            loss = loss + super().bev_recon_loss(feat_student, feat_teacher)
        if self.loss_fg_cos_weight > 0:
            loss = loss + self.loss_fg_cos_weight * self.fg_cos_loss(feat_student, feat_teacher.detach())
        return loss


@MAPPERS.register_module()
class FMKDContrastDistiller(FMKDDistiller):

    element_boxes = LabelKDDistiller.element_boxes
    contrastive_loss = LabelKDDistiller.contrastive_loss

    def __init__(self, contrast_tau=0.1, boundary_label=2, contrast_classes=None,
                 loss_contrast_weight=1.0, loss_flow_weight=1.0, **kwargs):
        """
        Args:
            contrast_tau / boundary_label / contrast_classes: see FTKDContrastDistiller
            loss_contrast_weight: weight of the contrastive loss, added to the flow loss
            loss_flow_weight: weight of the flow matching loss, 0 trains the refine module
                with the task and contrastive losses only (no flow matching)
            kwargs: FMKDDistiller arguments
        """
        super().__init__(**kwargs)
        self.contrast_tau = contrast_tau
        self.boundary_label = boundary_label
        self.contrast_classes = contrast_classes
        self.loss_contrast_weight = loss_contrast_weight
        self.loss_flow_weight = loss_flow_weight
        self._gts = None

    def forward_train(self, img, vectors, semantic_mask, **kwargs):
        self._gts = self.student.batch_data(vectors, None, None, img.device)[0]
        try:
            return super().forward_train(img, vectors, semantic_mask, **kwargs)
        finally:
            self._gts = None

    def bev_recon_loss(self, feat_student, feat_teacher):
        # feat_student is the BEV the head reads (refined by the flow in the refine modes)
        loss = self.loss_contrast_weight * self.contrastive_loss(feat_student, feat_teacher.detach(), self._gts)
        if self.loss_flow_weight > 0:
            loss = loss + self.loss_flow_weight * super().bev_recon_loss(feat_student, feat_teacher)
        return loss
