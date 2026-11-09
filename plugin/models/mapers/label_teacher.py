"""
    Label teacher for map distillation, LabelDistill (Kim et al., ECCV 2024) adapted to
    vectorized maps: a label encoder g turns the G.T. map elements into a BEV feature map,
    trained as an approximate inverse of a frozen map decoder h (autoencoder whose decoder
    is fixed), i.e. min_g L_map(h(g(y)), y).
        g(y) = f(q(Phi_cls(c) + Phi_geom(p)))
        - Phi_cls: class embedding, Phi_geom: MLP on the 20 normalized polyline points
        - q: paints each instance embedding on the BEV cells its polyline crosses
          (densified polyline, optionally dilated), overlapping instances are summed
        - f: conv refinement
    h is the BEV-to-map part (seg head + vector head) of a trained current-frame MapTracker,
    so the label BEV lives in the BEV space a MapTracker head reads.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import load_checkpoint
from mmdet.utils import get_root_logger
from mmdet3d.models.builder import build_model

from .base_mapper import BaseMapper, MAPPERS


class LabelEncoder(nn.Module):

    def __init__(self, bev_h, bev_w, channels=256, num_classes=3, num_points=20,
                 samples_per_seg=8, dilation=1, num_convs=3):
        super().__init__()
        self.bev_h, self.bev_w = bev_h, bev_w
        self.num_points = num_points
        self.samples_per_seg = samples_per_seg
        self.dilation = dilation
        self.cls_embed = nn.Embedding(num_classes, channels)
        self.geom_embed = nn.Sequential(
            nn.Linear(2 * num_points, channels), nn.ReLU(inplace=True), nn.Linear(channels, channels))
        layers = []
        for _ in range(num_convs):
            layers += [nn.Conv2d(channels, channels, 3, padding=1), nn.GroupNorm(32, channels), nn.ReLU(inplace=True)]
        layers.append(nn.Conv2d(channels, channels, 3, padding=1))
        self.refine = nn.Sequential(*layers)

    def instance_masks(self, lines):
        """lines: (N, 2*num_points) normalized to [0, 1] -> (N, H, W) 0/1 masks.
        BEV layout follows MapTracker.plane: columns go along x from -roi/2 to roi/2, rows
        along y from +roi/2 to -roi/2."""
        H, W = self.bev_h, self.bev_w
        pts = lines.view(len(lines), -1, 2)
        t = torch.linspace(0, 1, self.samples_per_seg + 1, device=lines.device)[:-1].view(1, 1, -1, 1)
        dense = pts[:, :-1, None] * (1 - t) + pts[:, 1:, None] * t
        dense = torch.cat([dense.flatten(1, 2), pts[:, -1:]], dim=1)
        cols = (dense[..., 0] * (W - 1)).round().long().clamp(0, W - 1)
        rows = ((1 - dense[..., 1]) * (H - 1)).round().long().clamp(0, H - 1)
        masks = lines.new_zeros(len(lines), H * W)
        masks.scatter_(1, rows * W + cols, 1.0)
        masks = masks.view(-1, 1, H, W)
        if self.dilation > 0:
            k = 2 * self.dilation + 1
            masks = F.max_pool2d(masks, k, stride=1, padding=self.dilation)
        return masks[:, 0]

    def forward(self, labels_list, lines_list):
        """Per sample labels (N,), lines (N, 2*num_points) or (N, num_permute, 2*num_points).
        Returns the label BEV (bs, C, H, W) and the G.T. foreground mask (bs, H, W)."""
        bevs, fgs = [], []
        for labels, lines in zip(labels_list, lines_list):
            if lines.dim() == 3:
                lines = lines[:, 0] # the original point order
            masks = self.instance_masks(lines)
            embeds = self.cls_embed(labels) + self.geom_embed(lines * 2 - 1)
            bevs.append(torch.einsum('nhw,nc->chw', masks, embeds))
            fgs.append(masks.amax(0) if len(masks) else masks.new_zeros(self.bev_h, self.bev_w))
        return self.refine(torch.stack(bevs)), torch.stack(fgs)


@MAPPERS.register_module()
class LabelMapTeacher(BaseMapper):

    def __init__(self,
                 mapper_cfg,
                 mapper_ckpt,
                 encoder_cfg=dict(),
                 **kwargs):
        """
        Args:
            mapper_cfg: config of the current-frame MapTracker whose seg head and vector
                head are the frozen decoder h
            mapper_ckpt: its checkpoint
            encoder_cfg: LabelEncoder arguments
        """
        super().__init__()
        self.mapper = build_model(mapper_cfg)
        assert not self.mapper.temporal, 'the decoder is a current-frame MapTracker'
        self.mapper_ckpt = mapper_ckpt
        self.encoder = LabelEncoder(self.mapper.bev_h, self.mapper.bev_w, **encoder_cfg)
        self._freeze_mapper()

    def init_weights(self, pretrained=None):
        load_checkpoint(self.mapper, self.mapper_ckpt, map_location='cpu', strict=True,
                        logger=get_root_logger())
        self._freeze_mapper()

    def _freeze_mapper(self):
        self.mapper.eval()
        for p in self.mapper.parameters():
            p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        # MapTracker.train() / eval() unfreeze its backbone and seg head, freeze again
        self._freeze_mapper()
        return self

    @property
    def num_iter(self):
        return self.mapper.num_iter

    @num_iter.setter
    def num_iter(self, value):
        self.mapper.num_iter = value

    def encode(self, vectors, device):
        """G.T. vectors -> (label BEV, G.T. foreground mask, gts in the MapTracker format)"""
        gts = self.mapper.batch_data(vectors, None, None, device)[0]
        bev, fg = self.encoder(gts[0]['labels'], gts[0]['lines'])
        return bev, fg, gts

    def forward_train(self, vectors, semantic_mask, img_metas=None, img=None, **kwargs):
        bev, _, gts = self.encode(vectors, semantic_mask.device)
        gt_semantic = torch.flip(semantic_mask, [2,])
        _, _, seg_loss, seg_dice_loss = self.mapper.seg_decoder(bev, gt_semantic, None, return_loss=True)
        _, loss_dict, _, _, _ = self.mapper.head(
            bev_features=bev, img_metas=img_metas, gts=gts, track_query_info=None,
            memory_bank=None, return_loss=True)
        loss_dict['seg'] = seg_loss
        loss_dict['seg_dice'] = seg_dice_loss
        loss = sum(loss_dict.values())
        log_vars = {k: v.item() for k, v in loss_dict.items()}
        log_vars['total'] = loss.item()
        return loss, log_vars, len(vectors)

    @torch.no_grad()
    def forward_test(self, img, img_metas=None, seq_info=None, vectors=None, **kwargs):
        """Upper bound: the MapTracker inference with its BEV replaced by the label BEV."""
        bev = self.encode(vectors, img.device)[0]
        self.mapper.bev_refine_fn = lambda _: bev
        try:
            return self.mapper.forward_test(img, img_metas=img_metas, seq_info=seq_info, **kwargs)
        finally:
            self.mapper.bev_refine_fn = None
