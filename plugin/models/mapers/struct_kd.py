"""
    11-14: structure-level KD for the current-frame student (no feature MSE, no branch, no SMG).
    Both run on the cached future teacher (teacher_cache/branch_s3_bs8_f0base) and the plain student.
    B. relation KD (BEV level)
        f = avgpool_k(BEV) -> (bs, C, H/k, W/k), L2-normalized per cell, N = H/k * W/k cells
        A = softmax_row(f^T f / tau)  (N x N cell-to-cell affinity)
        L_rel = mean_rows KL(A_T || A_S) * tau^2
        scale-free: only "which cells look alike" is transferred, not the teacher's feature values.
    C. query KD (decoder level), on the Hungarian-matched (student, teacher) query pairs
        L_logit = Bernoulli KL of the class logits (existing AMap term, temperature T)
        L_emb   = sum_pairs p_T * (1 - cos(P(h_S), h_T)) / sum p_T     P: training-only 512x512 linear, identity init
        L_line  = sum_pairs p_T * min(|l_S - l_T|_1, |l_S - rev(l_T)|_1) / n_pts / sum p_T   (normalized coords)
        p_T = max class prob of the teacher query: background teacher queries do not pull lines / embeddings.
    D. (11-14 later) dense seg-probability KD (head space)
        z_T = SegHead_T(F_T) with the teacher's own frozen seg head on the cached teacher BEV (200x100 canvas)
        L_seg = mean_pixels,classes BernoulliKL(sigmoid(z_T/T) || sigmoid(z_S/T)) * T^2
        z_S = the student's seg logits (its normal seg head); no feature-space target at all.
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .amap_distiller import AMapDistiller
from .base_mapper import MAPPERS


@MAPPERS.register_module()
class StructKDDistiller(AMapDistiller):

    def __init__(self,
                 loss_rel_weight=0.0,
                 rel_pool=2,
                 rel_tau=0.1,
                 loss_emb_weight=0.0,
                 loss_line_weight=0.0,
                 loss_segkd_weight=0.0,
                 segkd_temperature=1.0,
                 teacher_ckpt_for_seg=None,
                 **kwargs):
        kwargs.setdefault('loss_feat_basic_weight', 0.0)
        kwargs.setdefault('loss_feat_refined_weight', 0.0)
        kwargs.setdefault('loss_logits_weight', 0.0)
        super().__init__(**kwargs)
        assert self.teacher_cache is not None
        assert self.loss_feat_basic_weight == 0 and self.loss_feat_refined_weight == 0, 'no feature MSE here'
        self.loss_rel_weight = loss_rel_weight
        self.rel_pool = rel_pool
        self.rel_tau = rel_tau
        self.loss_emb_weight = loss_emb_weight
        self.loss_line_weight = loss_line_weight
        self.emb_proj = None
        if loss_emb_weight > 0:
            self.emb_proj = nn.Linear(512, 512)
            with torch.no_grad():
                self.emb_proj.weight.copy_(torch.eye(512))
                self.emb_proj.bias.zero_()
        self.loss_segkd_weight = loss_segkd_weight
        self.segkd_T = segkd_temperature
        self.teacher_seg = None
        if loss_segkd_weight > 0:
            assert teacher_ckpt_for_seg is not None
            self.teacher_seg = copy.deepcopy(self.student.seg_decoder)
            sd = torch.load(teacher_ckpt_for_seg, map_location='cpu')['state_dict']
            sd = {k[len('seg_decoder.'):]: v for k, v in sd.items() if k.startswith('seg_decoder.')}
            self.teacher_seg.load_state_dict(sd, strict=True)
            for p in self.teacher_seg.parameters():
                p.requires_grad_(False)
            self.teacher_seg.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.teacher_seg is not None:
            self.teacher_seg.eval()
        return self

    def segkd_loss(self, seg_s, bev_t):
        T = self.segkd_T
        with torch.no_grad():
            zt, _ = self.teacher_seg(bev_t, return_loss=False)
            pt = (zt / T).sigmoid()
        zs = seg_s / T
        # Bernoulli KL(p_T || p_S) = BCE(z_S, p_T) - H(p_T)
        bce = F.binary_cross_entropy_with_logits(zs, pt, reduction='none')
        ent = F.binary_cross_entropy(pt, pt, reduction='none')
        return (bce - ent).mean() * T * T

    def relation_loss(self, bev_s, bev_t):
        k, tau = self.rel_pool, self.rel_tau
        fs = F.normalize(F.avg_pool2d(bev_s, k).flatten(2), dim=1)            # (bs, C, N)
        ft = F.normalize(F.avg_pool2d(bev_t.detach(), k).flatten(2), dim=1)
        ls = torch.bmm(fs.transpose(1, 2), fs) / tau                          # (bs, N, N)
        lt = torch.bmm(ft.transpose(1, 2), ft) / tau
        pt = lt.softmax(-1)
        kl = (pt * (lt.log_softmax(-1) - ls.log_softmax(-1))).sum(-1)         # (bs, N)
        return kl.mean() * tau * tau

    def query_losses(self, s, t):
        n_pts = s['lines'][0].shape[-1] // 2
        emb, line = [], []
        for b, (lines_s, scores_s, lines_t, scores_t) in enumerate(
                zip(s['lines'], s['scores'], t['lines'], t['scores'])):
            sid, tid = self.match_queries(lines_s.detach(), scores_s.detach(), lines_t, scores_t)
            w = scores_t[tid].sigmoid().max(-1)[0].detach()                   # teacher confidence
            wn = w / w.sum().clamp(min=1e-6)
            if self.loss_emb_weight > 0:
                hs = self.emb_proj(s['hs_embeds'][b][sid])
                ht = t['hs_embeds'][b][tid].detach()
                emb.append((wn * (1 - F.cosine_similarity(hs, ht, dim=-1))).sum())
            if self.loss_line_weight > 0:
                ls = lines_s[sid]
                lt = lines_t[tid].detach()
                lt_rev = lt.view(len(lt), n_pts, 2).flip(1).reshape(len(lt), -1)
                d = torch.minimum((ls - lt).abs().sum(-1), (ls - lt_rev).abs().sum(-1)) / n_pts
                line.append((wn * d).sum())
        out = {}
        if emb:
            out['skd_emb'] = self.loss_emb_weight * torch.stack(emb).mean()
        if line:
            out['skd_line'] = self.loss_line_weight * torch.stack(line).mean()
        return out

    def forward_train(self, img, vectors, semantic_mask, points=None, img_metas=None, all_prev_data=None,
                      all_local2global_info=None, **kwargs):
        t = self._load_teacher_cache(img_metas, img.device)
        loss, log_vars, num_sample, s = self.student.forward_train(
            img, vectors, semantic_mask, points=points, img_metas=img_metas, return_kd_outputs=True)
        kd = {}
        if self.loss_rel_weight > 0:
            kd['skd_rel'] = self.loss_rel_weight * self.relation_loss(s['bev_refined'], t['bev_refined'])
        if self.loss_segkd_weight > 0:
            kd['skd_seg'] = self.loss_segkd_weight * self.segkd_loss(s['seg_preds'], t['bev_refined'])
        if s['lines'] is not None:   # stage 1 skips the vector head
            if self.loss_logits_weight > 0:
                l, nq = self.matched_logits_loss(s, t)
                kd['skd_logits'] = self.loss_logits_weight * l
            if self.loss_emb_weight > 0 or self.loss_line_weight > 0:
                kd.update(self.query_losses(s, t))
        for k, v in kd.items():
            loss = loss + v
            log_vars[k] = v.item()
        log_vars['total'] = loss.item()
        return loss, log_vars, num_sample
