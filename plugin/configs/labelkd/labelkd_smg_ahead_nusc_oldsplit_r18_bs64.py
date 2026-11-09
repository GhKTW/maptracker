# Label distillation (N2): L4 with the ahead mask (0.75 ahead / 0.25 rear), a conv generator
# recovers the masked student BEV and its box-pooled features are aligned (SMG InfoNCE)
_base_ = ['./labelkd_mgd_ahead_nusc_oldsplit_r18_bs64.py']

model = dict(loss_type='contrastive', bev_mask_mode='ahead', bev_mask_ratio=0.5,
             ahead_mask_ratio=0.75, contrast_tau=0.1)
