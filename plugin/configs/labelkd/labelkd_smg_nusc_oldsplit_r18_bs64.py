# Label distillation (L4): the SMG contrastive alignment (box-pooled features, two boxes per
# road boundary, symmetric InfoNCE, weight 1 as in SMG), the label BEV as the G.T. feature
_base_ = ['./labelkd_mgd_ahead_nusc_oldsplit_r18_bs64.py']

model = dict(loss_type='contrastive', bev_mask_mode='none', contrast_tau=0.1)
