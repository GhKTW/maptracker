# Label distillation (N1): L3 + L4, the LabelDistill foreground MSE (adapter) and the SMG
# contrastive alignment together, weight 1 each, no mask
_base_ = ['./labelkd_mgd_ahead_nusc_oldsplit_r18_bs64.py']

model = dict(loss_type=['fg_mse', 'contrastive'], bev_mask_mode='none', contrast_tau=0.1)
