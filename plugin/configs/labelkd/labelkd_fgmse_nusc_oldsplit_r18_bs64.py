# Label distillation (L3): the LabelDistill loss, 1x1 adapter + MSE on the G.T. foreground
# cells, no mask
_base_ = ['./labelkd_mgd_ahead_nusc_oldsplit_r18_bs64.py']

model = dict(loss_type='fg_mse', bev_mask_mode='none')
