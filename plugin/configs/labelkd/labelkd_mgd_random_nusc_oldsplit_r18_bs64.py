# Label distillation (L2): L1 with the random mask of MGD / FTKD (0.5)
_base_ = ['./labelkd_mgd_ahead_nusc_oldsplit_r18_bs64.py']

model = dict(bev_mask_mode='random', bev_mask_ratio=0.5)
