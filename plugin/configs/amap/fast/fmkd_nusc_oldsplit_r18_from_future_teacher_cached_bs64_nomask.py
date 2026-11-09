# FM-KD (E2): E1 without the BEV mask, the flow noise is the only corruption
_base_ = ['./fmkd_nusc_oldsplit_r18_from_future_teacher_cached_bs64.py']

model = dict(bev_mask_ratio=0.0)
