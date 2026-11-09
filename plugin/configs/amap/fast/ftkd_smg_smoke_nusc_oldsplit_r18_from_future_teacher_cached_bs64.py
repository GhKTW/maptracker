# smoke test of the new FTKDContrastDistiller options (not a run)
_base_ = ['./ftkd_smg_nusc_oldsplit_r18_from_future_teacher_cached_bs64.py']

model = dict(contrast_pool='line', keep_bev_mse=True, loss_fg_cos_weight=1.0)
