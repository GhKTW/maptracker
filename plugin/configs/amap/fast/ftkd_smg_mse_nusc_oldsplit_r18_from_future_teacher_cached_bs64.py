# N3 (FTKD + SMG) keeping FTKD's masked BEV MSE reconstruction next to the SMG loss
_base_ = ['./ftkd_smg_nusc_oldsplit_r18_from_future_teacher_cached_bs64.py']

model = dict(keep_bev_mse=True)
