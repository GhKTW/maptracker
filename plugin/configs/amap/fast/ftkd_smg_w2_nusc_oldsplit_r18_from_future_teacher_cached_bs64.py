# N3 (FTKD + SMG) with the SMG loss weight 2 (loss values at the end of N3: SMG 3.6, query recon 3.5, task 8.3)
_base_ = ['./ftkd_smg_nusc_oldsplit_r18_from_future_teacher_cached_bs64.py']

model = dict(loss_bev_recon_weight=2.0)
