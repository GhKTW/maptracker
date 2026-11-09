# N3 (FTKD + SMG) with the query reconstruction weight 16 -> 4 (as large as the SMG loss at 16, FTKD alone showed no gain)
_base_ = ['./ftkd_smg_nusc_oldsplit_r18_from_future_teacher_cached_bs64.py']

model = dict(loss_query_recon_weight=4.0)
