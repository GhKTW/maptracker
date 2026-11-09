# N3 (FTKD + SMG) plus a dense per-cell cosine alignment to the teacher BEV on the polyline cells
_base_ = ['./ftkd_smg_nusc_oldsplit_r18_from_future_teacher_cached_bs64.py']

model = dict(loss_fg_cos_weight=1.0)
