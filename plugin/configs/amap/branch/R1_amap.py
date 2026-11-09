_base_ = ['./../fast/amap_kd_nusc_oldsplit_r18_from_future_teacher_cached_bs64.py']

# AMap reproduction (GT-mask BEV L2 + Hungarian logit KL), same recipe
optimizer = dict(lr=1.4e-3)
lr_config = dict(warmup_iters=500)
