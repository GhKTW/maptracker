# N3 (FTKD + SMG) pooling each element over its rasterized polyline cells instead of its bounding box(es)
_base_ = ['./ftkd_smg_nusc_oldsplit_r18_from_future_teacher_cached_bs64.py']

model = dict(contrast_pool='line')
