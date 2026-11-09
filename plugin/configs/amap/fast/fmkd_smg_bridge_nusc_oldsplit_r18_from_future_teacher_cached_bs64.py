# E5 (flow bridge student BEV -> teacher BEV in the inference path) + the SMG contrastive
# loss of N3 on the refined BEV the head reads, the two runs with the best A-divider
_base_ = ['./fmkd_nusc_oldsplit_r18_from_future_teacher_cached_bs64_bridge.py']

model = dict(type='FMKDContrastDistiller', contrast_tau=0.1, boundary_label=2, loss_contrast_weight=1.0)
optimizer = dict(lr=7e-4)
