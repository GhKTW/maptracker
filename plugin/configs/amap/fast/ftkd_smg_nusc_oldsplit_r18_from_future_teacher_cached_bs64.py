# FTKD with the SMG contrastive BEV alignment (N3): the masked BEV reconstruction is replaced
# by the box-pooled InfoNCE towards the cached future teacher BEV, the query losses are FTKD's
_base_ = ['./ftkd_nusc_oldsplit_r18_from_future_teacher_cached_bs64.py']

model = dict(type='FTKDContrastDistiller', contrast_tau=0.1, boundary_label=2, contrast_mask=False)

# the lr of every bs64 run so far (the FTKD / FM-KD queues passed it on the command line)
optimizer = dict(lr=7e-4)
