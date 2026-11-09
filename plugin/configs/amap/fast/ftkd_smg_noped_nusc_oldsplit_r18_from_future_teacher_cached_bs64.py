# N3 with the SMG contrastive loss on the dividers and the road boundaries only (no ped
# crossing boxes): N3 raised A-divider (+1.3 over 3 seeds) but lowered A-ped crossing
_base_ = ['./ftkd_smg_nusc_oldsplit_r18_from_future_teacher_cached_bs64.py']

model = dict(contrast_classes=[1, 2])
