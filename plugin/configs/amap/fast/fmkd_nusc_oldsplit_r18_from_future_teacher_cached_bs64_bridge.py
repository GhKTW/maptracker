# FM-KD (E5): E4 with the student BEV as the flow source (current -> future bridge)
# instead of noise
_base_ = ['./fmkd_nusc_oldsplit_r18_from_future_teacher_cached_bs64.py']

model = dict(refine_mode='bridge', refine_steps=1)
