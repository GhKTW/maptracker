# FM-KD (E4): E1 with the flow net kept at inference, noise -> teacher-like BEV conditioned
# on the student BEV, the decoder reads the refined BEV (1 Euler step in training and test)
_base_ = ['./fmkd_nusc_oldsplit_r18_from_future_teacher_cached_bs64.py']

model = dict(refine_mode='noise', refine_steps=1)
