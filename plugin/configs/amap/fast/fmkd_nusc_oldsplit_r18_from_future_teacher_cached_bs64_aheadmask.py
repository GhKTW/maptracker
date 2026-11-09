# FM-KD (E3): E1 with the mask moved to the ahead half (A-mAP region) at the same mean
# ratio 0.5: ahead positions dropped with 0.75, rear ones with 0.25
_base_ = ['./fmkd_nusc_oldsplit_r18_from_future_teacher_cached_bs64.py']

model = dict(bev_mask_mode='ahead', bev_mask_ratio=0.5, ahead_mask_ratio=0.75)
