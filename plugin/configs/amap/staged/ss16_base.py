_base_ = ['../fast/amap_student_nusc_oldsplit_r18_current_frame.py']
# 11-16: the original single-stage recipe (no staging: joint from the start, 48 ep, total batch 16, lr 2.5e-4,
# warmup 500, cosine) that reached 67.4 A / 67.0 R on 8 GPU x 2 -> here 2 GPU x 8 (same total batch, same iters).
# Question: after rcp32 (staged, batch 32) = 65.3 / 66.0, is the remaining ~2 pt gap the staging itself?
data = dict(samples_per_gpu=8)
checkpoint_config = dict(interval=83904 // 12)   # resume points after container restarts
