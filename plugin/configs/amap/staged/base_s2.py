_base_ = ['../fast/amap_student_nusc_oldsplit_r18_current_frame_bs64.py']

# Staged baseline, stage 2: vector-module warmup, BEV frozen (backbone + seg head; the neck stays
# trainable, same as the paper's freeze_bev). Losses: seg + cls + reg. 2 ep = 436 iters.
num_iters_per_epoch = 27968 // (2 * 64)
total_iters = 2 * num_iters_per_epoch
model = dict(freeze_bev=True)
runner = dict(max_iters=total_iters)
checkpoint_config = dict(interval=total_iters)
evaluation = dict(interval=10 ** 9)
optimizer = dict(lr=5e-4)
lr_config = dict(warmup_iters=100)
find_unused_parameters = True        # frozen parameters (paper stage 2 also sets True)
load_from = 'work_dirs/stg_base_s1/latest.pth'
