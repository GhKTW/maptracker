_base_ = ['../fast/amap_student_nusc_oldsplit_r18_current_frame_bs64.py']

# Staged baseline, stage 3: joint finetuning, all losses. 40 ep = 8,720 iters.
num_iters_per_epoch = 27968 // (2 * 64)
total_iters = 40 * num_iters_per_epoch
runner = dict(max_iters=total_iters)
checkpoint_config = dict(interval=num_iters_per_epoch * 4)  # resume after container restarts
evaluation = dict(interval=total_iters)
optimizer = dict(lr=7e-4)            # overridden by the sweep result
lr_config = dict(warmup_iters=500)
load_from = 'work_dirs/stg_base_s2/latest.pth'
