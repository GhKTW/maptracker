_base_ = ['./base_s2.py']
# 11-15 recipe-gap test, stage 2 (vector warmup, BEV frozen), total batch 32, lr 2.5e-4
num_iters_per_epoch = 27968 // (2 * 16)
total_iters = 2 * num_iters_per_epoch
data = dict(samples_per_gpu=16)
runner = dict(max_iters=total_iters)
checkpoint_config = dict(interval=total_iters)
optimizer = dict(lr=2.5e-4)
lr_config = dict(warmup_iters=200)
load_from = 'work_dirs/stg_rcp32_base_s1/latest.pth'
