_base_ = ['./base_s3.py']
# 11-15 recipe-gap test, stage 3 (joint, all losses), total batch 32, 40 ep = 34,960 iters, lr 2.5e-4
num_iters_per_epoch = 27968 // (2 * 16)
total_iters = 40 * num_iters_per_epoch
data = dict(samples_per_gpu=16)
runner = dict(max_iters=total_iters)
checkpoint_config = dict(interval=num_iters_per_epoch * 4)
evaluation = dict(interval=total_iters)
optimizer = dict(lr=2.5e-4)
lr_config = dict(warmup_iters=1000)
load_from = 'work_dirs/stg_rcp32_base_s2/latest.pth'
