_base_ = ['./base_s1.py']
# 11-15 recipe-gap test: staged baseline (with block, no KD) at total batch 32 (2 GPU x 16) instead of 128.
# Same epochs (S1 6 / S2 2 / S3 40) -> 4x the steps; lr halved (sqrt rule): S1 5e-4, S2/S3 2.5e-4 (= the bs16
# single-stage recipe's lr that reached 67.4 A-mAP). Reference: staged bs128 base 61.6/64.1/64.3.
num_iters_per_epoch = 27968 // (2 * 16)
total_iters = 6 * num_iters_per_epoch
data = dict(samples_per_gpu=16)
runner = dict(max_iters=total_iters)
checkpoint_config = dict(interval=total_iters)
optimizer = dict(lr=5e-4)
lr_config = dict(warmup_iters=500)
