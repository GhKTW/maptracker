_base_ = ['../branch/C4_capacity.py']

# branch arch (future predictor), no KD
# 1 GPU x 64 = total batch 64 (8-GPU version: 512) -> 437 iters/epoch, 48 epochs = 20,976 iters
# lr / warmup chosen by the 1-GPU sweep (11-09), see datasets/taeu_outputs/branch_kd_1g/
num_iters_per_epoch = 27968 // 64
total_iters = 48 * num_iters_per_epoch
runner = dict(max_iters=total_iters)
checkpoint_config = dict(interval=num_iters_per_epoch * 4)  # every 4 ep (~2.5 h): container restarts kill runs (11-05, 11-09), resume with --resume-from
evaluation = dict(interval=total_iters)
optimizer = dict(lr=5e-4)
lr_config = dict(warmup_iters=500)
