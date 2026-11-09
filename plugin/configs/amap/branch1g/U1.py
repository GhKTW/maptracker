_base_ = ['../branch/U1_invvar.py']

# F1 + future-branch weight 1/sigma^2
# 1 GPU x 64 = total batch 64 (8-GPU version: 512) -> 437 iters/epoch, 48 epochs = 20,976 iters
# lr / warmup chosen by the 1-GPU sweep (11-09), see datasets/taeu_outputs/branch_kd_1g/
num_iters_per_epoch = 27968 // 64
total_iters = 48 * num_iters_per_epoch
runner = dict(max_iters=total_iters)
checkpoint_config = dict(interval=num_iters_per_epoch * 4)  # every 4 ep (~2.5 h): container restarts kill runs (11-05, 11-09), resume with --resume-from
evaluation = dict(interval=total_iters)
optimizer = dict(lr=5e-4)
lr_config = dict(warmup_iters=500)
# monitoring only: log the future-branch error split into predictable (|B_fut - mu|^2) and floor (|Delta - mu|^2)
model = dict(log_fut_split=True, probe_ckpt='datasets/taeu_outputs/branch_kd/probe/probe.pth')
