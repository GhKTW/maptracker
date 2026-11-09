_base_ = ['../branch/U3_probemu.py']

# F1 with the future-branch target = probe mean mu of Delta (predictable part only); B_cur <- F_Tcur, SMG, logit KD as F1
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
# KD weights from the 1-GPU round-2 sweep (11-09): feature KD lambda 5 (was 20), SMG 2 (was 1)
model = dict(log_fut_split=True, probe_ckpt='datasets/taeu_outputs/branch_kd/probe/probe.pth',
             loss_cur_weight=5.0, loss_fut_weight=5.0, loss_smg_weight=2.0)
