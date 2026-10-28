# AMap distillation with the bs8 teacher of this folder. The student schedule is
# unchanged (2 per GPU: larger batches give no speed-up for the KD step), only the
# evaluation runs at the middle and at the end instead of every 6 epochs.
_base_ = ['../student/amap_kd_nusc_oldsplit_r18_from_future_teacher.py']

num_iters_per_epoch = 27968 // (8 * 2)
num_epochs = 48

model = dict(
    teacher_ckpt='work_dirs/amap_teacher_nusc_oldsplit_future_stage3_joint_finetune_bs8/latest.pth')

evaluation = dict(interval=num_epochs // 2 * num_iters_per_epoch)

# one endless sampler over the epochs (same index order): no prefetch drain at each
# epoch boundary
data = dict(infinite_sampler=True)
