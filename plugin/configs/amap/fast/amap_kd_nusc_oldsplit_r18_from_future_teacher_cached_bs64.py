# Large-batch cached distillation: 64 per GPU (~125 GB of each H200), 48 epochs.
# bs x 6 cams = 384 is a multiple of the ms_deform_attn im2col_step (64).
# The schedule is shared with amap_student_nusc_oldsplit_r18_current_frame_bs64.py.
_base_ = ['./amap_kd_nusc_oldsplit_r18_from_future_teacher_cached.py']

samples_per_gpu = 64
num_gpus = 8
num_iters_per_epoch = 27968 // (num_gpus * samples_per_gpu)
num_epochs = 48
total_iters = num_epochs * num_iters_per_epoch

data = dict(samples_per_gpu=samples_per_gpu, workers_per_gpu=12)
# 32x the batch of the bs2 recipe (2.5e-4): sqrt scaling gives 1.4e-3, picked by the
# short lr runs (see datasets/taeu_outputs/README.md)
optimizer = dict(lr=1.4e-3)
lr_config = dict(warmup_iters=100)
runner = dict(max_iters=total_iters)
checkpoint_config = dict(interval=num_iters_per_epoch * 12)
log_config = dict(interval=10)
evaluation = dict(interval=total_iters)
