# Large-batch student baseline (w/o KD): same schedule as
# amap_kd_nusc_oldsplit_r18_from_future_teacher_cached_bs64.py, so that the comparison
# isolates the distillation.
_base_ = ['./amap_student_nusc_oldsplit_r18_current_frame.py']

samples_per_gpu = 64
num_gpus = 8
num_iters_per_epoch = 27968 // (num_gpus * samples_per_gpu)
num_epochs = 48
total_iters = num_epochs * num_iters_per_epoch

data = dict(samples_per_gpu=samples_per_gpu, workers_per_gpu=12)
optimizer = dict(lr=1.4e-3)
lr_config = dict(warmup_iters=100)
runner = dict(max_iters=total_iters)
checkpoint_config = dict(interval=num_iters_per_epoch * 12)
log_config = dict(interval=10)
evaluation = dict(interval=total_iters)
