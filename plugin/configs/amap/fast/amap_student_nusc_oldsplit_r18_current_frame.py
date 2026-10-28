# AMap student baseline (w/o KD), same recipe as the KD student (bs2 x 8 GPUs, lr 2.5e-4,
# 48 epochs) so that the comparison isolates the distillation. Only the run-time knobs
# that do not change the training differ:
# - one endless sampler over the epochs (same index order): no prefetch drain per epoch
# - no unused-parameter search in DDP (every student parameter gets a gradient)
# - a single evaluation at the end (one full val eval takes ~25 min)
_base_ = ['../student/amap_student_nusc_oldsplit_r18_current_frame.py']

num_iters_per_epoch = 27968 // (8 * 2)
num_epochs = 48

data = dict(infinite_sampler=True)
find_unused_parameters = False
evaluation = dict(interval=num_epochs * num_iters_per_epoch)
