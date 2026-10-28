# AMap teacher stage 3 with 8 samples per GPU (total 64) instead of 2 (total 16).
# On 8x H200 this is ~16% faster per sample (the step time is bound by per-sample
# CPU-side work, not by GPU compute, so larger batches barely help).
# The lr was picked from short runs from the public MapTracker stage 2 checkpoint.
# Evaluation only at the middle and at the end: one full val eval (with the
# ahead / rear region metrics) takes ~25 min.
_base_ = ['../teacher/amap_teacher_nusc_oldsplit_future_stage3_joint_finetune.py']

num_gpus = 8
batch_size = 8
num_iters_per_epoch = 27968 // (num_gpus * batch_size)
num_epochs = 48

# infinite_sampler: one endless sampler over the epochs (same index order), so the
# prefetch does not drain at each epoch boundary (~2.5 min per epoch otherwise)
data = dict(samples_per_gpu=batch_size, workers_per_gpu=12, infinite_sampler=True)

# sqrt scaling of the batch x4. In 180-iter runs the linear scaling (2e-3) diverged,
# while 1e-3 shows the same early loss rise as the original bs2 / 5e-4 recipe
optimizer = dict(lr=1e-3)
# same warmup length in samples as the original (500 iters x 16)
lr_config = dict(warmup_iters=125)

evaluation = dict(interval=num_epochs // 2 * num_iters_per_epoch)
checkpoint_config = dict(interval=num_epochs // 8 * num_iters_per_epoch)
runner = dict(max_iters=num_epochs * num_iters_per_epoch)
