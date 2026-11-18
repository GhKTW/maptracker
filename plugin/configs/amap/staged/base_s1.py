_base_ = ['../fast/amap_student_nusc_oldsplit_r18_current_frame_bs64.py']

# Staged baseline (MapTracker paper's 3-stage recipe for the current-frame R18 student), 11-09.
# 2 GPUs x 64 = total batch 128 -> 218 iters/epoch. Stages: S1 6 ep / S2 2 ep / S3 40 ep (= 48 ep total).
# Stage 1: BEV pretraining, seg losses only (vector head skipped, like the paper's stage 1).
num_iters_per_epoch = 27968 // (2 * 64)
total_iters = 6 * num_iters_per_epoch
model = dict(skip_vector_head=True)
runner = dict(max_iters=total_iters)
checkpoint_config = dict(interval=total_iters)
evaluation = dict(interval=10 ** 9)  # no vector head -> no map evaluation
optimizer = dict(lr=7e-4)            # overridden by the sweep result (--cfg-options)
lr_config = dict(warmup_iters=200)
find_unused_parameters = True        # vector head unused (paper stage 1 also sets True)
