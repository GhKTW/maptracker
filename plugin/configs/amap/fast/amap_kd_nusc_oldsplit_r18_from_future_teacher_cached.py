# AMap distillation from the teacher outputs precomputed by tools/amap_cache_teacher.py
# (one future clip per sample instead of a new draw at every epoch): only the student
# runs, so the train data needs no clip and the step takes the student's time.
_base_ = ['./amap_kd_nusc_oldsplit_r18_from_future_teacher.py']

model = dict(
    teacher_cache='datasets/taeu_outputs/teacher_cache/amap_teacher_nusc_oldsplit_future_stage3_joint_finetune_bs8')

data = dict(train=dict(multi_frame=False))
# no teacher parameters any more, every student parameter gets a gradient
find_unused_parameters = False
