# AMap distillation (Future -> Static): the frozen future-aware teacher (R50) guides the
# current-frame-only student (R18). The student, the schedule and the evaluation are the
# same as amap_student_nusc_oldsplit_r18_current_frame.py, only the KD losses are added.
_base_ = ['./amap_student_nusc_oldsplit_r18_current_frame.py']

model = dict(
    _delete_=True,
    type='AMapDistiller',
    student_cfg={{_base_.model}},
    student_img_norm={{_base_.img_norm_cfg}},
    teacher_config='plugin/configs/amap/teacher/amap_teacher_nusc_oldsplit_future_stage3_joint_finetune.py',
    teacher_ckpt='work_dirs/amap_teacher_nusc_oldsplit_future_stage3_joint_finetune/latest.pth',
    # multi-level BEV distillation (Eq. 2), G.T.-masked
    loss_feat_basic_weight=1.0,
    loss_feat_refined_weight=1.0,
    feat_mask=True,
    # asymmetric query distillation (Eq. 3)
    loss_logits_weight=1.0,
    kd_temperature=1.0,
    logits_reduction='sum',
    match_cls_weight=5.0,
    match_reg_weight=50.0,
)

data = dict(
    train=dict(
        # the clip for the teacher: 4 frames sampled from the next 10 frames + the current one,
        # the student only takes the current frame
        multi_frame=5,
        sampling_span=10,
        temporal_direction='future',
    ),
)
