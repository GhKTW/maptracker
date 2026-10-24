# AMap distillation (Past -> Static): the official MapTracker checkpoint (history frames,
# R50) as the teacher. Used to develop / check the distillation before the future-aware
# teacher is trained, and as the "Past -> Static" setting of the paper's Table 4.
_base_ = ['./amap_kd_nusc_oldsplit_r18_from_future_teacher.py']

model = dict(
    teacher_config='plugin/configs/maptracker/nuscenes_oldsplit/maptracker_nusc_oldsplit_5frame_span10_stage3_joint_finetune.py',
    teacher_ckpt='ckpts/maptracker_nusc_oldsplit_5frame_span10_stage3_joint_finetune/latest.pth',
)

data = dict(
    train=dict(temporal_direction='past'),
)
