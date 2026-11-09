# Label distillation (L1): the frozen label teacher (G.T. -> label BEV, label_teacher_*.py)
# guides the current-frame R18 student with the MGD / FTKD masked reconstruction, mask on
# the ahead half (0.75 ahead / 0.25 rear, mean 0.5, the best mask of the FM-KD runs).
# Same student and bs64 schedule as amap_student_nusc_oldsplit_r18_current_frame_bs64.py.
_base_ = ['../amap/fast/amap_student_nusc_oldsplit_r18_current_frame_bs64.py']

model = dict(
    _delete_=True,
    type='LabelKDDistiller',
    student_cfg={{_base_.model}},
    teacher_config='plugin/configs/labelkd/label_teacher_nusc_oldsplit_r18_bs64.py',
    teacher_ckpt='work_dirs/label_teacher_bs64/latest.pth',
    loss_type='mgd',
    loss_bev_weight=1.0,
    bev_channels=256,
    bev_mask_mode='ahead',
    bev_mask_ratio=0.5,
    ahead_mask_ratio=0.75,
)

# the lr of every bs64 run so far
optimizer = dict(lr=7e-4)
find_unused_parameters = True
