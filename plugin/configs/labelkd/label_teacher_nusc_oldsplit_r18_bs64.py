# LabelDistill-style label teacher: G.T. map -> label encoder -> label BEV, decoded by the
# frozen seg / vector heads of the bs2 current-frame R18 baseline (mAP 64.7), only the
# encoder is trained (approximate inverse of the heads). Trained on G.T. only, no images.
# Evaluation runs the baseline inference with its BEV replaced by the label BEV (upper bound).
_base_ = ['../amap/student/amap_student_nusc_oldsplit_r18_current_frame.py']

roi_size = (60, 30)
coords_dim = 2
num_points = 20
permute = True
canvas_size = (200, 100)
thickness = 3

model = dict(
    _delete_=True,
    type='LabelMapTeacher',
    mapper_cfg={{_base_.model}},
    mapper_ckpt='datasets/taeu_outputs/checkpoints/amap_student_nusc_oldsplit_r18_current_frame/latest.pth',
    encoder_cfg=dict(channels=256, num_classes=3, num_points=num_points,
                     samples_per_seg=8, dilation=1, num_convs=3),
)

meta_keys = ('token', 'ego2img', 'sample_idx', 'ego2global_translation',
             'ego2global_rotation', 'scene_name')
train_pipeline = [
    dict(type='VectorizeMap', coords_dim=coords_dim, roi_size=roi_size, sample_num=num_points,
         normalize=True, permute=permute),
    dict(type='RasterizeMap', roi_size=roi_size, coords_dim=coords_dim, canvas_size=canvas_size,
         thickness=thickness, semantic_mask=True),
    dict(type='FormatBundleMap'),
    dict(type='Collect3D', keys=['vectors', 'semantic_mask'], meta_keys=meta_keys),
]
test_pipeline = [
    dict(type='VectorizeMap', coords_dim=coords_dim, roi_size=roi_size, sample_num=num_points,
         normalize=True, permute=False),
    dict(type='LoadMultiViewImagesFromFiles', to_float32=True),
    dict(type='ResizeMultiViewImages', size=(480, 800), change_intrinsics=True),
    dict(type='Normalize3D', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
    dict(type='PadMultiViewImages', size_divisor=32),
    dict(type='FormatBundleMap'),
    dict(type='Collect3D', keys=['img', 'vectors'], meta_keys=meta_keys + ('img_shape',)),
]

samples_per_gpu = 64
num_gpus = 8
num_iters_per_epoch = 27968 // (num_gpus * samples_per_gpu)
num_epochs = 24
total_iters = num_epochs * num_iters_per_epoch

data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=12,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline),
)
# LabelDistill step 1: lr 1e-3
optimizer = dict(_delete_=True, type='AdamW', lr=1e-3, weight_decay=1e-2)
lr_config = dict(warmup_iters=100)
runner = dict(max_iters=total_iters)
checkpoint_config = dict(interval=total_iters)
log_config = dict(interval=10)
evaluation = dict(interval=total_iters)
find_unused_parameters = False
