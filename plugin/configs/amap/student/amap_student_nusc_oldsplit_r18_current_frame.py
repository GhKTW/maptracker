# AMap student baseline: MapTracker with all temporal modules removed (current frame only),
# ResNet-18 backbone. Trained with the final MapTracker stage only (no BEV pretrain / warmup),
# from ImageNet weights. Also the "w/o KD" baseline of the distillation experiments.
_base_ = ['../../maptracker/nuscenes_oldsplit/maptracker_nusc_oldsplit_5frame_span10_stage3_joint_finetune.py']

# torchvision ResNet-18 weights expect RGB inputs with the ImageNet statistics
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

img_h = 480
img_w = 800
img_size = (img_h, img_w)
roi_size = (60, 30)
coords_dim = 2
num_points = 20
permute = True
canvas_size = (200, 100)
thickness = 3
embed_dims = 512

model = dict(
    temporal=False,
    history_steps=0,
    test_time_history_steps=0,
    use_memory=False,
    mem_len=None,
    backbone_cfg=dict(
        # no BEV memory: no warped history BEV, current-frame-only refinement layer
        history_steps=0,
        img_backbone=dict(
            _delete_=True,
            type='ResNet',
            depth=18,
            num_stages=4,
            out_indices=(1, 2, 3),
            frozen_stages=-1,
            norm_cfg=dict(type='BN2d'),
            norm_eval=True,
            style='pytorch',
            with_cp=False,
            pretrained='torchvision://resnet18'),
        img_neck=dict(in_channels=[128, 256, 512]),
        transformer=dict(encoder=dict(mem_history_steps=0)),
    ),
    head_cfg=dict(
        transformer=dict(
            decoder=dict(
                transformerlayers=dict(
                    # no vector memory cross attention
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=embed_dims,
                            num_heads=8,
                            attn_drop=0.1,
                            proj_drop=0.1,
                        ),
                        dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=embed_dims,
                            num_heads=8,
                            num_levels=1,
                            num_points=num_points,
                            dropout=0.1,
                        ),
                    ],
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'),
                ),
            ),
        ),
    ),
)

train_pipeline = [
    dict(
        type='VectorizeMap',
        coords_dim=coords_dim,
        roi_size=roi_size,
        sample_num=num_points,
        normalize=True,
        permute=permute,
    ),
    dict(
        type='RasterizeMap',
        roi_size=roi_size,
        coords_dim=coords_dim,
        canvas_size=canvas_size,
        thickness=thickness,
        semantic_mask=True,
    ),
    dict(type='LoadMultiViewImagesFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='ResizeMultiViewImages',
         size=img_size, # H, W
         change_intrinsics=True,
         ),
    dict(type='Normalize3D', **img_norm_cfg),
    dict(type='PadMultiViewImages', size_divisor=32),
    dict(type='FormatBundleMap'),
    dict(type='Collect3D', keys=['img', 'vectors', 'semantic_mask'], meta_keys=(
        'token', 'ego2img', 'sample_idx', 'ego2global_translation',
        'ego2global_rotation', 'img_shape', 'scene_name'))
]

test_pipeline = [
    dict(type='LoadMultiViewImagesFromFiles', to_float32=True),
    dict(type='ResizeMultiViewImages',
         size=img_size, # H, W
         change_intrinsics=True,
         ),
    dict(type='Normalize3D', **img_norm_cfg),
    dict(type='PadMultiViewImages', size_divisor=32),
    dict(type='FormatBundleMap'),
    dict(type='Collect3D', keys=['img'], meta_keys=(
        'token', 'ego2img', 'sample_idx', 'ego2global_translation',
        'ego2global_rotation', 'img_shape', 'scene_name'))
]

data = dict(
    train=dict(
        pipeline=train_pipeline,
        # single frames, no temporal clip / track matching
        multi_frame=False,
        matching=False,
    ),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline),
)

# trained from ImageNet weights, not from the MapTracker stage 2 checkpoint
load_from = None

# half of the MapTracker stage 3 lr: stage 3 fine-tunes a pretrained model, and from
# ImageNet weights 5e-4 makes the gradients explode right after the warmup (~iter 900,
# with or without KD) and the model never recovers (KD run collapsed to 0.2 mAP).
# 2.5e-4 and 1.25e-4 were both stable in 1800-iter runs, 2.5e-4 with the lower losses.
optimizer = dict(lr=2.5e-4)
