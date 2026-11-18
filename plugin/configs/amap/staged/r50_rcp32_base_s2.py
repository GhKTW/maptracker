_base_ = ['./rcp32_base_s2.py']
# 11-15: R50 student (torchvision ImageNet R50, pytorch style, no DCN) instead of R18; everything else = rcp32_base_s2
# (staged, total batch 32, no KD, with memory block). Answers: does image-backbone capacity raise the current-frame student?
model = dict(backbone_cfg=dict(
    img_backbone=dict(depth=50, pretrained='torchvision://resnet50'),
    img_neck=dict(in_channels=[512, 1024, 2048])))
load_from = 'work_dirs/stg_r50_rcp32_base_s1/latest.pth'
