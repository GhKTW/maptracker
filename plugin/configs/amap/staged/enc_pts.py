_base_ = ['./ss16_base.py']
# 11-18 BEV encoder capacity. Recipe = ss16_base (see enc_d6.py).
# S: image sampling per BEV cell x4. Height anchors per pillar 4 -> 8 (z from -3 to 5 m, 1 m apart instead of 2 m) and SCA
# deformable points 8 -> 32 per head and level (num_points is split over the anchors: 2 -> 4 offsets per anchor).
model = dict(backbone_cfg=dict(transformer=dict(encoder=dict(
    num_points_in_pillar=8,
    transformerlayers=dict(attn_cfgs=[
        dict(type='TemporalSelfAttention', embed_dims=256, num_levels=1),
        dict(type='SpatialCrossAttention', embed_dims=256,
             deformable_attention=dict(type='MSDeformableAttention3D', embed_dims=256, num_points=32, num_levels=3)),
    ])))))
