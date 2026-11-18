_base_ = ['./enc_res.py']
# 11-18 BEV encoder capacity. Recipe = ss16_base (see enc_d6.py). D + S + R together: 6 layers, 8 anchors x 32 points,
# 100 x 200 grid = the largest encoder this series tests (upper bound of "more encoder capacity", channels unchanged).
model = dict(backbone_cfg=dict(transformer=dict(encoder=dict(
    num_layers=6,
    num_points_in_pillar=8,
    transformerlayers=dict(attn_cfgs=[
        dict(type='TemporalSelfAttention', embed_dims=256, num_levels=1),
        dict(type='SpatialCrossAttention', embed_dims=256,
             deformable_attention=dict(type='MSDeformableAttention3D', embed_dims=256, num_points=32, num_levels=3)),
    ])))))
