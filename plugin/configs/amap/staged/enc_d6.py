_base_ = ['./ss16_base.py']
# 11-18 BEV encoder capacity (last series). Recipe = ss16_base exactly (single stage, 48 ep = 83,904 it, 2 GPU x 8 = total
# batch 16, lr 2.5e-4, warmup 500, cosine). Reference ss16_base region reeval 65.4 / A 67.5 / R 68.1 (orig 8x2 run 64.7/67.4/67.0).
# D: encoder depth 2 -> 6 layers (BEVFormer-base depth). Each layer = TSA + SCA + FFN + the current-frame conv block, so
# self-attention between BEV cells now happens in 5 image-dependent layers instead of 1 (layer-0 TSA sees only bev_embedding).
model = dict(backbone_cfg=dict(transformer=dict(encoder=dict(num_layers=6))))
