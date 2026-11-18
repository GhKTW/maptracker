_base_ = ['./ss16_base.py']
# 11-18 BEV encoder capacity. Recipe = ss16_base (see enc_d6.py).
# R: BEV grid 50 x 100 (0.6 m cells) -> 100 x 200 (0.3 m cells), 4x the BEV queries; channels and decoder unchanged.
# The seg head's canvas is 200 x 100 already, so its single 2x upsample block disappears (one 3x3 conv fewer, ~0.6M params);
# the vector decoder reads the BEV with normalized deformable sampling + normalized sine positions (size-agnostic).
bev_h_ = 100
bev_w_ = 200
model = dict(
    bev_h=bev_h_, bev_w=bev_w_,
    backbone_cfg=dict(bev_h=bev_h_, bev_w=bev_w_,
                      positional_encoding=dict(row_num_embed=bev_h_, col_num_embed=bev_w_)),
    seg_cfg=dict(bev_size=(bev_w_, bev_h_)))
