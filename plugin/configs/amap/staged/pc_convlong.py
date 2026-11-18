_base_ = ['./pc_base.py']
# label only: conv-only long-range hourglass completer (4x downsample + dilated blocks, about +-24 m)
model = dict(completer='conv_long', loss_kd_weight=0.0)
