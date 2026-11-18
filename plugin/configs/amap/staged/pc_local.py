_base_ = ['./pc_base.py']
# label only, capacity control: the old local predictor (3x3 convs, +-4.2 m, 3.5M params) in place of the attention completer
model = dict(completer='local', loss_kd_weight=0.0)
