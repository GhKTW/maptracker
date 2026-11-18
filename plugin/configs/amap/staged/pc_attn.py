_base_ = ['./pc_base.py']
# label only: F_Tcur -> global self-attention completer (whole 60 x 30 m visible to every cell) -> decoder, G.T. losses only
model = dict(completer='attn', loss_kd_weight=0.0)
