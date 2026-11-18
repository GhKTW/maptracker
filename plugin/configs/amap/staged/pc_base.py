_base_ = ['./oracle_FTcur.py']
# 11-17 predictability ceiling. Input = cached F_Tcur (the teacher's current-frame-only BEV, empty memory), then a
# completion module G (zero-init residual), then the student's decoder from scratch — exactly the oracle_FTcur
# procedure (24 ep, 2 GPUs x 64, lr 5e-4) plus G. References by the same procedure:
#   oracle_FTcur (no G) 62.4 / A 64.4 / R 65.3      oracle_FT (decoder on the future BEV) 69.4 / A 73.9 / R 68.5
# kd_delta_raw = per-element MSE(F_Tcur + G, F_T); its value at iter 1 (G = 0, ~0.18) means G explains nothing of Delta.
model = dict(completer='attn', loss_kd_weight=0.0)
