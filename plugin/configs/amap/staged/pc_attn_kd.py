_base_ = ['./pc_base.py']
# label + KD: same as pc_attn plus 10 x MSE(F_Tcur + G(F_Tcur), F_T), the future teacher's BEV as the target of the completed BEV
# lambda 10: KD = 1.8 at init (Delta energy ~0.18), under 1/2 of the decoder's final L_map (4.0 in oracle_FTcur)
model = dict(completer='attn', loss_kd_weight=10.0)
