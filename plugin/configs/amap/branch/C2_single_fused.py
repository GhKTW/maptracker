_base_ = ['./branch_base_bs64.py']

# single branch: predictor kept (same params as C3), fused BEV <- F_T
model = dict(loss_cur_weight=0.0, loss_fut_weight=0.0, loss_fused_weight=20.0)
