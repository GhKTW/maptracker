_base_ = ['./branch_base_bs64.py']

# current branch only: student BEV <- F_Tcur, no future branch
model = dict(branch=False, loss_cur_weight=20.0, loss_fut_weight=0.0)
