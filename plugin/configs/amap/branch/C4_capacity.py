_base_ = ['./branch_base_bs64.py']

# branched architecture, no KD (capacity control)
model = dict(loss_cur_weight=0.0, loss_fut_weight=0.0)
