_base_ = ['./oracle_base.py']
# control: decoder on F_0, a KD-free bs64 R18 student's own frozen BEV (checks the oracle procedure itself)
model = dict(bev_key='f0_bev')
