_base_ = ['./oracle_base.py']
# decoder on F_T: the future-aware teacher's head-input BEV (upper bound if the BEV were teacher-level)
model = dict(bev_key='bev_refined')
