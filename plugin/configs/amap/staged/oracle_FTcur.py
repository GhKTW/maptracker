_base_ = ['./oracle_base.py']
# decoder on F_Tcur: the same R50 teacher with empty memory = current frame only (best current-frame BEV we have)
model = dict(bev_key='cur_bev_refined')
