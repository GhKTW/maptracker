_base_ = ['./C3_branch.py']

# C3 + stop-gradient on the predictor input
model = dict(pred_detach_input=True)
