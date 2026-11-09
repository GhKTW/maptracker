_base_ = ['./F1_full.py']

# future-branch loss weighted by the G.T. fg mask (AMap-style)
model = dict(fut_weight='gt')
