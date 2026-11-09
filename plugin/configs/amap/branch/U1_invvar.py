_base_ = ['./F1_full.py']

# future-branch loss weighted by clip(1/sigma^2) of the frozen probe
model = dict(fut_weight='inv_var', probe_ckpt='datasets/taeu_outputs/branch_kd/probe/probe.pth')
