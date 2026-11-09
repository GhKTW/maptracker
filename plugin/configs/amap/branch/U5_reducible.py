_base_ = ['./F1_full.py']

# future-branch loss weighted by max(e_S - sigma^2, 0), w = 1 for the first 300 iters
model = dict(fut_weight='reducible', weight_warmup_iters=300, probe_ckpt='datasets/taeu_outputs/branch_kd/probe/probe.pth')
