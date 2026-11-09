_base_ = ['./F1_full.py']

# future-branch target = probe mean of Delta (predictable part only)
model = dict(fut_target='probe_mu', probe_ckpt='datasets/taeu_outputs/branch_kd/probe/probe.pth')
