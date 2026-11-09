_base_ = ['./C3_branch.py']

# C3 + SMG (fused vs F_T, G.T. boxes) + Hungarian logit KL (T=2, mean)
model = dict(loss_smg_weight=1.0, loss_logits_weight=1.0)
