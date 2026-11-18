_base_ = ['./skd_base.py']
# 11-17: C query KD (StructKDDistiller: matched-query logit KL x1, embedding cosine x2, line L1 x5; same weights as
# skdC / rcp32_C / r50_rcp32_C) on the original single-stage recipe = ss16_base exactly (student cfg + pipeline verified
# identical): joint from the start, 48 ep = 83,904 iters, 2 GPU x 8 = total batch 16, lr 2.5e-4, warmup 500, cosine.
# Question: is "query KD does not help" an artifact of the staged recipes? Reference ss16_base 65.4/67.5/68.1 (region reeval).
model = dict(loss_logits_weight=1.0, loss_emb_weight=2.0, loss_line_weight=5.0)
data = dict(samples_per_gpu=8, workers_per_gpu=8)
optimizer = dict(lr=2.5e-4)
lr_config = dict(warmup_iters=500)
runner = dict(max_iters=83904)
evaluation = dict(interval=83904)
checkpoint_config = dict(interval=83904 // 12)   # resume points after container restarts
find_unused_parameters = True
