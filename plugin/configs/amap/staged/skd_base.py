# 11-14 structure-level KD (StructKDDistiller, plugin/models/mapers/struct_kd.py) on the plain student WITH the
# memory-fusion block. No feature MSE, no branch, no SMG. Teacher = cached future teacher (F_T, query logits /
# lines / hs_embeds). Staged recipe identical to base_s{1,2,3} (2 GPUs x 64, S1 6 ep / S2 2 ep / S3 40 ep).
_base_ = ['../fast/amap_kd_nusc_oldsplit_r18_from_future_teacher_cached_bs64.py']
num_iters_per_epoch = 27968 // (2 * 64)
model = dict(
    type='StructKDDistiller',
    teacher_cache='datasets/taeu_outputs/teacher_cache/branch_s3_bs8_f0base',
    loss_feat_basic_weight=0.0, loss_feat_refined_weight=0.0, loss_logits_weight=0.0,
    kd_temperature=2.0, logits_reduction='mean')
