# FTKD (masked feature reconstruction + future-guided logit distillation) with the AMap
# future-aware teacher and current-frame R18 student, same large-batch schedule as
# amap_kd_nusc_oldsplit_r18_from_future_teacher_cached_bs64.py.
# The cache also holds the teacher's query features (tools/amap_cache_teacher.py).
_base_ = ['./amap_kd_nusc_oldsplit_r18_from_future_teacher_cached_bs64.py']

model = dict(
    type='FTKDDistiller',
    teacher_cache='datasets/taeu_outputs/teacher_cache/ftkd_amap_teacher_nusc_oldsplit_future_stage3_joint_finetune_bs8',
    # the AMap losses are replaced by the FTKD ones
    loss_feat_basic_weight=0.0,
    loss_feat_refined_weight=0.0,
    loss_logits_weight=0.0,
    bev_channels=256,
    query_channels=512,
    # paper defaults: mask ratio 0.5, lambda_2 = 16, lambda_3 = 1, alpha = 2, beta = 0.25.
    # lambda_1 (1e-3 in the paper, for FPN features) is set for the BEV map, whose values
    # are ~unit scale (mean square 0.33) like the normalized query features
    bev_mask_ratio=0.5,
    query_mask_ratio=0.5,
    loss_bev_recon_weight=1.0,
    loss_query_recon_weight=16.0,
    loss_fld_weight=1.0,
    fld_cls_weight=2.0,
    fld_reg_weight=0.25,
)
