# Branched current / future KD (2026-11-05), bs64 recipe: 64/GPU x 8 = 512, 48 ep (2,592 iters),
# lr 1.4e-3, warmup 500. Joint cache: F_T, F_Tcur (teacher w/o memory), F_0, G.T. fg mask.
_base_ = ['../fast/amap_kd_nusc_oldsplit_r18_from_future_teacher_cached_bs64.py']

model = dict(
    type='BranchKDDistiller',
    teacher_cache='datasets/taeu_outputs/teacher_cache/branch_s3_bs8_f0base',
    loss_feat_basic_weight=0.0,
    loss_feat_refined_weight=0.0,
    loss_logits_weight=0.0,
    kd_temperature=2.0,
    logits_reduction='mean',
    branch=True,
    # 20: per-element MSE x 20 -> KD ~0.5x L_map at the start (lambda=1 was ~3%, 11-06)
    loss_cur_weight=20.0,
    loss_fut_weight=20.0,
    loss_fused_weight=0.0,
    fut_target='delta',
    fut_weight='none',
)
optimizer = dict(lr=1.4e-3)
lr_config = dict(warmup_iters=500)
find_unused_parameters = False
