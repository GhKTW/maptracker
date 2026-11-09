# FM-KD (E1): FTKD with the BEV reconstruction replaced by conditional flow matching
# (noise -> teacher BEV, conditioned on the randomly masked student BEV). Everything else,
# incl. the query reconstruction / logit distillation and the schedule, as in
# ftkd_nusc_oldsplit_r18_from_future_teacher_cached_bs64.py.
_base_ = ['./ftkd_nusc_oldsplit_r18_from_future_teacher_cached_bs64.py']

model = dict(
    type='FMKDDistiller',
    flow_hidden=256,
    flow_num_blocks=2,
    bev_mask_mode='random',
    bev_mask_ratio=0.5,
    t_logit_mean=0.0,
    t_logit_std=1.0,
    loss_bev_recon_weight=1.0,
)
