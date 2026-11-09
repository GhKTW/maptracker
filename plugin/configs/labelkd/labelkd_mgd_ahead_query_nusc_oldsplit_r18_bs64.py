# Label distillation (L5): L1 + the FTKD query losses towards the teacher decoder's outputs
# on the label BEV (masked query reconstruction x16, focal x2 + L1 x0.25 logit distillation)
_base_ = ['./labelkd_mgd_ahead_nusc_oldsplit_r18_bs64.py']

model = dict(
    query_channels=512,
    query_mask_ratio=0.5,
    loss_query_recon_weight=16.0,
    loss_fld_weight=1.0,
    fld_cls_weight=2.0,
    fld_reg_weight=0.25,
)
