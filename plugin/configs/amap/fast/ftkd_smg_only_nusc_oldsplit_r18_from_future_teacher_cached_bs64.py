# N3 without the FTKD query reconstruction and logit distillation: the SMG contrastive BEV
# loss towards the future teacher alone, to tell which part gives the A-divider gain
_base_ = ['./ftkd_smg_nusc_oldsplit_r18_from_future_teacher_cached_bs64.py']

model = dict(loss_query_recon_weight=0.0, loss_fld_weight=0.0)
find_unused_parameters = True
