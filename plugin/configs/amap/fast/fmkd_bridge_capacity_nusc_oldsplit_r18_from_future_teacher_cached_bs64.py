# Capacity control: the E5 bridge refine module in the inference path with every
# distillation loss off (no flow matching, no SMG, no FTKD query losses), trained by the task
# loss only. Tells how much of E5 / E5 + SMG comes from the extra module itself
_base_ = ['./fmkd_nusc_oldsplit_r18_from_future_teacher_cached_bs64_bridge.py']

model = dict(loss_bev_recon_weight=0.0, loss_query_recon_weight=0.0, loss_fld_weight=0.0)
optimizer = dict(lr=7e-4)
find_unused_parameters = True
