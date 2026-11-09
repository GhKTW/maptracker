# E5 bridge + SMG without the flow matching loss: the same refine module in the inference
# path, trained by the task loss, the SMG contrastive loss and the FTKD query losses only.
# Tells whether flow matching is the module that raised E5 + SMG
_base_ = ['./fmkd_smg_bridge_nusc_oldsplit_r18_from_future_teacher_cached_bs64.py']

model = dict(loss_flow_weight=0.0)
find_unused_parameters = True
