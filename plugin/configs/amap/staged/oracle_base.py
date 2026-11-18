_base_ = ['./skd_base.py']
# 11-15 oracle diagnosis (OracleBEVMapper, plugin/models/mapers/oracle_bev.py): the student's decoder (vector head +
# seg head, trained from scratch) on a FIXED cached BEV instead of its own; image backbone + BEV encoder bypassed.
# One stage = S2 + S3 length (42 ep), lr 5e-4 as base_s2/s3. Train BEVs = branch_s3_bs8_f0base (train tokens),
# val BEVs = branch_s3_bs8_f0base_val (tools/branch_cache.py --val).
num_iters_per_epoch = 27968 // (2 * 64)
total_iters = 24 * num_iters_per_epoch   # shortened from 42 ep (2.2 s/it, decoder-only); same for all three oracles
model = dict(type='OracleBEVMapper', bev_key='bev_refined',
             val_cache='datasets/taeu_outputs/teacher_cache/branch_s3_bs8_f0base_val')
runner = dict(max_iters=total_iters)
checkpoint_config = dict(interval=num_iters_per_epoch * 4)
evaluation = dict(interval=total_iters)
optimizer = dict(lr=5e-4)
lr_config = dict(warmup_iters=500)
find_unused_parameters = True
