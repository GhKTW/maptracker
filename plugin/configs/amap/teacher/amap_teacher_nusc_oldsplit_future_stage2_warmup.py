# AMap teacher (stage 2/3): MapTracker with the time axis reversed
_base_ = ['../../maptracker/nuscenes_oldsplit/maptracker_nusc_oldsplit_5frame_span10_stage2_warmup.py']

data = dict(
    train=dict(temporal_direction='future'),
    val=dict(temporal_direction='future'),
    test=dict(temporal_direction='future'),
)

load_from = "work_dirs/amap_teacher_nusc_oldsplit_future_stage1_bev_pretrain/latest.pth"
