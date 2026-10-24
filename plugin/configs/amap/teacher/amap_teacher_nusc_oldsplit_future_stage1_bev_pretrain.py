# AMap teacher (stage 1/3): MapTracker with the time axis reversed, i.e. the "history"
# is the future frames. Same settings as MapTracker otherwise.
_base_ = ['../../maptracker/nuscenes_oldsplit/maptracker_nusc_oldsplit_5frame_span10_stage1_bev_pretrain.py']

data = dict(
    train=dict(temporal_direction='future'),
    val=dict(temporal_direction='future'),
    test=dict(temporal_direction='future'),
)
