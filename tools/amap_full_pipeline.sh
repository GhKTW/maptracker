#!/usr/bin/env bash
# AMap full training on 8 GPUs: future-aware teacher (stage 1 -> 2 -> 3) then the
# distillation into the current-frame student. Each step loads the previous one's
# work_dirs/<config name>/latest.pth, as set in the configs.
#   nohup bash tools/amap_full_pipeline.sh > logs/amap_full_pipeline.log 2>&1 &
# START_STEP=3 skips the steps before it (e.g. to resume after a finished stage 2).
set -e
cd "$(dirname "$0")/.."
START_STEP=${START_STEP:-1}
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-maptracker}"
export TORCH_HOME=${TORCH_HOME:-$(pwd)/.cache/torch}

STEPS=(
    plugin/configs/amap/teacher/amap_teacher_nusc_oldsplit_future_stage1_bev_pretrain.py
    plugin/configs/amap/teacher/amap_teacher_nusc_oldsplit_future_stage2_warmup.py
    plugin/configs/amap/fast/amap_teacher_nusc_oldsplit_future_stage3_joint_finetune_bs8.py
    plugin/configs/amap/fast/amap_kd_nusc_oldsplit_r18_from_future_teacher.py
)

for i in "${!STEPS[@]}"; do
    step=$((i + 1))
    [ "${step}" -lt "${START_STEP}" ] && continue
    echo "[$(date '+%F %T')] step ${step}: ${STEPS[$i]}"
    bash tools/dist_train.sh "${STEPS[$i]}" 8
done
echo "[$(date '+%F %T')] all done"
