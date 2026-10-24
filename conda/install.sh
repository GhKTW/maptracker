#!/bin/bash
# MapTracker conda 환경 생성 (H200 / sm_90, 드라이버 CUDA 13.0 에서 cu118 바이너리 사용)
#   bash conda/install.sh            # 환경 이름 기본값: maptracker
#   ENV_NAME=foo bash conda/install.sh
set -e
cd "$(dirname "$0")/.."
REPO_DIR="$(pwd)"
ENV_NAME=${ENV_NAME:-maptracker}
MMDET3D_DIR=${MMDET3D_DIR:-${REPO_DIR}/third_party/mmdetection3d}

source "$(conda info --base)/etc/profile.d/conda.sh"

# 1) Python 3.8 + 빌드 도구
#    nvcc 11.8 은 gcc<=11 만 지원 → 호스트 gcc 버전과 무관하게 conda gcc 11 사용
conda create -y -n "${ENV_NAME}" python=3.8
conda activate "${ENV_NAME}"
conda install -y -c "nvidia/label/cuda-11.8.0" cuda-toolkit
conda install -y -c conda-forge gcc_linux-64=11 gxx_linux-64=11 ninja

export CUDA_HOME="${CONDA_PREFIX}"
export CC="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
export CXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++"
export CPATH="${CONDA_PREFIX}/include:${CONDA_PREFIX}/targets/x86_64-linux/include:${CPATH}"
export LIBRARY_PATH="${CONDA_PREFIX}/lib:${LIBRARY_PATH}"
export TORCH_CUDA_ARCH_LIST="9.0"
export FORCE_CUDA=1
export MMCV_WITH_OPS=1

python -m pip install --upgrade "pip<24" "setuptools==59.5.0" wheel

# 2) PyTorch 2.0.1 + cu118 (sm_90 포함)
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118

# 3) mmcv-full 1.7.2 — sm_90 으로 소스 빌드 (10~20분 소요)
pip install --no-build-isolation mmcv-full==1.7.2

# 4) mmdet / mmseg / MapTracker 의존성
pip install -r conda/requirements.txt

# 5) mmdetection3d v1.0.0rc6 + mmcv 상한 수정 (1.7.0 -> 1.7.2)
if [ ! -d "${MMDET3D_DIR}" ]; then
    git clone --depth 1 --branch v1.0.0rc6 https://github.com/open-mmlab/mmdetection3d.git "${MMDET3D_DIR}"
fi
sed -i "s/^mmcv_maximum_version = '1.7.0'/mmcv_maximum_version = '1.7.2'/" "${MMDET3D_DIR}/mmdet3d/__init__.py"
pip install --no-build-isolation -e "${MMDET3D_DIR}"

# requirements 에서 다시 끌려올 수 있는 버전 재고정
pip install numpy==1.23.5 numba==0.53.0 "setuptools==59.5.0"

# 6) 확인: sm_90 및 mmcv CUDA op
python - <<'EOF'
import torch, mmcv, mmdet, mmseg, mmdet3d
print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_arch_list())
print('mmcv', mmcv.__version__, 'mmdet', mmdet.__version__, 'mmseg', mmseg.__version__, 'mmdet3d', mmdet3d.__version__)
from mmcv.ops import MultiScaleDeformableAttention
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction
v = torch.rand(1, 64, 8, 32, device='cuda')
shapes = torch.tensor([[8, 8]], device='cuda')
start = torch.tensor([0], device='cuda')
loc = torch.rand(1, 10, 8, 1, 4, 2, device='cuda')
w = torch.rand(1, 10, 8, 1, 4, device='cuda').softmax(-1)
out = MultiScaleDeformableAttnFunction.apply(v, shapes, start, loc, w, 64)
print('ms_deform_attn on', torch.cuda.get_device_name(0), 'OK', tuple(out.shape))
EOF

echo ""
echo "Done. conda activate ${ENV_NAME}"
