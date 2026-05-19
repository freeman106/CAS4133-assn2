#!/usr/bin/env bash
# ASSN2 Vessl Workspace 셋업 스크립트
# 새 워크스페이스에서: cd ~/CAS4133-assn2-stu && bash setup.sh

set -e

cd "$(dirname "$0")"

step() { echo; echo "==== $* ===="; }

step "0. 환경 확인"
nvidia-smi | head -5 || { echo "nvidia-smi 실패"; exit 1; }
[ -d "third_party/OpenRLHF" ] || { echo "third_party/OpenRLHF 없음. 압축 풀린 디렉토리에서 실행하세요."; exit 1; }

step "1. apt 패키지"
apt-get update -y
apt-get install -y --no-install-recommends git unzip python3-dev build-essential

step "2. Python alias + pip 업그레이드"
[ -f /usr/bin/python3 ] && ln -sf /usr/bin/python3 /usr/local/bin/python || true
pip install -U pip --no-cache-dir

step "3. torch 2.10.0+cu126 (기존 torch 제거 후 재설치)"
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
pip install --index-url https://download.pytorch.org/whl/cu126 \
    torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --no-cache-dir

step "4. GPU 동작 검증"
python -c "import torch; assert torch.cuda.is_available(), 'CUDA 사용 불가'; x=torch.randn(2,2,device='cuda'); print('torch', torch.__version__, 'gpu', torch.cuda.device_count(), 'matmul', (x@x).cpu().sum().item()); print('OK')"

step "5. OpenRLHF requirements에서 flash-attn 제거"
if ! grep -q "^flash-attn" third_party/OpenRLHF/requirements.txt 2>/dev/null; then
    echo "(이미 제거됨, 스킵)"
else
    sed -i.bak '/^flash-attn/d' third_party/OpenRLHF/requirements.txt
fi

step "6. OpenRLHF editable install"
pip install -e third_party/OpenRLHF --no-build-isolation --no-cache-dir

step "7. 학습 의존성"
pip install deepspeed==0.18.9 --no-cache-dir
pip install jupyter ipykernel papermill nbconvert hf_transfer --no-cache-dir

step "8. vLLM"
pip install vllm==0.19.1 --no-cache-dir

step "9. Jupyter kernel 등록"
python -m ipykernel install --user --name hw2 --display-name "Python (hw2)"

step "10. 최종 검증"
python -c "import openrlhf, torch, vllm; print('openrlhf:', openrlhf.__file__); print('torch:', torch.__version__); print('vllm:', vllm.__version__)"

step "DONE"
echo
echo "다음 단계:"
echo "  1. huggingface-cli login --token hf_xxxxxxxx"
echo "  2. (필요 시) export ASSN2_NUM_GPUS=2, ASSN2_SFT_MAX_LEN=4096 등 env var"
echo "  3. JupyterLab 또는 VSCode에서 assn2.ipynb 열고 kernel 'Python (hw2)' 선택"
