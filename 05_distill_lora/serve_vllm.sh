#!/usr/bin/env bash
# 図表つきの Qwen3.8 と Gemma 4 を、同じ vLLM 0.24 で起動する.
# 学習用 .venv は使わない. 同じ場所の .venv_vllm を使う.
# --language-model-only は付けない. 図表が読めなくなる.
# CUDA 13.0 ドライバ（580 以上）が要る. 12.8 では Gemma 4 の wheel が落ちる.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
VENV="$HERE/.venv_vllm"
WANT_VLLM="0.24.0"
ROLE="${1:-}"

if [[ "$ROLE" != "qwen" && "$ROLE" != "gemma" ]]; then
  echo "使い方: bash serve_vllm.sh qwen|gemma" >&2
  echo "  qwen  : 蒸留（思考オンの Qwen3.8）" >&2
  echo "  gemma : 評価（思考オフの Gemma 4。LoRA があれば載せる）" >&2
  exit 2
fi

installed=""
if [[ -x "$VENV/bin/python" ]]; then
  installed="$("$VENV/bin/python" -c "import importlib.metadata as m; print(m.version('vllm'))" 2>/dev/null || true)"
fi
if [[ "$installed" != "$WANT_VLLM" ]]; then
  echo "vLLM ${WANT_VLLM} を .venv_vllm に入れます. 今は ${installed:-未インストール} です."
  if [[ ! -x "$VENV/bin/python" ]]; then
    uv venv "$VENV" --python 3.13
  fi
  uv pip uninstall --python "$VENV/bin/python" -y \
    vllm torch torchvision torchaudio torchcodec xformers 2>/dev/null || true
  uv pip install --python "$VENV/bin/python" "vllm==${WANT_VLLM}"
  uv pip install --python "$VENV/bin/python" "transformers>=5.8.0,<5.15.0"
fi

cudart_dir=""
nvidia_libs=""
if [[ -x "$VENV/bin/python" ]]; then
  cudart_dir="$("$VENV/bin/python" -c "from pathlib import Path; import sys; hits=list(Path(sys.prefix).rglob('libcudart.so.13')); print(hits[0].parent if hits else '')")"
  nvidia_libs="$("$VENV/bin/python" -c "
from pathlib import Path
import sys
root = Path(sys.prefix)
dirs = []
for pat in ('**/nvidia/*/lib', '**/torch/lib', '**/nvidia/**/lib'):
    for p in root.glob(pat):
        if p.is_dir():
            dirs.append(str(p))
print(':'.join(dict.fromkeys(dirs)))
")"
fi
if [[ -z "$cudart_dir" ]]; then
  echo "libcudart.so.13 が無いので CUDA 13 runtime を .venv_vllm に入れます."
  uv pip install --python "$VENV/bin/python" nvidia-cuda-runtime-cu13
  cudart_dir="$("$VENV/bin/python" -c "from pathlib import Path; import sys; hits=list(Path(sys.prefix).rglob('libcudart.so.13')); print(hits[0].parent if hits else '')")"
fi
if [[ -z "$cudart_dir" ]]; then
  echo "libcudart.so.13 が見つかりません. 次を試してください:" >&2
  echo "  sudo apt-get install -y cuda-cudart-13-0" >&2
  exit 1
fi
export LD_LIBRARY_PATH="${cudart_dir}${nvidia_libs:+:$nvidia_libs}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# FlashInfer の JIT は ninja と nvcc が要る. Toolkit は入れていないのでサンプラーは PyTorch 側にする.
export VLLM_USE_FLASHINFER_SAMPLER=0
export FLASHINFER_DISABLE_JIT=1
uv pip install --python "$VENV/bin/python" ninja >/dev/null
export PATH="$VENV/bin:$PATH"
echo "CUDA runtime: ${cudart_dir}"

export MAX_JOBS="${MAX_JOBS:-4}"
LOG="$HERE/serve_${ROLE}.log"

if [[ "$ROLE" == "qwen" ]]; then
  MODEL="${VLLM_MODEL:-Qwen/Qwen3.8-27B}"
  GPU_UTIL="${GPU_UTIL:-0.90}"
  MAX_LEN="${MAX_LEN:-8192}"
  MM_IMAGE=1
  extra=()
else
  MODEL="${VLLM_MODEL:-google/gemma-4-31B-it}"
  GPU_UTIL="${GPU_UTIL:-0.85}"
  MAX_LEN="${MAX_LEN:-16384}"
  MM_IMAGE=12
  extra=(--enforce-eager)
  SFT_DIR="${SFT_DIR:-$HERE/outputs/gemma4_hakusho_sft}"
  if [[ -f "$SFT_DIR/adapter_config.json" ]]; then
    extra+=(--enable-lora --lora-modules "sft=${SFT_DIR}" --max-loras 1 --max-lora-rank 16)
    echo "LoRA を載せます: sft=${SFT_DIR}"
  else
    echo "LoRA が無いのでベースだけ起動します: ${SFT_DIR}"
  fi
fi

echo "vLLM ${WANT_VLLM} を起動します: ${ROLE} ${MODEL}"
echo "環境: ${VENV}"
echo "待ち受けは 127.0.0.1:8000 のみ"
echo "ログ: ${LOG}"

exec > >(tee -a "$LOG") 2>&1
exec "$VENV/bin/vllm" serve "$MODEL" \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype auto \
  --gpu-memory-utilization "$GPU_UTIL" \
  --max-model-len "$MAX_LEN" \
  --limit-mm-per-prompt "{\"image\": ${MM_IMAGE}}" \
  "${extra[@]}"
