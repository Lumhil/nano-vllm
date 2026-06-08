#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-nano-vllm-qwen3}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"  # 使用 3.10 更稳定
MODEL_DIR="${MODEL_DIR:-$REPO_ROOT/huggingface/Qwen3-0.6B}"
HF_FORCE_DOWNLOAD="${HF_FORCE_DOWNLOAD:-1}"
CONDA_CHANNEL="${CONDA_CHANNEL:-conda-forge}"

# PyTorch 2.4.0 with CUDA 12.1 (最兼容 flash-attn)
TORCH_VERSION="2.4.0"
TORCHVISION_VERSION="0.19.0"
TORCHAUDIO_VERSION="2.4.0"
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda is required but was not found in PATH." >&2
    exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"

# 删除旧环境（如果存在）
if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    echo "Removing existing environment: $ENV_NAME"
    conda deactivate 2>/dev/null || true
    conda env remove -n "$ENV_NAME" -y
fi

# 创建新环境
echo "Creating new environment: $ENV_NAME with Python $PYTHON_VERSION"
conda create -y -n "$ENV_NAME" --override-channels -c "$CONDA_CHANNEL" "python=$PYTHON_VERSION"

conda activate "$ENV_NAME"

# 确保 pip 已安装（conda create 有时不会自动安装 pip）
echo "Installing pip..."
conda install -y -n "$ENV_NAME" --override-channels -c "$CONDA_CHANNEL" pip

# 确保 pip 是最新的
python -m pip install --upgrade pip

# 安装 setuptools 兼容版本
python -m pip install "setuptools<82" wheel

# 安装基础依赖
python -m pip install "huggingface_hub[hf_transfer]" packaging psutil ninja

# 安装 PyTorch 2.4.0 (与 flash-attn 预编译包兼容)
echo "Installing PyTorch $TORCH_VERSION with CUDA support..."
python -m pip install --index-url "$TORCH_INDEX_URL" \
    torch==$TORCH_VERSION \
    torchvision==$TORCHVISION_VERSION \
    torchaudio==$TORCHAUDIO_VERSION

# 安装其他依赖
python -m pip install "triton>=3.0.0" "transformers>=4.51.0" xxhash numpy safetensors tqdm einops

# ============================================
# 安装 flash-attn (使用预编译包)
# ============================================
echo "Installing flash-attn..."

# 检查 PyTorch 和 CUDA
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
"

# 方法1: 从 PyPI 使用清华镜像
if pip install flash-attn==2.6.3 -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir --timeout 100 2>/dev/null; then
    echo "✓ flash-attn installed successfully from PyPI mirror"
else
    # 方法2: 尝试官方 PyPI
    echo "Trying official PyPI..."
    if pip install flash-attn==2.6.3 --no-cache-dir --timeout 100 2>/dev/null; then
        echo "✓ flash-attn installed successfully from official PyPI"
    else
        # 方法3: 直接下载预编译 wheel
        echo "Downloading pre-compiled wheel directly..."
        cd /tmp
        wget -O flash_attn.whl https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4.0cxx11abiTRUE-cp310-cp310-linux_x86_64.whl \
            2>/dev/null || wget -O flash_attn.whl https://ghproxy.com/https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4.0cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
        
        if [ -f flash_attn.whl ]; then
            pip install flash_attn.whl
            rm flash_attn.whl
            echo "✓ flash-attn installed from direct download"
        else
            # 方法4: 编译安装
            echo "Compiling flash-attn from source (this may take 10-15 minutes)..."
            export MAX_JOBS=2
            export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
            pip install flash-attn==2.6.3 --no-build-isolation --timeout 1000
        fi
    fi
fi

# 验证 flash-attn 安装
cd "$REPO_ROOT"
if python -c "import flash_attn; print(f'✓ flash-attn version: {flash_attn.__version__}')" 2>/dev/null; then
    echo "Flash-attn is working correctly"
else
    echo "⚠ Flash-attn verification failed, attempting fallback..."
    # 如果 flash-attn 不工作，安装 xformers 作为备选
    pip install xformers==0.0.28 --index-url "$TORCH_INDEX_URL"
    
    # 创建兼容层
    mkdir -p "$REPO_ROOT/nanovllm/layers"
    cat > "$REPO_ROOT/nanovllm/layers/flash_attn_compat.py" << 'EOF'
"""Compatibility layer for flash-attn using xformers"""
import torch
try:
    import xformers.ops as xops
    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False
    print("Warning: xformers not available, using native PyTorch")

def flash_attn_varlen_func(q, k, v, causal=True, softmax_scale=None):
    """Drop-in replacement for flash_attn_varlen_func"""
    if XFORMERS_AVAILABLE:
        # xformers expects [batch, seqlen, num_heads, head_dim]
        if q.dim() == 4:
            attn_mask = xops.LowerTriangularMask() if causal else None
            return xops.memory_efficient_attention(q, k, v, attn_bias=attn_mask, scale=softmax_scale)
    
    # Fallback to native PyTorch
    scale = softmax_scale or (q.shape[-1] ** -0.5)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if causal:
        causal_mask = torch.triu(torch.ones_like(scores), diagonal=1).bool()
        scores.masked_fill_(causal_mask, float('-inf'))
    attn_weights = torch.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, v)

def flash_attn_with_kvcache(q, k, v, k_cache=None, v_cache=None, causal=True):
    """Simplified version for kv cache"""
    return flash_attn_varlen_func(q, k, v, causal=causal)
EOF
    
    # 修改 attention.py 使用兼容层
    if [ -f "$REPO_ROOT/nanovllm/layers/attention.py" ]; then
        cp "$REPO_ROOT/nanovllm/layers/attention.py" "$REPO_ROOT/nanovllm/layers/attention.py.bak"
        sed -i 's/from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache/from nanovllm.layers.flash_attn_compat import flash_attn_varlen_func, flash_attn_with_kvcache/' \
            "$REPO_ROOT/nanovllm/layers/attention.py"
        echo "✓ Created fallback compatibility layer"
    fi
fi

# 安装主包
echo "Installing nano-vllm..."
python -m pip install -e "$REPO_ROOT" --no-deps

# 下载模型
echo "Downloading model..."
mkdir -p "$(dirname "$MODEL_DIR")"

# 设置 Hugging Face 镜像（国内加速）
export HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co}
if [ ! -d "$MODEL_DIR" ] || [ "$HF_FORCE_DOWNLOAD" = "1" ]; then
    download_args=(
        download
        Qwen/Qwen3-0.6B
        --local-dir
        "$MODEL_DIR"
    )
    
    if [ "$HF_FORCE_DOWNLOAD" = "1" ]; then
        download_args+=(--force-download)
    fi
    
    hf "${download_args[@]}"
else
    echo "Model already exists at $MODEL_DIR"
fi

# 最终验证
echo ""
echo "============================================="
echo "Verifying installation..."
echo "============================================="

python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"

python -c "import transformers; print(f'Transformers version: {transformers.__version__}')" 2>/dev/null || echo "Transformers: not installed"

if python -c "import flash_attn; print(f'Flash-Attn: {flash_attn.__version__}')" 2>/dev/null; then
    echo "✓ Flash-Attn: Working"
elif python -c "import xformers; print(f'XFormers: {xformers.__version__}')" 2>/dev/null; then
    echo "✓ XFormers: Working (as fallback)"
else
    echo "⚠ Attention backend: Using native PyTorch"
fi

cat <<EOF

=============================================
✅ Setup Complete!
=============================================

Activate the environment:
  conda activate $ENV_NAME

Run the example:
  python $REPO_ROOT/example.py

Environment details:
  - Python: $PYTHON_VERSION
  - PyTorch: $TORCH_VERSION (CUDA 12.1)
  - Model: $MODEL_DIR

If you encounter network issues, try:
  export HF_ENDPOINT=https://hf-mirror.com  # Use HF mirror in China
  python $REPO_ROOT/example.py

=============================================
EOF