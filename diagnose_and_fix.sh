#!/bin/bash
# MemTorch GPU安装诊断和修复脚本

echo "=========================================="
echo "MemTorch GPU 诊断和修复工具"
echo "=========================================="
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 步骤1：检查当前MemTorch版本
echo "步骤1：检查当前MemTorch安装"
echo "----------------------------------------"
python3 << EOF
try:
    import memtorch
    version = memtorch.__version__
    print(f"MemTorch版本: {version}")
    if '-cpu' in version:
        print("❌ 问题：安装的是CPU版本")
    else:
        print("✅ 安装的是GPU版本")
except ImportError:
    print("❌ MemTorch未安装")
EOF
echo ""

# 步骤2：检查CUDA环境
echo "步骤2：检查CUDA环境"
echo "----------------------------------------"
if command -v nvidia-smi &> /dev/null; then
    echo "✅ nvidia-smi可用"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    echo "❌ nvidia-smi不可用 - GPU驱动可能未安装"
fi
echo ""

if command -v nvcc &> /dev/null; then
    echo "✅ CUDA Toolkit已安装"
    nvcc --version | grep "release"
else
    echo "❌ CUDA Toolkit未安装或不在PATH中"
fi
echo ""

# 步骤3：检查PyTorch CUDA支持
echo "步骤3：检查PyTorch CUDA支持"
echo "----------------------------------------"
python3 << EOF
import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU数量: {torch.cuda.device_count()}")
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    print("✅ PyTorch支持CUDA")
else:
    print("❌ PyTorch不支持CUDA")
EOF
echo ""

# 步骤4：修复建议
echo "=========================================="
echo "修复方案"
echo "=========================================="
echo ""

python3 << 'PYTHON_EOF'
import torch
import sys

try:
    import memtorch
    is_cpu_version = '-cpu' in memtorch.__version__
except ImportError:
    is_cpu_version = True

cuda_available = torch.cuda.is_available()

if not cuda_available:
    print("⚠️  问题：PyTorch不支持CUDA")
    print("解决方案：")
    print("  1. 检查autodl平台是否选择了GPU实例")
    print("  2. 重新安装支持CUDA的PyTorch：")
    print("     pip uninstall torch torchvision -y")
    print("     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    sys.exit(1)

if is_cpu_version:
    print("⚠️  问题：MemTorch是CPU版本")
    print("解决方案：")
    print("")
    print("【方法1】重新编译GPU版本（推荐）：")
    print("  cd /workspace")
    print("  pip uninstall memtorch memtorch-cpu -y")
    print("  sed -i 's/CUDA = False/CUDA = True/' setup.py")
    print("  pip install -e .")
    print("")
    print("【方法2】安装预编译GPU版本：")
    print("  pip uninstall memtorch memtorch-cpu -y")
    print("  pip install memtorch")
    print("")
else:
    print("✅ MemTorch GPU版本已正确安装")

PYTHON_EOF

echo ""
echo "=========================================="
echo "内存优化建议"
echo "=========================================="
echo ""
cat << 'EOF'
如果仍然遇到内存问题，请在Jupyter Notebook中添加以下优化：

```python
# 在notebook开头添加
import gc
import torch

# 1. 启用混合精度训练
torch.set_default_dtype(torch.float16)

# 2. 减小批次大小
batch_size = 32  # 从128减到32

# 3. 减小交叉阵列规模
tile_shape = (64, 64)  # 从(128, 128)减小

# 4. 启用梯度检查点（如果训练）
# model = torch.utils.checkpoint.checkpoint_sequential(model, ...)

# 5. 定期清理内存
def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# 在每个实验后调用
cleanup_memory()

# 6. 限制数据加载器的worker数量
num_workers = 2  # 不要超过2
```
EOF

echo ""
echo "=========================================="
echo "完成"
echo "=========================================="
