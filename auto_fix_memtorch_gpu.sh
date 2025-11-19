#!/bin/bash
# MemTorch GPU 一键修复脚本（AutoDL平台）

set -e  # 遇到错误立即退出

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}"
echo "=========================================="
echo "  MemTorch GPU 一键修复脚本"
echo "  适用于AutoDL平台"
echo "=========================================="
echo -e "${NC}"

# 函数：打印步骤
step() {
    echo -e "\n${BLUE}▶ $1${NC}"
}

# 函数：检查命令是否成功
check() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ 成功${NC}"
    else
        echo -e "${RED}❌ 失败${NC}"
        exit 1
    fi
}

# 步骤0：检查是否在正确的目录
step "步骤0：检查工作目录"
if [ ! -f "setup.py" ]; then
    echo -e "${YELLOW}警告：setup.py不在当前目录${NC}"
    if [ -f "/workspace/setup.py" ]; then
        cd /workspace
        echo "切换到 /workspace"
    else
        echo -e "${RED}错误：找不到MemTorch源码目录${NC}"
        exit 1
    fi
fi
echo "当前目录: $(pwd)"

# 步骤1：检查GPU
step "步骤1：检查GPU环境"
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}❌ 错误：nvidia-smi不可用${NC}"
    echo "请确保AutoDL实例选择了GPU"
    exit 1
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
check

# 步骤2：检查CUDA
step "步骤2：检查CUDA Toolkit"
if ! command -v nvcc &> /dev/null; then
    echo -e "${YELLOW}⚠️  警告：nvcc不在PATH中，尝试添加${NC}"
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
fi

if command -v nvcc &> /dev/null; then
    nvcc --version | grep "release"
    check
else
    echo -e "${RED}❌ 错误：CUDA Toolkit未安装${NC}"
    echo "请在AutoDL镜像中选择包含CUDA的镜像"
    exit 1
fi

# 步骤3：检查PyTorch CUDA支持
step "步骤3：验证PyTorch CUDA支持"
python3 << 'EOF'
import torch
import sys

print(f"PyTorch版本: {torch.__version__}")
cuda_available = torch.cuda.is_available()
print(f"CUDA可用: {cuda_available}")

if not cuda_available:
    print("\n❌ PyTorch不支持CUDA！")
    print("解决方案：")
    print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    sys.exit(1)
else:
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    sys.exit(0)
EOF
check

# 步骤4：卸载旧版本
step "步骤4：卸载旧版本MemTorch"
pip uninstall memtorch memtorch-cpu -y 2>/dev/null || echo "未安装旧版本"

# 步骤5：安装编译依赖
step "步骤5：安装编译依赖"
pip install ninja -q
check

# 步骤6：修改setup.py启用CUDA
step "步骤6：修改setup.py启用CUDA"
if grep -q "CUDA = False" setup.py; then
    sed -i 's/CUDA = False/CUDA = True/' setup.py
    echo "已将 CUDA = False 改为 CUDA = True"
else
    echo "setup.py已经启用CUDA"
fi
check

# 步骤7：编译安装GPU版本
step "步骤7：编译安装GPU版本（需要5-10分钟）"
echo "开始编译CUDA扩展..."
pip install -e . 2>&1 | tee /tmp/memtorch_install.log

# 检查编译日志中是否有严重错误
if grep -qi "error" /tmp/memtorch_install.log | grep -v "warning"; then
    echo -e "${YELLOW}⚠️  编译过程中可能有错误，请检查日志${NC}"
    echo "日志保存在: /tmp/memtorch_install.log"
fi
check

# 步骤8：验证安装
step "步骤8：验证GPU版本安装"
python3 << 'EOF'
import sys
import torch

# 检查MemTorch
try:
    import memtorch
    version = memtorch.__version__
    print(f"MemTorch版本: {version}")
    
    if '-cpu' in version:
        print("❌ 仍然是CPU版本！")
        sys.exit(1)
    else:
        print("✅ GPU版本安装成功")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

# 测试GPU
print("\n测试GPU计算...")
try:
    x = torch.randn(1000, 1000).cuda()
    y = torch.mm(x, x)
    torch.cuda.synchronize()
    print("✅ GPU计算正常")
except Exception as e:
    print(f"❌ GPU计算失败: {e}")
    sys.exit(1)

# 测试MemTorch模块
print("\n测试MemTorch模块...")
try:
    from memtorch.bh.memristor import LinearIonDrift
    from memtorch.mn import Linear
    device = LinearIonDrift()
    print("✅ MemTorch模块加载正常")
except Exception as e:
    print(f"❌ 模块加载失败: {e}")
    sys.exit(1)

print("\n" + "="*50)
print("✅ 所有测试通过！MemTorch GPU版本已就绪")
print("="*50)
EOF
check

# 步骤9：创建优化的notebook配置
step "步骤9：创建优化配置文件"
cat > /workspace/memtorch_config.py << 'EOF'
"""
MemTorch优化配置
在Jupyter Notebook开头导入: from memtorch_config import *
"""

import gc
import torch
import os

# 环境变量优化
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# GPU信息
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"   显存: {mem_gb:.1f} GB")
    torch.cuda.empty_cache()
else:
    print("❌ GPU不可用")

# 根据显存自动调整配置
mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0

if mem_gb >= 20:  # A100, RTX 3090等
    CONFIG = {
        'batch_size': 128,
        'tile_shape': (128, 128),
        'num_workers': 4,
    }
    print("   配置: 大显存模式")
elif mem_gb >= 10:  # RTX 3080等
    CONFIG = {
        'batch_size': 64,
        'tile_shape': (64, 64),
        'num_workers': 2,
    }
    print("   配置: 中等显存模式")
else:  # 小显存
    CONFIG = {
        'batch_size': 32,
        'tile_shape': (32, 32),
        'num_workers': 1,
    }
    print("   配置: 小显存模式")

# 内存清理函数
def cleanup_memory():
    """定期调用以清理内存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
print(f"\n优化配置已加载: {CONFIG}")
print("使用 cleanup_memory() 清理内存")
EOF

echo "配置文件已创建: /workspace/memtorch_config.py"
check

# 完成
echo -e "\n${GREEN}"
echo "=========================================="
echo "  ✅ MemTorch GPU版本安装完成！"
echo "=========================================="
echo -e "${NC}"

echo -e "\n${YELLOW}下一步：${NC}"
echo "1. 重启Jupyter Notebook内核"
echo "2. 在notebook开头添加："
echo "   from memtorch_config import *"
echo "3. 运行示例代码"
echo ""
echo -e "${YELLOW}监控GPU使用：${NC}"
echo "   watch -n 1 nvidia-smi"
echo ""
echo -e "${YELLOW}查看完整指南：${NC}"
echo "   cat /workspace/autodl_memtorch_fix_guide.md"
echo ""
