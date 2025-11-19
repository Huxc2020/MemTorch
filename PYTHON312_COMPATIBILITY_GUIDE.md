# Python 3.12 与 MemTorch 兼容性指南

## 🚨 核心问题

**MemTorch在Python 3.12下存在严重的兼容性问题**，主要原因：

### 1. Python 3.12的重大变更

```
Python 3.12新特性（会影响MemTorch）：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ 新的内存管理器（更严格的内存安全检查）
✓ C API的breaking changes
✓ GIL优化（影响C扩展）
✓ 更严格的类型检查
✓ 改进的错误检测（会暴露之前的bug）
```

### 2. MemTorch的CUDA代码bug

```cpp
// ❌ 这在Python 3.12下必定崩溃
Eigen::VectorXf partial_sum = ...;
free(&partial_sum);  // 尝试释放栈内存！
```

Python 3.12的新内存管理器立即检测到这个错误并导致崩溃。

---

## ✅ 解决方案（3个选项）

### 选项A：修复代码（推荐，如果你需要Python 3.12）

```bash
# 1. 运行自动修复脚本
cd /workspace
python3 apply_cuda_fixes.py

# 2. 重新编译
pip uninstall memtorch -y
pip install -e .

# 3. 测试
python3 -c "import memtorch; print('OK')"
```

**优点：** 保持Python 3.12
**缺点：** 需要重新编译，可能还有其他未知问题

---

### 选项B：降级到Python 3.11（最稳定）

#### AutoDL平台操作：

```bash
# 1. 使用conda创建Python 3.11环境
conda create -n memtorch_py311 python=3.11 -y
conda activate memtorch_py311

# 2. 安装依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas scipy scikit-learn matplotlib seaborn ipython lmfit ninja

# 3. 安装MemTorch
cd /workspace
sed -i 's/CUDA = False/CUDA = True/' setup.py
sed -i 's/"sklearn"/"scikit-learn"/' setup.py
pip install -e .

# 4. 注册Jupyter kernel
pip install ipykernel
python3 -m ipykernel install --user --name memtorch_py311 --display-name "MemTorch (Py3.11)"

# 5. 在Jupyter中选择 "MemTorch (Py3.11)" kernel
```

**优点：** 
- ✅ 最稳定
- ✅ MemTorch官方测试的版本
- ✅ 无需修改代码

**缺点：** 
- 需要管理多个Python环境

---

### 选项C：使用Docker容器（最彻底）

```dockerfile
# Dockerfile for MemTorch
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# 安装Python 3.11
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3-pip \
    git ninja-build

# 安装MemTorch
RUN git clone --recursive https://github.com/coreylammie/MemTorch /opt/memtorch
WORKDIR /opt/memtorch
RUN sed -i 's/CUDA = False/CUDA = True/' setup.py
RUN pip install -e .

CMD ["/bin/bash"]
```

---

## 🔬 Python版本兼容性矩阵

| Python版本 | MemTorch兼容性 | 推荐度 | 备注 |
|-----------|---------------|--------|------|
| 3.6-3.8   | ✅ 完美 | ⭐⭐⭐ | 官方测试版本 |
| 3.9       | ✅ 完美 | ⭐⭐⭐⭐⭐ | 最推荐 |
| 3.10      | ✅ 良好 | ⭐⭐⭐⭐ | 稳定 |
| 3.11      | ✅ 良好 | ⭐⭐⭐⭐ | 推荐 |
| 3.12      | ⚠️ 有bug | ⭐⭐ | 需要修复代码 |
| 3.13+     | ❌ 未知 | ⭐ | 不推荐 |

---

## 🎯 快速决策树

```
你需要Python 3.12的特定功能吗？
    ├─ 是 → 选项A（修复代码）
    └─ 否 ↓
    
你在AutoDL或类似平台上吗？
    ├─ 是 → 选项B（conda环境）
    └─ 否 ↓
    
你有Docker权限吗？
    ├─ 是 → 选项C（Docker）
    └─ 否 → 选项A（修复代码）
```

---

## 📋 修复后的验证清单

```bash
# 1. Python版本
python3 --version
# 期望: Python 3.9-3.11 或 修复后的3.12

# 2. PyTorch CUDA
python3 -c "import torch; print(torch.cuda.is_available())"
# 期望: True

# 3. MemTorch版本
python3 -c "import memtorch; print(memtorch.__version__)"
# 期望: "1.1.6" (不是 "1.1.6-cpu")

# 4. CUDA绑定
python3 -c "import memtorch_cuda_bindings; print('OK')"
# 期望: OK

# 5. 运行小测试
python3 << 'EOF'
import torch
import memtorch
from memtorch.bh.memristor import LinearIonDrift
from memtorch.mn import Linear

# 创建小型测试
layer = torch.nn.Linear(10, 5)
mem_layer = Linear(
    layer,
    memristor_model=LinearIonDrift,
    memristor_model_params={'r_on': 100, 'r_off': 16000},
    tile_shape=(8, 8)
)

# 测试推理
x = torch.randn(2, 10).cuda()
mem_layer.forward_legacy_enabled = False
y = mem_layer(x)
print(f"✅ 测试通过！输出形状: {y.shape}")
EOF
```

---

## 🆘 如果仍然崩溃

### 调试步骤：

```bash
# 1. 启用CUDA调试
export CUDA_LAUNCH_BLOCKING=1

# 2. 启用PyTorch调试
export TORCH_SHOW_CPP_STACKTRACES=1

# 3. 运行时检查
python3 -c "
import torch
torch.cuda.init()
print(f'CUDA初始化成功')
"

# 4. 查看编译日志
cat /tmp/memtorch_install.log | grep -i error

# 5. 测试最小示例
python3 /workspace/minimal_test.py
```

### 常见错误和解决方案：

**错误1：Kernel崩溃但没有具体错误**
```
原因：内存损坏
解决：确保运行了 apply_cuda_fixes.py
```

**错误2：ImportError: undefined symbol**
```
原因：编译不完整或版本不匹配
解决：
  pip uninstall memtorch -y
  rm -rf build/ *.egg-info
  pip install -e .
```

**错误3：RuntimeError: CUDA out of memory**
```
原因：显存不足
解决：减小 batch_size 和 tile_shape
```

---

## 📚 推荐的完整配置（AutoDL）

```bash
#!/bin/bash
# memtorch_setup.sh - 完整的AutoDL设置脚本

# 1. 创建Python 3.11环境
conda create -n memtorch python=3.11 -y
conda activate memtorch

# 2. 安装CUDA工具链（如果需要）
conda install -c nvidia cuda-toolkit=11.8 -y

# 3. 安装PyTorch
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# 4. 安装依赖
pip install numpy pandas scipy scikit-learn matplotlib seaborn ipython lmfit ninja

# 5. 修复并安装MemTorch
cd /workspace
python3 apply_cuda_fixes.py
sed -i 's/CUDA = False/CUDA = True/' setup.py
pip install -e .

# 6. 配置Jupyter
pip install jupyter ipykernel
python3 -m ipykernel install --user --name memtorch --display-name "MemTorch"

# 7. 验证
python3 << 'EOF'
import torch
import memtorch
assert torch.cuda.is_available(), "CUDA不可用"
assert not '-cpu' in memtorch.__version__, "是CPU版本"
print("✅ 环境配置成功！")
print(f"   Python: {sys.version}")
print(f"   PyTorch: {torch.__version__}")
print(f"   MemTorch: {memtorch.__version__}")
print(f"   GPU: {torch.cuda.get_device_name(0)}")
EOF

echo "完成！现在可以启动Jupyter并选择 'MemTorch' kernel"
```

保存为 `memtorch_setup.sh` 并运行：
```bash
chmod +x memtorch_setup.sh
./memtorch_setup.sh
```

---

## 💡 最终建议

**对于你的情况（Python 3.12 + CUDA 13.0）：**

### 推荐方案：降级到Python 3.11

理由：
1. ✅ MemTorch在3.11上经过充分测试
2. ✅ 无需修改源代码
3. ✅ 避免未知的兼容性问题
4. ✅ AutoDL支持conda环境管理

### 如果必须用Python 3.12：

1. 运行 `python3 apply_cuda_fixes.py` 修复内存bug
2. 重新编译安装
3. 仔细测试所有功能
4. 准备应对可能的其他问题

---

祝你成功！🚀
