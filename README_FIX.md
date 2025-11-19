# 🚨 MemTorch Kernel崩溃修复指南

## ⚡ 快速开始（3步）

### 如果你想要最快最稳定的解决方案：

```bash
# 方案A：降级到Python 3.11（推荐⭐⭐⭐⭐⭐）
conda create -n memtorch_py311 python=3.11 -y
conda activate memtorch_py311
cd /workspace
bash autodl_memtorch_fix_guide.md  # 按照指南操作
```

### 如果你必须使用Python 3.12：

```bash
# 方案B：修复代码并重新编译
cd /workspace
python3 apply_cuda_fixes.py  # 自动修复bug
pip uninstall memtorch -y
pip install -e .
python3 minimal_test.py  # 验证修复
```

---

## 📚 完整文档索引

| 文件 | 用途 | 推荐阅读顺序 |
|------|------|-------------|
| **SOLUTION_SUMMARY.md** | 完整解决方案总结 | ① 首先阅读 |
| **CRITICAL_BUGS_ANALYSIS.md** | 技术bug分析 | ② 理解问题 |
| **PYTHON312_COMPATIBILITY_GUIDE.md** | Python 3.12兼容性 | ③ 深入了解 |
| **apply_cuda_fixes.py** | 自动修复脚本 | ④ 执行修复 |
| **minimal_test.py** | 验证测试脚本 | ⑤ 验证结果 |
| **autodl_memtorch_fix_guide.md** | AutoDL平台指南 | ⑥ 平台特定 |

---

## 🎯 问题诊断

### 你的症状：
- ✅ GPU使用率 = 0%
- ✅ CPU使用率 = 100%
- ✅ 内存持续上升
- ✅ Kernel crashed

### 根本原因：
```cpp
// memtorch/cu/tile_matmul_kernels.cu 第48和84行
Eigen::VectorXf partial_sum = ...;
free(&partial_sum);  // ❌ 严重错误！试图释放栈内存
```

### 为什么在Python 3.12崩溃：
- Python 3.12有更严格的内存管理
- 立即检测到内存损坏并崩溃
- 旧版本更"宽容"，bug没有立即暴露

---

## 🔧 修复选项

### 选项1：降级Python（最简单）⭐⭐⭐⭐⭐

**优点：**
- 无需修改代码
- 最稳定
- MemTorch官方支持

**步骤：**
```bash
conda create -n memtorch_py311 python=3.11 -y
conda activate memtorch_py311
# 然后按照 autodl_memtorch_fix_guide.md 操作
```

**耗时：** 15分钟

---

### 选项2：修复代码（如果必须用Python 3.12）⭐⭐⭐

**优点：**
- 保持Python 3.12
- 修复根本bug

**步骤：**
```bash
cd /workspace
python3 apply_cuda_fixes.py
pip uninstall memtorch -y
pip install -e .
python3 minimal_test.py
```

**耗时：** 10分钟（+ 编译时间）

---

### 选项3：优化配置（临时方案）⭐⭐

如果上述方案都不可行，至少优化配置以减少崩溃几率：

```python
# 在Jupyter开头
CONFIG = {
    'batch_size': 32,       # 减小
    'tile_shape': (32, 32), # 减小
    'num_workers': 1,       # 减少
}
```

---

## ✅ 验证方法

### 快速测试：
```bash
python3 /workspace/minimal_test.py
```

### 详细检查：
```python
import torch
import memtorch

# 1. 版本检查
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"MemTorch: {memtorch.__version__}")  # 不应有 "-cpu"

# 2. CUDA检查
assert torch.cuda.is_available()
print(f"GPU: {torch.cuda.get_device_name(0)}")

# 3. 功能测试
from memtorch.mn import Linear
from memtorch.bh.memristor import LinearIonDrift

layer = torch.nn.Linear(10, 5)
mem_layer = Linear(layer, memristor_model=LinearIonDrift,
                   memristor_model_params={'r_on': 100, 'r_off': 16000})

x = torch.randn(2, 10).cuda()
y = mem_layer(x)
print(f"✅ 测试通过!")
```

---

## 🆘 故障排除

### 问题：修复后仍崩溃

```bash
# 1. 确认修复已应用
grep "free(&partial_sum)" /workspace/memtorch/cu/tile_matmul_kernels.cu
# 应该没有输出（或被注释）

# 2. 清理重新编译
cd /workspace
pip uninstall memtorch -y
rm -rf build/ dist/ *.egg-info
pip install -e .

# 3. 检查编译错误
cat /tmp/install.log | grep -i error
```

### 问题：GPU使用率仍为0%

```bash
# 检查是否是CPU版本
python3 -c "import memtorch; print(memtorch.__version__)"
# 如果显示 "1.1.6-cpu"，说明编译错误

# 检查setup.py
grep "CUDA = " /workspace/setup.py
# 应该是 "CUDA = True"
```

### 问题：内存仍持续增长

```python
# 添加内存清理
import gc
import torch

def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# 每个实验后调用
cleanup()
```

---

## 📖 详细文档

### 查看完整解决方案：
```bash
cat /workspace/SOLUTION_SUMMARY.md
```

### 查看bug技术分析：
```bash
cat /workspace/CRITICAL_BUGS_ANALYSIS.md
```

### 查看Python 3.12兼容性：
```bash
cat /workspace/PYTHON312_COMPATIBILITY_GUIDE.md
```

### 查看AutoDL平台指南：
```bash
cat /workspace/autodl_memtorch_fix_guide.md
```

---

## 📊 预期结果

修复成功后，你应该看到：

| 指标 | 修复前 | 修复后 |
|-----|--------|--------|
| GPU使用率 | 0% ❌ | 60-90% ✅ |
| GPU显存 | 闲置8GB | 使用6-12GB ✅ |
| CPU使用率 | 100% ❌ | 20-30% ✅ |
| 内存占用 | 持续上升 ❌ | 稳定 ✅ |
| Kernel状态 | 崩溃 ❌ | 正常 ✅ |
| 运行速度 | 极慢 ❌ | 快50-100倍 ✅ |

---

## 💡 建议

### 对于新手：
→ **使用选项1（降级Python）**，最简单最稳定

### 对于高级用户：
→ 使用选项2（修复代码），了解底层机制

### 对于紧急情况：
→ 先用选项3（优化配置）暂时运行，然后找时间做选项1或2

---

## 🎓 学到的经验

1. **Python 3.12是很新的版本**，不是所有库都完全支持
2. **未定义行为在新版本中会暴露**，旧代码的bug可能突然出现
3. **CUDA编程需要严格的内存管理**，栈和堆不能混淆
4. **大规模仿真会触发罕见代码路径**，暴露隐藏的bug

---

## 🚀 现在开始

```bash
# 如果你已经决定了方案，立即执行：

# 方案1（推荐）
conda create -n memtorch_py311 python=3.11 -y
conda activate memtorch_py311

# 方案2
python3 /workspace/apply_cuda_fixes.py

# 然后
python3 /workspace/minimal_test.py
```

---

**祝你成功！** 🎉

如有问题，请查看详细文档或在MemTorch GitHub提交issue。

*创建日期: 2024*
*适用于: MemTorch 1.1.6, Python 3.12.3, CUDA 13.0, AutoDL平台*
