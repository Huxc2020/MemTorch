# 🚨 AutoDL平台MemTorch GPU问题修复

## 📋 问题现象

- ❌ GPU使用率 = 0%
- ❌ CPU使用率 = 100%
- ❌ 内存持续上升
- ❌ Kernel crashed

## 🎯 根本原因

**MemTorch安装的是CPU版本，没有使用GPU！**

## ⚡ 快速修复（3步）

### 方法1：自动修复（推荐）

```bash
# 在AutoDL终端运行
cd /workspace
bash auto_fix_memtorch_gpu.sh
```

等待5-10分钟完成编译，然后**重启Jupyter内核**。

---

### 方法2：手动修复

```bash
# 1. 卸载CPU版本
pip uninstall memtorch memtorch-cpu -y

# 2. 启用CUDA
cd /workspace
sed -i 's/CUDA = False/CUDA = True/' setup.py

# 3. 安装依赖
pip install ninja

# 4. 重新安装GPU版本
pip install -e .

# 5. 验证
python3 -c "import memtorch; print(memtorch.__version__)"
# 应该输出 "1.1.6" 而不是 "1.1.6-cpu"
```

---

## ✅ 验证安装

```bash
python3 /workspace/check_gpu_setup.py
```

如果看到所有"✅"标记，说明安装成功！

---

## 📝 在Jupyter Notebook中使用

在notebook**最开头**添加：

```python
# 导入优化配置
from memtorch_config import *

# 或手动配置
import torch
import gc

# 确认GPU可用
assert torch.cuda.is_available(), "GPU不可用！"
print(f"✅ GPU: {torch.cuda.get_device_name(0)}")

# 内存清理函数
def cleanup_memory():
    gc.collect()
    torch.cuda.empty_cache()

# 优化配置（根据显存大小调整）
CONFIG = {
    'batch_size': 64,        # 8GB显存用32，16GB+用64
    'tile_shape': (64, 64),  # 显存小用(32,32)
    'num_workers': 2,
}
```

---

## 🎮 监控GPU使用

打开新终端运行：

```bash
watch -n 1 nvidia-smi
```

正常情况应该看到：
- ✅ GPU使用率 > 50%
- ✅ 显存稳定在 6-12GB

---

## 🐛 如果仍然崩溃

### 减小内存占用

修改Exemplar_Simulations.ipynb中的参数：

```python
# 数据加载器
train_loader = DataLoader(
    trainset,
    batch_size=32,    # ← 减小批次
    num_workers=2,    # ← 减少线程
)

# 交叉阵列
tile_shape = (32, 32)  # ← 减小瓦片大小

# 每个实验后清理
del model
cleanup_memory()
```

### 只运行部分实验

```python
# 只运行Figure 1A，跳过其他
experiments = ['1A']  # 不运行1E等耗内存的实验
```

---

## 📚 完整文档

- 详细指南: `/workspace/autodl_memtorch_fix_guide.md`
- 诊断脚本: `/workspace/check_gpu_setup.py`
- 自动修复: `/workspace/auto_fix_memtorch_gpu.sh`

---

## 🆘 获取帮助

如果问题仍未解决：

1. **查看安装日志**
   ```bash
   cat /tmp/memtorch_install.log
   ```

2. **检查AutoDL实例配置**
   - 确认选择了GPU实例
   - 推荐：RTX 3090 (24GB) 或更高

3. **提交Issue**
   - https://github.com/coreylammie/MemTorch/issues

---

## ✨ 成功标志

安装成功后，运行示例时应该看到：

```
✅ MemTorch版本: 1.1.6 (没有-cpu)
✅ GPU使用率: 60-95%
✅ 显存: 稳定在6-12GB
✅ CPU使用率: <30%
✅ 运行速度: 比之前快50-100倍
```

祝你好运！🚀
