# AutoDL平台运行MemTorch GPU版本完整指南

## 🔍 问题诊断

你遇到的问题症状：
- ✅ GPU使用率 = 0%
- ✅ GPU显存占用 = 8GB（PyTorch占用但未使用）
- ✅ CPU使用率 = 100%
- ✅ 内存持续上升
- ✅ Kernel crashed (内存溢出)

**根本原因：MemTorch安装的是CPU版本**

---

## 🚀 快速修复（AutoDL平台）

### 方法1：重新安装GPU版本（推荐）

```bash
# 1. 打开AutoDL的终端

# 2. 卸载CPU版本
pip uninstall memtorch memtorch-cpu -y

# 3. 检查环境
nvidia-smi  # 应该看到GPU信息
python3 -c "import torch; print(torch.cuda.is_available())"  # 应该输出True

# 4. 修改setup.py
cd /workspace
sed -i 's/CUDA = False/CUDA = True/' setup.py

# 5. 安装依赖
pip install ninja  # CUDA编译需要

# 6. 重新安装（会编译CUDA扩展，需要5-10分钟）
pip install -e .

# 7. 验证
python3 -c "import memtorch; print(memtorch.__version__)"
# 应该输出 "1.1.6" 而不是 "1.1.6-cpu"
```

### 方法2：使用预编译版本（更快）

```bash
pip uninstall memtorch memtorch-cpu -y
pip install memtorch  # 不是memtorch-cpu
```

---

## 📝 修改Notebook以优化内存使用

即使安装了GPU版本，大规模仿真仍可能需要优化。在Jupyter Notebook开头添加：

```python
# ============================================
# 内存优化配置 - 在所有代码之前运行
# ============================================

import gc
import torch
import os

# 1. 设置环境变量
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# 2. 清理GPU缓存
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"✅ GPU可用: {torch.cuda.get_device_name(0)}")
    print(f"   显存总量: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("❌ GPU不可用，将在CPU上运行（非常慢！）")

# 3. 内存清理函数
def cleanup_memory():
    """定期调用以清理内存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
# 4. 优化配置（根据AutoDL显存大小调整）
CONFIG = {
    'batch_size': 64,          # 根据显存调整：8GB→32, 16GB→64, 24GB→128
    'num_workers': 2,          # 数据加载线程
    'tile_shape': (64, 64),    # 交叉阵列瓦片大小：显存小用(32,32)
    'pin_memory': True,        # 加速数据传输
}

print(f"配置: {CONFIG}")
```

---

## 🔧 针对Exemplar_Simulations.ipynb的修改

### 找到并修改这些地方：

#### 1. 数据加载器部分

**原代码：**
```python
train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=4
)
```

**修改为：**
```python
train_loader = torch.utils.data.DataLoader(
    trainset, 
    batch_size=CONFIG['batch_size'],  # 64而不是128
    shuffle=True, 
    num_workers=CONFIG['num_workers'],  # 2而不是4
    pin_memory=CONFIG['pin_memory']
)
```

#### 2. 交叉阵列配置

**原代码：**
```python
patched_model = memtorch.mn.Linear(
    model.linear,
    memristor_model=memristor_model,
    memristor_model_params=memristor_model_params,
    tile_shape=(128, 128),  # 大！
    ...
)
```

**修改为：**
```python
patched_model = memtorch.mn.Linear(
    model.linear,
    memristor_model=memristor_model,
    memristor_model_params=memristor_model_params,
    tile_shape=CONFIG['tile_shape'],  # (64, 64)
    ...
)
```

#### 3. 在每个实验后添加内存清理

```python
# 运行实验
accuracy = test(patched_model, test_loader)
print(f"Accuracy: {accuracy}%")

# 清理内存（重要！）
del patched_model
cleanup_memory()
print("内存已清理")
```

---

## 🎯 AutoDL平台特定建议

### 1. 选择合适的实例

```
最低要求：
- GPU: RTX 3090 (24GB)
- 内存: 32GB+
- 存储: 50GB+

推荐配置：
- GPU: A100 (40GB) 或 RTX 4090 (24GB)
- 内存: 64GB
```

### 2. 持久化环境

在AutoDL上每次重启会丢失环境，建议：

```bash
# 创建安装脚本 install_memtorch.sh
cat > ~/install_memtorch.sh << 'EOF'
#!/bin/bash
cd /workspace
pip uninstall memtorch memtorch-cpu -y
sed -i 's/CUDA = False/CUDA = True/' setup.py
pip install ninja
pip install -e .
EOF

chmod +x ~/install_memtorch.sh

# 每次重启后运行
~/install_memtorch.sh
```

### 3. 使用AutoDL的Jupyter配置

```bash
# 增加Jupyter的内存限制
echo "c.NotebookApp.max_buffer_size = 10737418240" >> ~/.jupyter/jupyter_notebook_config.py
```

---

## 🐛 调试脚本

运行这个脚本检查一切是否正常：

```python
# check_memtorch_gpu.py
import torch
import sys

print("=" * 50)
print("MemTorch GPU环境检查")
print("=" * 50)

# 1. PyTorch
print(f"\n1. PyTorch")
print(f"   版本: {torch.__version__}")
print(f"   CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   CUDA版本: {torch.version.cuda}")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
    mem_alloc = torch.cuda.memory_allocated(0) / 1e9
    mem_cached = torch.cuda.memory_reserved(0) / 1e9
    print(f"   显存: 总{mem_total:.1f}GB, 已分配{mem_alloc:.1f}GB, 已缓存{mem_cached:.1f}GB")

# 2. MemTorch
print(f"\n2. MemTorch")
try:
    import memtorch
    version = memtorch.__version__
    print(f"   版本: {version}")
    if '-cpu' in version:
        print(f"   ❌ CPU版本 - 需要重新安装GPU版本！")
        sys.exit(1)
    else:
        print(f"   ✅ GPU版本")
except ImportError:
    print(f"   ❌ 未安装")
    sys.exit(1)

# 3. 测试GPU计算
print(f"\n3. GPU计算测试")
try:
    x = torch.randn(1000, 1000).cuda()
    y = torch.mm(x, x)
    torch.cuda.synchronize()
    print(f"   ✅ GPU计算正常")
except Exception as e:
    print(f"   ❌ GPU计算失败: {e}")
    sys.exit(1)

# 4. 测试MemTorch
print(f"\n4. MemTorch功能测试")
try:
    from memtorch.bh.memristor import LinearIonDrift
    device = LinearIonDrift()
    print(f"   ✅ 忆阻器模型加载正常")
    
    # 检查设备
    import torch
    device_str = "cpu" if "cpu" in memtorch.__version__ else "cuda"
    print(f"   计算设备: {device_str}")
    
except Exception as e:
    print(f"   ❌ 功能测试失败: {e}")
    sys.exit(1)

print("\n" + "=" * 50)
print("✅ 所有检查通过！可以开始运行示例了")
print("=" * 50)
```

保存为文件并运行：
```bash
python3 check_memtorch_gpu.py
```

---

## 📊 监控运行状态

在运行Exemplar_Simulations时，打开另一个终端监控：

```bash
# 监控GPU使用
watch -n 1 nvidia-smi

# 或者更详细的监控
pip install gpustat
watch -n 1 gpustat -cpu
```

正常情况下应该看到：
- ✅ GPU使用率 > 50%
- ✅ GPU显存使用逐渐增加并稳定
- ✅ CPU使用率较低（< 30%）

---

## ⚠️ 如果仍然崩溃

如果安装GPU版本后仍然内存不足：

```python
# 极限优化配置
CONFIG = {
    'batch_size': 16,           # 最小批次
    'tile_shape': (32, 32),     # 最小瓦片
    'num_workers': 1,           # 单线程
    'test_samples': 1000,       # 只测试1000个样本
}

# 只运行部分实验
RUN_EXPERIMENTS = ['1A']  # 只运行Figure 1A，不运行1E
```

---

## 📞 获取帮助

如果问题仍未解决：

1. **检查AutoDL实例配置**
   - 确认选择了GPU实例
   - 确认显存足够（至少16GB）

2. **查看完整错误日志**
   ```bash
   # Jupyter日志位置
   cat ~/.local/share/jupyter/runtime/*.log
   ```

3. **在MemTorch GitHub提问**
   - https://github.com/coreylammie/MemTorch/issues

---

## ✅ 成功标志

正确安装和运行后，你应该看到：

```
MemTorch版本: 1.1.6 (没有-cpu后缀)
GPU使用率: 60-95%
GPU显存: 稳定在6-12GB
CPU使用率: 10-30%
运行速度: 比CPU版本快50-100倍
```

祝运行顺利！🚀
