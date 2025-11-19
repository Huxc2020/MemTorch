"""
在Jupyter Notebook中运行此代码来诊断真实的运行环境
复制整个文件内容到Jupyter cell中执行
"""

import sys
import os
import subprocess

print("="*70)
print("MemTorch GPU问题诊断 - Jupyter环境版本")
print("="*70)

# ============================================================
# 1. Jupyter使用的Python环境
# ============================================================
print("\n【1. Jupyter Python环境】")
print(f"Python可执行文件: {sys.executable}")
print(f"Python版本: {sys.version}")
print(f"Python路径: {sys.path[0]}")

# ============================================================
# 2. PyTorch检查
# ============================================================
print("\n【2. PyTorch状态】")
try:
    import torch
    print(f"✅ PyTorch已安装")
    print(f"   版本: {torch.__version__}")
    print(f"   安装路径: {torch.__file__}")
    
    cuda_available = torch.cuda.is_available()
    print(f"   CUDA可用: {cuda_available}")
    
    if cuda_available:
        print(f"   CUDA版本: {torch.version.cuda}")
        print(f"   cuDNN版本: {torch.backends.cudnn.version()}")
        print(f"   GPU数量: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"   GPU {i}:")
            print(f"     名称: {torch.cuda.get_device_name(i)}")
            print(f"     显存: {props.total_memory / 1e9:.2f} GB")
            print(f"     计算能力: {props.major}.{props.minor}")
        
        # 测试GPU
        try:
            print(f"\n   测试GPU计算...")
            x = torch.randn(1000, 1000).cuda()
            y = torch.mm(x, x)
            torch.cuda.synchronize()
            print(f"   ✅ GPU计算测试通过")
            
            # 显存使用
            allocated = torch.cuda.memory_allocated(0) / 1e9
            reserved = torch.cuda.memory_reserved(0) / 1e9
            print(f"   显存使用: 已分配{allocated:.2f}GB, 已保留{reserved:.2f}GB")
        except Exception as e:
            print(f"   ❌ GPU计算失败: {e}")
    else:
        print(f"\n   ❌ PyTorch不支持CUDA！")
        print(f"   这是CPU版本的PyTorch")
        print(f"   安装命令应该是:")
        print(f"     pip install torch --index-url https://download.pytorch.org/whl/cu118")
        
except ImportError as e:
    print(f"❌ PyTorch未安装: {e}")

# ============================================================
# 3. MemTorch检查（最关键！）
# ============================================================
print("\n【3. MemTorch状态（最关键！）】")
try:
    import memtorch
    print(f"✅ MemTorch已安装")
    version = memtorch.__version__
    print(f"   版本: {version}")
    print(f"   安装路径: {memtorch.__file__}")
    
    # 关键判断
    is_cpu_version = '-cpu' in version
    print(f"\n   版本类型: ", end="")
    if is_cpu_version:
        print(f"❌❌❌ CPU版本 ❌❌❌")
        print(f"\n   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"   🎯 这就是GPU使用率为0%的原因！")
        print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"\n   MemTorch被编译为CPU版本，永远不会使用GPU")
        print(f"   即使你有GPU硬件和CUDA驱动")
    else:
        print(f"✅ GPU版本")
    
    # 检查绑定
    print(f"\n   检查C++/CUDA绑定:")
    try:
        import memtorch_bindings
        print(f"   ✅ CPU绑定 (memtorch_bindings) 存在")
    except ImportError:
        print(f"   ❌ CPU绑定不存在")
    
    try:
        import memtorch_cuda_bindings
        print(f"   ✅ CUDA绑定 (memtorch_cuda_bindings) 存在")
    except ImportError:
        print(f"   ❌ CUDA绑定不存在（说明是CPU版本）")
    
    # 检查实际运行设备
    print(f"\n   MemTorch运行设备:")
    device_type = "cpu" if "cpu" in version else "cuda"
    print(f"   {device_type.upper()}")
    
except ImportError as e:
    print(f"❌ MemTorch未安装: {e}")

# ============================================================
# 4. 环境变量检查
# ============================================================
print("\n【4. 相关环境变量】")
env_vars = ['CUDA_HOME', 'CUDA_PATH', 'LD_LIBRARY_PATH', 'PATH']
for var in env_vars:
    value = os.environ.get(var, '未设置')
    if var in ['LD_LIBRARY_PATH', 'PATH']:
        # 只显示CUDA相关的路径
        if 'cuda' in value.lower():
            cuda_paths = [p for p in value.split(':') if 'cuda' in p.lower()]
            if cuda_paths:
                print(f"{var} (CUDA相关):")
                for p in cuda_paths[:3]:  # 只显示前3个
                    print(f"  - {p}")
    else:
        print(f"{var}: {value}")

# ============================================================
# 5. 已安装的包
# ============================================================
print("\n【5. 相关已安装包】")
try:
    import pkg_resources
    packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
    
    relevant = ['torch', 'torchvision', 'memtorch', 'memtorch-cpu', 
                'numpy', 'cuda-python', 'nvidia-cuda-runtime']
    
    for pkg in relevant:
        if pkg in packages:
            print(f"  {pkg}: {packages[pkg]}")
except:
    pass

# ============================================================
# 6. 测试MemTorch功能
# ============================================================
print("\n【6. MemTorch功能测试】")
try:
    import torch
    import memtorch
    from memtorch.bh.memristor import LinearIonDrift
    from memtorch.mn import Linear
    
    print("尝试创建忆阻器层...")
    
    # 创建简单测试
    layer = torch.nn.Linear(4, 2)
    mem_layer = Linear(
        layer,
        memristor_model=LinearIonDrift,
        memristor_model_params={'r_on': 100, 'r_off': 16000},
        tile_shape=(2, 2),
        verbose=False
    )
    
    print("✅ 忆阻器层创建成功")
    
    # 测试前向传播
    print("\n测试前向传播:")
    
    # CPU模式
    x_cpu = torch.randn(2, 4)
    mem_layer.forward_legacy_enabled = True
    y_cpu = mem_layer(x_cpu)
    print(f"  CPU模式: ✅ 输入{x_cpu.shape} -> 输出{y_cpu.shape}")
    
    # GPU模式（如果可用）
    if torch.cuda.is_available() and '-cpu' not in memtorch.__version__:
        print("\n  尝试GPU模式...")
        x_gpu = torch.randn(2, 4).cuda()
        mem_layer.forward_legacy_enabled = False
        try:
            y_gpu = mem_layer(x_gpu)
            print(f"  GPU模式: ✅ 输入{x_gpu.shape} -> 输出{y_gpu.shape}")
        except Exception as e:
            print(f"  GPU模式: ❌ 失败 - {e}")
    else:
        if not torch.cuda.is_available():
            print(f"  ⏭️  跳过GPU测试（PyTorch不支持CUDA）")
        else:
            print(f"  ⏭️  跳过GPU测试（MemTorch是CPU版本）")
    
except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()

# ============================================================
# 7. 总结和建议
# ============================================================
print("\n" + "="*70)
print("🎯 诊断结论")
print("="*70)

try:
    import torch
    import memtorch
    
    torch_cuda = torch.cuda.is_available()
    memtorch_cpu = '-cpu' in memtorch.__version__
    
    if memtorch_cpu:
        print("""
╔══════════════════════════════════════════════════════════════════╗
║  问题确认：MemTorch是CPU版本                                     ║
╔══════════════════════════════════════════════════════════════════╝

根本原因：
  MemTorch版本带有 '-cpu' 后缀，这意味着它是在
  CUDA=False的情况下编译的。

  无论你的GPU硬件多么强大，无论CUDA驱动是否正确安装，
  CPU版本的MemTorch永远不会使用GPU。

症状：
  ✗ GPU使用率 = 0%
  ✗ CPU使用率 = 100%
  ✗ 内存持续上升
  ✗ 运行缓慢

解决方案：
  必须重新编译MemTorch为GPU版本。

即使你更换到Python 3.8 + CUDA 11.8，如果MemTorch仍然是
之前安装的CPU版本，问题会完全一样。
        """)
    elif not torch_cuda:
        print("""
╔══════════════════════════════════════════════════════════════════╗
║  问题确认：PyTorch不支持CUDA                                     ║
╚══════════════════════════════════════════════════════════════════╝

根本原因：
  你安装的PyTorch是CPU版本。
  即使MemTorch是GPU版本，也无法使用GPU。

解决方案：
  安装支持CUDA的PyTorch版本。
        """)
    else:
        print("""
✅ PyTorch支持CUDA
✅ MemTorch是GPU版本

如果仍然GPU使用率为0%，可能的原因：
  1. 代码中使用了CPU模式
  2. 数据没有移到GPU
  3. Jupyter kernel缓存问题
  4. 特定的运行时错误
        """)
        
except:
    print("无法完成诊断（缺少必要的包）")

print("\n" + "="*70)
print("诊断完成")
print("="*70)
print("\n请将上述完整输出发送给我，我会根据实际情况给出准确的解决方案。")
