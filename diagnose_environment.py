#!/usr/bin/env python3
"""
环境诊断脚本 - 不修改任何代码，只诊断
找出为什么GPU使用率为0%的真正原因
"""

import sys
import os
import subprocess
import json

def run_command(cmd):
    """安全运行命令"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        return result.stdout.strip(), result.returncode
    except Exception as e:
        return str(e), -1

def check_section(title):
    """打印章节标题"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def main():
    report = {}
    
    print("""
╔══════════════════════════════════════════════════════════════════╗
║           MemTorch GPU问题完整诊断报告                           ║
║           不修改任何代码，只分析问题                              ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # ============================================================
    # 1. Shell环境检查
    # ============================================================
    check_section("1. Shell环境检查")
    
    output, _ = run_command("echo $SHELL")
    print(f"当前Shell: {output}")
    
    output, _ = run_command("whoami")
    print(f"当前用户: {output}")
    
    output, _ = run_command("pwd")
    print(f"当前目录: {output}")
    
    # ============================================================
    # 2. Python环境检查
    # ============================================================
    check_section("2. Python环境检查")
    
    print(f"Python解释器: {sys.executable}")
    print(f"Python版本: {sys.version}")
    
    output, _ = run_command("which python")
    print(f"python路径: {output}")
    
    output, _ = run_command("which python3")
    print(f"python3路径: {output}")
    
    # 检查conda
    output, rc = run_command("conda --version")
    if rc == 0:
        print(f"✅ Conda已安装: {output}")
        
        output, _ = run_command("conda info --envs")
        print("\nConda环境列表:")
        print(output)
        
        output, _ = run_command("echo $CONDA_DEFAULT_ENV")
        if output:
            print(f"\n当前激活环境: {output}")
        else:
            print("\n⚠️  没有激活的conda环境")
    else:
        print("❌ Conda未安装或不在PATH中")
    
    report['python_version'] = sys.version
    report['python_executable'] = sys.executable
    
    # ============================================================
    # 3. PyTorch检查
    # ============================================================
    check_section("3. PyTorch检查")
    
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
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            
            # 测试CUDA功能
            try:
                x = torch.randn(10, 10).cuda()
                y = x + x
                torch.cuda.synchronize()
                print(f"   ✅ CUDA计算测试通过")
            except Exception as e:
                print(f"   ❌ CUDA计算测试失败: {e}")
        else:
            print(f"   ❌ PyTorch不支持CUDA")
            print(f"   这是CPU版本的PyTorch！")
        
        report['torch_installed'] = True
        report['torch_version'] = torch.__version__
        report['torch_cuda_available'] = cuda_available
        
    except ImportError as e:
        print(f"❌ PyTorch未安装: {e}")
        report['torch_installed'] = False
    
    # ============================================================
    # 4. CUDA工具链检查
    # ============================================================
    check_section("4. CUDA工具链检查")
    
    # 检查nvidia-smi
    output, rc = run_command("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader")
    if rc == 0:
        print(f"✅ nvidia-smi可用")
        print(f"GPU信息:\n{output}")
    else:
        print(f"❌ nvidia-smi不可用")
    
    # 检查nvcc
    output, rc = run_command("nvcc --version")
    if rc == 0:
        print(f"\n✅ nvcc可用")
        print(output)
        report['nvcc_available'] = True
    else:
        print(f"\n❌ nvcc不在PATH中")
        report['nvcc_available'] = False
        
        # 尝试查找CUDA
        print("\n尝试查找CUDA安装位置...")
        for path in ['/usr/local/cuda', '/usr/local/cuda-11.8', '/usr/local/cuda-11.7', 
                     '/opt/cuda', '/usr/lib/cuda']:
            if os.path.exists(path):
                print(f"   找到: {path}")
                nvcc_path = os.path.join(path, 'bin', 'nvcc')
                if os.path.exists(nvcc_path):
                    output, _ = run_command(f"{nvcc_path} --version")
                    print(f"   nvcc版本: {output}")
    
    # ============================================================
    # 5. MemTorch检查（关键！）
    # ============================================================
    check_section("5. MemTorch安装检查（关键！）")
    
    try:
        import memtorch
        print(f"✅ MemTorch已安装")
        print(f"   版本: {memtorch.__version__}")
        print(f"   安装路径: {memtorch.__file__}")
        
        # 关键检查：是否是CPU版本
        is_cpu_version = '-cpu' in memtorch.__version__
        if is_cpu_version:
            print(f"\n   ❌❌❌ 这是CPU版本！ ❌❌❌")
            print(f"   这就是GPU使用率为0%的原因！")
        else:
            print(f"\n   ✅ 这是GPU版本")
        
        report['memtorch_installed'] = True
        report['memtorch_version'] = memtorch.__version__
        report['memtorch_is_cpu'] = is_cpu_version
        
        # 检查CUDA绑定
        try:
            import memtorch_cuda_bindings
            print(f"   ✅ CUDA绑定已加载")
            report['cuda_bindings_available'] = True
        except ImportError:
            print(f"   ❌ CUDA绑定未找到")
            print(f"   MemTorch是在没有CUDA支持下编译的")
            report['cuda_bindings_available'] = False
        
        # 检查CPU绑定
        try:
            import memtorch_bindings
            print(f"   ✅ CPU绑定已加载")
        except ImportError:
            print(f"   ❌ CPU绑定未找到")
        
    except ImportError as e:
        print(f"❌ MemTorch未安装: {e}")
        report['memtorch_installed'] = False
    
    # ============================================================
    # 6. setup.py配置检查
    # ============================================================
    check_section("6. setup.py配置检查")
    
    setup_path = '/workspace/setup.py'
    if os.path.exists(setup_path):
        print(f"✅ setup.py存在: {setup_path}")
        
        with open(setup_path, 'r') as f:
            content = f.read()
            
        # 查找CUDA配置
        import re
        match = re.search(r'CUDA\s*=\s*(True|False)', content)
        if match:
            cuda_enabled = match.group(1) == 'True'
            print(f"\n   setup.py中的CUDA配置:")
            print(f"   CUDA = {match.group(1)}")
            
            if cuda_enabled:
                print(f"   ✅ CUDA已启用")
            else:
                print(f"   ❌❌❌ CUDA被禁用！ ❌❌❌")
                print(f"   这是MemTorch编译为CPU版本的原因！")
            
            report['setup_cuda_enabled'] = cuda_enabled
        else:
            print(f"   ⚠️  未找到CUDA配置")
    else:
        print(f"❌ setup.py不存在: {setup_path}")
    
    # ============================================================
    # 7. 编译产物检查
    # ============================================================
    check_section("7. 编译产物检查")
    
    # 检查build目录
    build_path = '/workspace/build'
    if os.path.exists(build_path):
        print(f"✅ build目录存在")
        output, _ = run_command(f"find {build_path} -name '*.so' | head -5")
        if output:
            print(f"   找到的.so文件:\n{output}")
        else:
            print(f"   ⚠️  未找到.so文件")
    else:
        print(f"❌ build目录不存在（可能未编译）")
    
    # 检查egg-info
    output, _ = run_command("find /workspace -name 'memtorch*.egg-info' -type d")
    if output:
        print(f"\n✅ 找到egg-info: {output}")
    
    # ============================================================
    # 8. pip安装信息
    # ============================================================
    check_section("8. pip安装信息")
    
    output, rc = run_command("pip show memtorch")
    if rc == 0:
        print(output)
    else:
        print("❌ pip show memtorch 失败")
    
    output, _ = run_command("pip list | grep -i torch")
    print(f"\n已安装的torch相关包:")
    print(output if output else "未找到")
    
    # ============================================================
    # 9. 生成诊断报告
    # ============================================================
    check_section("9. 问题诊断总结")
    
    print("\n🔍 问题分析:")
    print("-" * 70)
    
    issues_found = []
    
    # 检查1: PyTorch CUDA
    if not report.get('torch_installed'):
        issues_found.append("❌ PyTorch未安装")
    elif not report.get('torch_cuda_available'):
        issues_found.append("❌ PyTorch不支持CUDA（安装的是CPU版本）")
    
    # 检查2: MemTorch版本
    if not report.get('memtorch_installed'):
        issues_found.append("❌ MemTorch未安装")
    elif report.get('memtorch_is_cpu'):
        issues_found.append("❌❌❌ MemTorch是CPU版本（这是主要问题！）")
    
    # 检查3: setup.py配置
    if report.get('setup_cuda_enabled') == False:
        issues_found.append("❌ setup.py中CUDA=False（导致编译为CPU版本）")
    
    # 检查4: CUDA工具链
    if not report.get('nvcc_available'):
        issues_found.append("⚠️  nvcc不在PATH中（编译GPU版本需要）")
    
    # 检查5: CUDA绑定
    if report.get('memtorch_installed') and not report.get('cuda_bindings_available'):
        issues_found.append("❌ MemTorch CUDA绑定不存在")
    
    if issues_found:
        print("\n发现的问题:")
        for issue in issues_found:
            print(f"  {issue}")
    else:
        print("\n✅ 未发现明显问题（需要更深入调查）")
    
    # 最关键的判断
    print("\n" + "="*70)
    print("🎯 GPU使用率为0%的根本原因:")
    print("="*70)
    
    if report.get('memtorch_is_cpu'):
        print("""
❌ MemTorch安装的是CPU版本！

原因链：
  1. setup.py中 CUDA = False
     ↓
  2. pip install时编译为CPU版本
     ↓
  3. memtorch.__version__ = "1.1.6-cpu"
     ↓
  4. 所有计算在CPU上运行
     ↓
  5. GPU使用率 = 0%

即使你：
  ✗ 有GPU硬件
  ✗ 安装了CUDA驱动
  ✗ PyTorch支持CUDA
  ✗ 更换到Python 3.8

如果MemTorch是CPU版本，它永远不会使用GPU！
        """)
    elif not report.get('torch_cuda_available'):
        print("""
❌ PyTorch不支持CUDA！

即使MemTorch是GPU版本，如果PyTorch本身不支持CUDA，
也无法在GPU上运行。

你安装的可能是：
  pip install torch  # CPU版本

应该安装：
  pip install torch --index-url https://download.pytorch.org/whl/cu118
        """)
    else:
        print("""
⚠️  需要更多信息来判断

当前检测结果：
  - PyTorch支持CUDA: ✅
  - MemTorch是GPU版本: ✅
  - 但仍然GPU使用率为0%

可能原因：
  1. Jupyter kernel使用了不同的Python环境
  2. 代码中使用了CPU模式（forward_legacy_enabled=True）
  3. 数据没有移到GPU上
  4. 特定的兼容性问题
        """)
    
    # ============================================================
    # 10. 建议的检查步骤
    # ============================================================
    check_section("10. 建议的下一步检查")
    
    print("""
请在Jupyter Notebook中运行以下代码并提供输出：

```python
# 检查1：环境信息
import sys
print(f"Python: {sys.executable}")
print(f"Version: {sys.version}")

# 检查2：PyTorch
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# 检查3：MemTorch版本
import memtorch
print(f"MemTorch: {memtorch.__version__}")
print("是CPU版本" if '-cpu' in memtorch.__version__ else "是GPU版本")

# 检查4：简单测试
if torch.cuda.is_available():
    x = torch.randn(10, 10).cuda()
    print(f"GPU测试: {x.device}")
```

然后告诉我输出结果。
    """)
    
    # 保存报告
    report_file = '/tmp/memtorch_diagnostic_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n完整报告已保存到: {report_file}")
    print("\n" + "="*70)
    print("诊断完成")
    print("="*70)

if __name__ == '__main__':
    main()
