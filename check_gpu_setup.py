#!/usr/bin/env python3
"""
MemTorch GPU环境快速检查脚本
使用方法: python3 check_gpu_setup.py
"""

import sys

def print_section(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def check_pytorch():
    """检查PyTorch和CUDA"""
    print_section("1. PyTorch & CUDA检查")
    
    try:
        import torch
        print(f"✅ PyTorch已安装")
        print(f"   版本: {torch.__version__}")
        
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"✅ CUDA可用")
            print(f"   CUDA版本: {torch.version.cuda}")
            print(f"   GPU数量: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                props = torch.cuda.get_device_properties(i)
                total_mem = props.total_memory / 1e9
                print(f"   GPU {i}: {name} ({total_mem:.1f} GB)")
            
            return True
        else:
            print("❌ CUDA不可用")
            print("\n解决方案:")
            print("  1. 确保AutoDL选择了GPU实例")
            print("  2. 重新安装支持CUDA的PyTorch:")
            print("     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
            return False
            
    except ImportError:
        print("❌ PyTorch未安装")
        print("\n安装命令:")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        return False

def check_memtorch():
    """检查MemTorch"""
    print_section("2. MemTorch检查")
    
    try:
        import memtorch
        version = memtorch.__version__
        print(f"✅ MemTorch已安装")
        print(f"   版本: {version}")
        
        if '-cpu' in version:
            print("❌ 这是CPU版本！")
            print("\n问题: 当前安装的是CPU版本，无法使用GPU加速")
            print("\n修复方案:")
            print("  方法1 - 运行自动修复脚本:")
            print("    bash /workspace/auto_fix_memtorch_gpu.sh")
            print("")
            print("  方法2 - 手动修复:")
            print("    cd /workspace")
            print("    pip uninstall memtorch memtorch-cpu -y")
            print("    sed -i 's/CUDA = False/CUDA = True/' setup.py")
            print("    pip install ninja")
            print("    pip install -e .")
            return False
        else:
            print("✅ 这是GPU版本")
            return True
            
    except ImportError:
        print("❌ MemTorch未安装")
        print("\n安装命令:")
        print("  cd /workspace")
        print("  sed -i 's/CUDA = False/CUDA = True/' setup.py")
        print("  pip install ninja")
        print("  pip install -e .")
        return False

def check_gpu_compute():
    """测试GPU计算"""
    print_section("3. GPU计算测试")
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("⏭️  跳过（CUDA不可用）")
            return False
        
        print("测试矩阵乘法...")
        x = torch.randn(1000, 1000).cuda()
        y = torch.mm(x, x)
        torch.cuda.synchronize()
        
        print("✅ GPU计算正常")
        return True
        
    except Exception as e:
        print(f"❌ GPU计算失败: {e}")
        return False

def check_memtorch_modules():
    """测试MemTorch模块"""
    print_section("4. MemTorch模块测试")
    
    try:
        from memtorch.bh.memristor import LinearIonDrift
        print("✅ 忆阻器模型模块正常")
        
        from memtorch.mn import Linear
        print("✅ 神经网络层模块正常")
        
        from memtorch.map import Parameter
        print("✅ 参数映射模块正常")
        
        # 检查实际运行设备
        import memtorch
        device_type = "cpu" if "cpu" in memtorch.__version__ else "cuda"
        print(f"\n   MemTorch将在 {device_type.upper()} 上运行")
        
        return True
        
    except Exception as e:
        print(f"❌ 模块加载失败: {e}")
        return False

def check_memory():
    """检查内存状态"""
    print_section("5. 内存状态")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                mem_allocated = torch.cuda.memory_allocated(i) / 1e9
                mem_reserved = torch.cuda.memory_reserved(i) / 1e9
                mem_total = torch.cuda.get_device_properties(i).total_memory / 1e9
                mem_free = mem_total - mem_reserved
                
                print(f"GPU {i}:")
                print(f"   总显存: {mem_total:.2f} GB")
                print(f"   已分配: {mem_allocated:.2f} GB")
                print(f"   已保留: {mem_reserved:.2f} GB")
                print(f"   可用: {mem_free:.2f} GB")
                
                if mem_free < 2.0:
                    print(f"   ⚠️  可用显存较少，可能需要清理")
                    print(f"      在Python中运行: torch.cuda.empty_cache()")
        
        # 系统内存
        try:
            import psutil
            mem = psutil.virtual_memory()
            print(f"\n系统内存:")
            print(f"   总内存: {mem.total / 1e9:.1f} GB")
            print(f"   已使用: {mem.used / 1e9:.1f} GB ({mem.percent}%)")
            print(f"   可用: {mem.available / 1e9:.1f} GB")
            
            if mem.percent > 90:
                print(f"   ⚠️  内存使用率过高")
        except ImportError:
            print("\n系统内存: (需要安装psutil查看)")
            
        return True
        
    except Exception as e:
        print(f"检查时出错: {e}")
        return False

def print_recommendations(results):
    """打印建议"""
    print_section("总结与建议")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("✅ 所有检查通过！MemTorch GPU环境已就绪")
        print("\n你可以:")
        print("  1. 启动Jupyter Notebook")
        print("  2. 运行示例: memtorch/examples/Exemplar_Simulations.ipynb")
        print("  3. 在notebook开头添加:")
        print("     from memtorch_config import *")
        print("\n监控GPU:")
        print("  watch -n 1 nvidia-smi")
        
    else:
        print("⚠️  发现问题，需要修复")
        
        if not results['pytorch']:
            print("\n🔧 修复PyTorch:")
            print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        
        if not results['memtorch']:
            print("\n🔧 修复MemTorch:")
            print("  bash /workspace/auto_fix_memtorch_gpu.sh")
            print("  或查看详细指南:")
            print("  cat /workspace/autodl_memtorch_fix_guide.md")
    
    print("\n" + "="*60)

def main():
    """主函数"""
    print("""
╔════════════════════════════════════════════════════════════╗
║           MemTorch GPU环境诊断工具                         ║
║           适用于AutoDL平台                                 ║
╚════════════════════════════════════════════════════════════╝
    """)
    
    results = {
        'pytorch': check_pytorch(),
        'memtorch': check_memtorch(),
        'gpu_compute': check_gpu_compute(),
        'modules': check_memtorch_modules(),
        'memory': check_memory(),
    }
    
    print_recommendations(results)
    
    # 返回状态码
    if all(results.values()):
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == '__main__':
    main()
