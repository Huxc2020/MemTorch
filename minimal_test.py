#!/usr/bin/env python3
"""
最小化测试脚本
用于验证MemTorch修复是否成功
"""

import sys
import traceback

def test_imports():
    """测试基本导入"""
    print("=" * 60)
    print("测试1: 基本导入")
    print("=" * 60)
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"   CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA版本: {torch.version.cuda}")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"❌ PyTorch导入失败: {e}")
        return False
    
    try:
        import memtorch
        version = memtorch.__version__
        print(f"✅ MemTorch: {version}")
        if '-cpu' in version:
            print("   ⚠️  这是CPU版本")
            return False
    except Exception as e:
        print(f"❌ MemTorch导入失败: {e}")
        traceback.print_exc()
        return False
    
    return True

def test_memristor_model():
    """测试忆阻器模型"""
    print("\n" + "=" * 60)
    print("测试2: 忆阻器模型")
    print("=" * 60)
    
    try:
        from memtorch.bh.memristor import LinearIonDrift
        device = LinearIonDrift(r_on=100, r_off=16000)
        print(f"✅ LinearIonDrift模型创建成功")
        print(f"   r_on: {device.r_on}, r_off: {device.r_off}")
        print(f"   g: {device.g}")
        return True
    except Exception as e:
        print(f"❌ 忆阻器模型测试失败: {e}")
        traceback.print_exc()
        return False

def test_simple_conversion():
    """测试简单的层转换"""
    print("\n" + "=" * 60)
    print("测试3: 神经网络层转换")
    print("=" * 60)
    
    try:
        import torch
        import torch.nn as nn
        from memtorch.mn import Linear
        from memtorch.bh.memristor import LinearIonDrift
        
        # 创建简单的线性层
        layer = nn.Linear(4, 2)
        layer.weight.data = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                                          [5.0, 6.0, 7.0, 8.0]])
        layer.bias.data = torch.tensor([0.1, 0.2])
        
        print("✅ 标准PyTorch层创建成功")
        
        # 转换为忆阻器层（使用小规模避免CUDA问题）
        mem_layer = Linear(
            layer,
            memristor_model=LinearIonDrift,
            memristor_model_params={
                'r_on': 100,
                'r_off': 16000,
                'time_series_resolution': 1e-4
            },
            tile_shape=(2, 2),  # 很小的瓦片
            verbose=False
        )
        
        print("✅ 忆阻器层转换成功")
        print(f"   交叉阵列形状: {mem_layer.crossbars[0].conductance_matrix.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 层转换测试失败: {e}")
        traceback.print_exc()
        return False

def test_cpu_forward():
    """测试CPU前向传播"""
    print("\n" + "=" * 60)
    print("测试4: CPU前向传播")
    print("=" * 60)
    
    try:
        import torch
        import torch.nn as nn
        from memtorch.mn import Linear
        from memtorch.bh.memristor import LinearIonDrift
        
        layer = nn.Linear(4, 2)
        mem_layer = Linear(
            layer,
            memristor_model=LinearIonDrift,
            memristor_model_params={'r_on': 100, 'r_off': 16000},
            tile_shape=(2, 2),
            verbose=False
        )
        
        # CPU推理（传统模式）
        x = torch.randn(2, 4)
        mem_layer.forward_legacy_enabled = True
        y = mem_layer(x)
        
        print(f"✅ CPU推理成功")
        print(f"   输入形状: {x.shape}")
        print(f"   输出形状: {y.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ CPU推理测试失败: {e}")
        traceback.print_exc()
        return False

def test_gpu_forward():
    """测试GPU前向传播（关键测试！）"""
    print("\n" + "=" * 60)
    print("测试5: GPU前向传播（关键！）")
    print("=" * 60)
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("⏭️  跳过GPU测试（CUDA不可用）")
            return True
        
        import torch.nn as nn
        from memtorch.mn import Linear
        from memtorch.bh.memristor import LinearIonDrift
        
        layer = nn.Linear(4, 2)
        mem_layer = Linear(
            layer,
            memristor_model=LinearIonDrift,
            memristor_model_params={'r_on': 100, 'r_off': 16000},
            tile_shape=(2, 2),
            verbose=False
        )
        
        # GPU推理（忆阻器模式）
        x = torch.randn(2, 4).cuda()
        mem_layer.forward_legacy_enabled = False
        
        print("   开始GPU推理...")
        y = mem_layer(x)
        torch.cuda.synchronize()
        
        print(f"✅ GPU推理成功（这是之前崩溃的地方！）")
        print(f"   输入形状: {x.shape}")
        print(f"   输出形状: {y.shape}")
        print(f"   输出设备: {y.device}")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU推理测试失败: {e}")
        print("\n   这可能表示CUDA代码仍有问题")
        print("   建议:")
        print("   1. 确保运行了 apply_cuda_fixes.py")
        print("   2. 重新编译: pip uninstall memtorch -y && pip install -e .")
        print("   3. 或降级到Python 3.11")
        traceback.print_exc()
        return False

def test_cuda_bindings():
    """测试CUDA绑定是否正确加载"""
    print("\n" + "=" * 60)
    print("测试6: CUDA绑定")
    print("=" * 60)
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("⏭️  跳过CUDA绑定测试（CUDA不可用）")
            return True
        
        import memtorch
        
        if '-cpu' in memtorch.__version__:
            print("⚠️  这是CPU版本，无CUDA绑定")
            return True
        
        try:
            import memtorch_cuda_bindings
            print("✅ CUDA绑定加载成功")
            return True
        except ImportError as e:
            print(f"❌ CUDA绑定未找到: {e}")
            print("   这表明MemTorch是在CPU模式编译的")
            return False
            
    except Exception as e:
        print(f"❌ CUDA绑定测试失败: {e}")
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("""
╔════════════════════════════════════════════════════════════╗
║                  MemTorch 最小化测试                        ║
║              用于验证修复是否成功                           ║
╚════════════════════════════════════════════════════════════╝
    """)
    
    print(f"Python版本: {sys.version}")
    print(f"Python路径: {sys.executable}")
    print()
    
    results = []
    
    # 运行所有测试
    results.append(("基本导入", test_imports()))
    
    if results[-1][1]:  # 如果导入成功，继续其他测试
        results.append(("忆阻器模型", test_memristor_model()))
        results.append(("层转换", test_simple_conversion()))
        results.append(("CPU推理", test_cpu_forward()))
        results.append(("GPU推理", test_gpu_forward()))
        results.append(("CUDA绑定", test_cuda_bindings()))
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name:20s}: {status}")
    
    print(f"\n总计: {passed}/{total} 通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！MemTorch已正确配置")
        print("\n现在可以运行 Exemplar_Simulations.ipynb 了")
        return 0
    else:
        print("\n⚠️  部分测试失败")
        print("\n建议操作:")
        print("1. 检查上面的错误信息")
        print("2. 运行修复脚本: python3 /workspace/apply_cuda_fixes.py")
        print("3. 重新编译: pip uninstall memtorch -y && pip install -e /workspace")
        print("4. 或查看完整指南: cat /workspace/PYTHON312_COMPATIBILITY_GUIDE.md")
        return 1

if __name__ == '__main__':
    sys.exit(main())
