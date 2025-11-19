#!/usr/bin/env python3
"""
自动修复MemTorch CUDA代码中的严重bug
针对Python 3.12 + CUDA 13.0环境
"""

import os
import sys
import shutil
from pathlib import Path

def backup_file(filepath):
    """备份文件"""
    backup_path = f"{filepath}.backup"
    shutil.copy2(filepath, backup_path)
    print(f"✅ 已备份: {backup_path}")
    return backup_path

def fix_tile_matmul_kernels():
    """修复 tile_matmul_kernels.cu 中的内存管理错误"""
    filepath = "/workspace/memtorch/cu/tile_matmul_kernels.cu"
    
    if not os.path.exists(filepath):
        print(f"❌ 文件不存在: {filepath}")
        return False
    
    print(f"\n修复文件: {filepath}")
    backup_file(filepath)
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # 修复1：移除第一个错误的free()
    if 'free(&partial_sum);' in content:
        # 计算出现次数
        count = content.count('free(&partial_sum);')
        print(f"   发现 {count} 处错误的 free(&partial_sum) 调用")
        
        # 替换所有出现
        content = content.replace(
            '    free(&partial_sum);',
            '    // FIXED: Removed incorrect free() call - Eigen objects are stack-allocated'
        )
        
        with open(filepath, 'w') as f:
            f.write(content)
        
        print(f"✅ 已移除 {count} 处错误的 free() 调用")
        return True
    else:
        print("   ⚠️  未发现 free(&partial_sum)，可能已修复")
        return True

def fix_setup_py():
    """修复 setup.py 中的编译参数和依赖"""
    filepath = "/workspace/setup.py"
    
    if not os.path.exists(filepath):
        print(f"❌ 文件不存在: {filepath}")
        return False
    
    print(f"\n修复文件: {filepath}")
    backup_file(filepath)
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    modified = False
    new_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # 修复1：sklearn → scikit-learn
        if '"sklearn"' in line:
            line = line.replace('"sklearn"', '"scikit-learn"')
            print("   ✅ sklearn → scikit-learn")
            modified = True
        
        # 修复2：更新CUDA编译参数
        if 'extra_compile_args=["-lineinfo", "-use_fast_math"]' in line:
            indent = len(line) - len(line.lstrip())
            new_lines.append(' ' * indent + 'extra_compile_args={\n')
            new_lines.append(' ' * (indent + 4) + "'cxx': ['-std=c++14', '-O3'],\n")
            new_lines.append(' ' * (indent + 4) + "'nvcc': [\n")
            new_lines.append(' ' * (indent + 8) + "'-std=c++14',\n")
            new_lines.append(' ' * (indent + 8) + "'-lineinfo',\n")
            new_lines.append(' ' * (indent + 8) + "'-use_fast_math',\n")
            new_lines.append(' ' * (indent + 8) + "'--expt-relaxed-constexpr',\n")
            new_lines.append(' ' * (indent + 4) + "]\n")
            new_lines.append(' ' * indent + '},\n')
            print("   ✅ 更新CUDA编译参数")
            modified = True
            i += 1
            continue
        
        # 修复3：更新C++编译参数
        if 'extra_compile_args=["-O3"]' in line and 'CppExtension' in ''.join(lines[max(0,i-10):i]):
            line = line.replace(
                'extra_compile_args=["-O3"]',
                'extra_compile_args=["-std=c++14", "-O3"]'
            )
            print("   ✅ 更新C++编译参数")
            modified = True
        
        new_lines.append(line)
        i += 1
    
    if modified:
        with open(filepath, 'w') as f:
            f.writelines(new_lines)
        print("✅ setup.py 修复完成")
        return True
    else:
        print("   ⚠️  setup.py 可能已修复")
        return True

def add_memory_safety_wrapper():
    """添加内存安全包装器"""
    filepath = "/workspace/memtorch_safe_wrapper.py"
    
    content = '''"""
MemTorch 安全包装器
为Python 3.12环境添加额外的内存安全检查
"""

import gc
import torch
import sys
import warnings

def safe_cuda_cleanup():
    """安全的CUDA内存清理"""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
    except Exception as e:
        warnings.warn(f"CUDA清理时出错: {e}")

def check_python_version():
    """检查Python版本兼容性"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 12:
        warnings.warn(
            f"检测到Python {version.major}.{version.minor}，这是一个很新的版本。"
            f"MemTorch主要在Python 3.9-3.11上测试。"
            f"如果遇到问题，请考虑使用Python 3.11。"
        )
    return version

def setup_safe_environment():
    """设置安全的运行环境"""
    check_python_version()
    
    # 设置更保守的CUDA内存分配
    if torch.cuda.is_available():
        # 减小CUDA内存池大小以避免碎片化
        torch.cuda.set_per_process_memory_fraction(0.9, 0)
    
    print("✅ 安全环境已配置")

# 自动设置
setup_safe_environment()
'''
    
    with open(filepath, 'w') as f:
        f.write(content)
    
    print(f"\n✅ 创建安全包装器: {filepath}")
    return True

def verify_fixes():
    """验证修复"""
    print("\n" + "="*60)
    print("验证修复结果")
    print("="*60)
    
    # 检查 tile_matmul_kernels.cu
    filepath = "/workspace/memtorch/cu/tile_matmul_kernels.cu"
    with open(filepath, 'r') as f:
        content = f.read()
    
    if 'free(&partial_sum)' in content:
        print("❌ tile_matmul_kernels.cu 仍包含错误的free()调用")
        return False
    else:
        print("✅ tile_matmul_kernels.cu 已正确修复")
    
    # 检查 setup.py
    filepath = "/workspace/setup.py"
    with open(filepath, 'r') as f:
        content = f.read()
    
    if '"sklearn"' in content:
        print("❌ setup.py 仍使用 sklearn（应为scikit-learn）")
        return False
    else:
        print("✅ setup.py 依赖已更新")
    
    return True

def main():
    """主函数"""
    print("="*60)
    print("MemTorch CUDA Bug自动修复工具")
    print("针对Python 3.12 + CUDA 13.0")
    print("="*60)
    
    print("\n⚠️  重要提示:")
    print("  1. 此脚本会修改源代码文件")
    print("  2. 原文件会自动备份（.backup后缀）")
    print("  3. 修复后需要重新编译安装MemTorch")
    
    input("\n按Enter继续，或Ctrl+C取消...")
    
    # 执行修复
    success = True
    
    try:
        success &= fix_tile_matmul_kernels()
        success &= fix_setup_py()
        success &= add_memory_safety_wrapper()
        
        if verify_fixes():
            print("\n" + "="*60)
            print("✅ 所有修复已成功应用！")
            print("="*60)
            
            print("\n下一步：")
            print("  1. 重新编译安装MemTorch:")
            print("     cd /workspace")
            print("     pip uninstall memtorch -y")
            print("     pip install -e .")
            print("")
            print("  2. 在Jupyter Notebook中导入安全包装器:")
            print("     from memtorch_safe_wrapper import *")
            print("")
            print("  3. 重启Jupyter kernel并重新运行")
            print("")
            print("  4. 如果仍然崩溃，考虑:")
            print("     - 使用Python 3.11代替3.12")
            print("     - 减小仿真规模（batch_size, tile_shape）")
            
            return 0
        else:
            print("\n❌ 验证失败，请检查错误信息")
            return 1
            
    except Exception as e:
        print(f"\n❌ 修复过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
