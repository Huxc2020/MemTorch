#!/usr/bin/env python3
"""
临时修复脚本：减小Exemplar_Simulations的规模以在CPU上运行
注意：这只是权宜之计，强烈建议安装GPU版本！
"""

import json

# 读取notebook
with open('memtorch/examples/Exemplar_Simulations.ipynb', 'r') as f:
    notebook = json.load(f)

# 需要修改的配置
modifications = {
    "batch_size": 32,      # 原本可能是128或256
    "num_workers": 2,      # 减少数据加载线程
    "tile_shape": (64, 64), # 原本是(128, 128)
}

print("⚠️  警告：这是临时方案，会显著降低仿真精度和速度")
print("强烈建议安装GPU版本的MemTorch！")
print("\n建议的修改：")
for key, value in modifications.items():
    print(f"  - {key}: {value}")

print("\n请手动在notebook中应用这些修改：")
print("1. 减小batch_size到32")
print("2. 减小tile_shape到(64, 64)或更小")
print("3. 减少测试样本数量")
print("4. 只运行部分实验（如只运行Figure 1A）")
