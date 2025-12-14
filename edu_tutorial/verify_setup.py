#!/usr/bin/env python3
"""
验证edu_tutorial文件夹设置是否完整
"""

import os
import sys

print("=" * 70)
print("🔍 Flow Matching + CBF Tutorial - Setup Verification")
print("=" * 70)

# 检查当前目录
current_dir = os.path.abspath(os.getcwd())
print(f"\n📂 Current directory: {current_dir}")

if not current_dir.endswith('edu_tutorial'):
    print("⚠️  Warning: Should run this script from edu_tutorial directory")
    print(f"   Current: {current_dir}")
    print(f"   Expected: .../edu_tutorial")

# 必需文件列表
required_files = {
    'Core': [
        'inference_maze_flow_educational.ipynb',
        'README.md',
        'MANIFEST.txt',
        'verify_setup.py'
    ],
    'Checkpoints': [
        'checkpoints/maze_fixed_point_best.pt'
    ],
    'Data': [
        'data/trajectories_fixed_point.npy'
    ],
    'Documentation': [
        'docs/USAGE_GUIDE.md',
        'docs/PARAMETER_TUNING.md',
        'docs/FAQ.md',
        'docs/QUICK_START.md'
    ],
    'Directories': [
        'results/'
    ]
}

# 检查文件
print("\n" + "=" * 70)
print("📋 Checking Files")
print("=" * 70)

all_ok = True
total_size = 0

for category, files in required_files.items():
    print(f"\n{category}:")
    for file_path in files:
        if file_path.endswith('/'):
            # 目录检查
            if os.path.isdir(file_path):
                print(f"  ✅ {file_path}")
            else:
                print(f"  ❌ {file_path} (missing)")
                all_ok = False
        else:
            # 文件检查
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path)
                total_size += size
                size_mb = size / (1024 * 1024)
                if size_mb > 1:
                    print(f"  ✅ {file_path} ({size_mb:.1f}MB)")
                else:
                    print(f"  ✅ {file_path} ({size/1024:.1f}KB)")
            else:
                print(f"  ❌ {file_path} (missing)")
                all_ok = False

# 总大小
print(f"\n📊 Total size: {total_size / (1024 * 1024):.1f}MB")

# 检查Python依赖
print("\n" + "=" * 70)
print("🐍 Checking Python Dependencies")
print("=" * 70)

dependencies = {
    'torch': 'PyTorch',
    'torchdiffeq': 'ODE Solver',
    'matplotlib': 'Visualization',
    'numpy': 'Numerical Computing',
    'jupyter': 'Notebook'
}

for module, name in dependencies.items():
    try:
        __import__(module)
        if module == 'torch':
            import torch
            cuda_available = torch.cuda.is_available()
            cuda_str = f"(CUDA: {'✅' if cuda_available else '❌'})"
            print(f"  ✅ {name:20s} {cuda_str}")
        else:
            print(f"  ✅ {name}")
    except ImportError:
        print(f"  ❌ {name} - Not installed")
        all_ok = False

# 检查notebook路径
print("\n" + "=" * 70)
print("📓 Checking Notebook Configuration")
print("=" * 70)

try:
    import json
    
    with open('inference_maze_flow_educational.ipynb', 'r') as f:
        nb = json.load(f)
    
    # 查找PROJECT_ROOT定义
    found_project_root = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'PROJECT_ROOT = os.path.abspath(os.getcwd())' in source:
                found_project_root = True
                print("  ✅ PROJECT_ROOT correctly set to current directory")
                break
    
    if not found_project_root:
        print("  ⚠️  PROJECT_ROOT not found or not set correctly")
    
    # 检查cell数量
    num_cells = len(nb['cells'])
    print(f"  ✅ Notebook has {num_cells} cells")
    
except Exception as e:
    print(f"  ❌ Error reading notebook: {e}")
    all_ok = False

# 最终状态
print("\n" + "=" * 70)
if all_ok:
    print("✅ SETUP COMPLETE - All files present and dependencies installed!")
    print("=" * 70)
    print("\n🚀 Ready to start! Run:")
    print("   jupyter notebook inference_maze_flow_educational.ipynb")
    sys.exit(0)
else:
    print("❌ SETUP INCOMPLETE - Some files or dependencies are missing")
    print("=" * 70)
    print("\n📖 Please refer to README.md for setup instructions")
    sys.exit(1)

