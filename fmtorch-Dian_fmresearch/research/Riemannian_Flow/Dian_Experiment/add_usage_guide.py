#!/usr/bin/env python3
"""
在notebook开头添加简洁的使用说明
"""

import nbformat as nbf

notebook_path = './pouring_overfit_test.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

# 更新第一个cell的说明
new_intro_cell = nbf.v4.new_markdown_cell("""# SE(3) 轨迹Flow Matching过拟合测试

## 📋 概述

本notebook对单条倒水轨迹（`1_water_200.pkl`）进行**Flow Matching过拟合测试**。

### 🎯 任务目标
1. **数据预处理**：对比两种SE(3)轨迹表示方法（9D绝对位姿 vs ΔSE(3)增量）
2. **重构验证**：通过数值和可视化评估预处理管线的正确性
3. **Flow Matching训练**：使用**fmtorch库**和**torchdiffeq**进行过拟合训练
4. **超参数扫描**：15-20组配置（网络大小、学习率调度、初始学习率）

### 🔧 技术栈
- **Flow Matching基础设施**：严格对齐`Car_FM_SingleStep_enhanced.ipynb`
  - `fmtorch.path.AffinePath` + `fmtorch.scheduler.CondOTScheduler`
  - `fmtorch.utils.ModelWrapper`（网络基类）
  - `torchdiffeq.odeint`（ODE求解）
- **网络架构**：CBAM Attention + 多残差Transformer块（比single-step car更大）
- **数据表示**：ΔSE(3)（方法2，6D增量：3D旋转ω + 3D平移v）

### 📂 输出结果
- `artifacts/pouring_overfit_single/`：所有中间结果和最终模型
- 最佳模型权重：`best_model_sweep_X.pt`
- 训练曲线、重构误差、可视化对比图

### 🚀 使用方法
按顺序运行所有cells即可。主要步骤：
1. **Cells 1-20**：数据加载和预处理
2. **Cells 21-40**：重构验证和误差分析
3. **Cells 41-50**：可视化对比
4. **Cells 51-70**：Flow Matching训练（超参数扫描）
5. **Cells 71-80**：推理和最终评估

---
""")

nb.cells[0] = new_intro_cell

# 在Flow Matching部分前添加一个分隔标题
fm_start_idx = None
for idx, cell in enumerate(nb.cells):
    if cell.cell_type == 'code' and '# 导入必要的库' in cell.source and 'torch' in cell.source:
        # 在这个cell前插入一个markdown标题
        fm_start_idx = idx
        break

if fm_start_idx:
    fm_section_cell = nbf.v4.new_markdown_cell("""---

# 🔥 Flow Matching训练部分

使用**方法2 (ΔSE(3))**进行Flow Matching过拟合训练。

## 核心组件
- **数据**：`Z_delta` (479, 6) - 白化后的se(3)增量序列
- **网络**：`EnhancedVectorFieldNet` - CBAM + 多残差Transformer（继承`ModelWrapper`）
- **路径**：`AffinePath(CondOTScheduler())` - OT-CFM插值：x_t = (1-t)x0 + t*x1
- **求解器**：`torchdiffeq.odeint(dopri5)` - 从噪声积分到数据

## 训练流程
1. 定义网络和辅助模块（Mish, ResidualFC, CBAM等）
2. 实现训练函数（使用`path.sample()`计算目标速度场）
3. 超参数扫描（15-20组配置）
4. ODE推理（从标准高斯噪声采样轨迹）
5. 重构SE(3)轨迹并评估误差

---
""")
    nb.cells.insert(fm_start_idx, fm_section_cell)

# 保存
with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("✓ 使用说明已添加到notebook开头")
print("✓ Flow Matching部分已添加分隔标题")

