# 🎓 Flow Matching + CBF 教育教程文件夹

这是一个**完全独立**的教学包，包含了学习Flow Matching和CBF所需的所有文件。

## 📁 文件夹内容

```
edu_tutorial/
├── 📓 inference_maze_flow_educational.ipynb  # 主教程notebook (37 cells)
├── 🏋️ checkpoints/                           # 预训练模型权重
│   └── maze_fixed_point_best.pt
├── 📊 data/                                   # 训练数据（用于对比）
│   └── trajectories_fixed_point.npy
├── 🖼️ results/                                # 生成结果保存目录
├── 📚 docs/                                   # 完整文档
│   ├── USAGE_GUIDE.md                        # 详细使用指南
│   ├── PARAMETER_TUNING.md                   # 调参手册
│   └── FAQ.md                                # 常见问题
└── 📖 README.md                              # 本文件

总大小: ~200MB (主要是模型权重)
```

## 🚀 快速开始（3步）

### 步骤1: 检查依赖
```bash
python -c "import torch, torchdiffeq, matplotlib; print('✅ All dependencies OK')"
```

### 步骤2: 启动Jupyter
```bash
# 在edu_tutorial目录下
jupyter notebook inference_maze_flow_educational.ipynb
```

### 步骤3: 运行
- 点击 "Cell" → "Run All" 或逐个运行cells

就这么简单！

---

## 🎯 这个教程适合谁？

- ✅ **初学者**: 从零开始学习Flow Matching和CBF
- ✅ **研究生**: 快速掌握安全AI控制方法
- ✅ **工程师**: 实现安全的机器人导航
- ✅ **教师**: 作为教学材料

---

## 📚 学习内容

### 理论部分
1. **Flow Matching**: 如何将噪声转换为有意义的轨迹
2. **Control Barrier Function (CBF)**: 数学上保证安全性
3. **时间门控**: 何时应用安全约束
4. **Blow-up惩罚**: 接近目标时增强约束

### 实践部分
1. **模型架构**: UpSampler → UNet → DownSampler
2. **CBF投影**: 闭式解算法实现
3. **参数调优**: 4种场景的解决方案
4. **可视化**: 对比有无CBF的效果

---

## ⏱️ 学习时间

- **快速预览**: 30分钟（运行全部cells）
- **基础学习**: 2-3小时（理解理论和代码）
- **深入掌握**: 4-5小时（含实验和调参）
- **完全精通**: 1-2天（含扩展和项目）

---

## 💻 系统要求

### 必需
- Python 3.8+
- PyTorch 1.10+
- torchdiffeq
- matplotlib
- numpy

### 推荐
- CUDA GPU（推理快10倍）
- 8GB+ RAM
- Jupyter Notebook或JupyterLab

### 安装依赖
```bash
pip install torch torchdiffeq matplotlib numpy jupyter
```

---

## 🎓 Notebook结构 (37 cells)

### Part 1: 理论 (6 cells)
- Flow Matching ODE
- CBF数学定义
- 时间门控策略

### Part 2: 模型 (9 cells)
- Scaler, UpSampler, DownSampler
- UNet with Attention
- 完整Pipeline

### Part 3: CBF (9 cells)
- 时间函数
- 椭圆h值计算
- 速度投影算法

### Part 4: 推理 (6 cells)
- 生成无CBF轨迹
- 生成有CBF轨迹
- 对比可视化

### Part 5: 总结 (7 cells)
- 调参指南
- 实验建议
- 进阶方向

---

## 🎛️ 关键参数（可调）

### 最重要: 时间门控
```python
cfg.MASK_GATE = 1.0        # CBF全局强度 [0,1]
cfg.SUPPRESS_FRAC = 0.50   # 前50%时间CBF=0
cfg.RAMP_FRAC = 0.80       # 80%后CBF=1

# 调参口诀:
# - 轨迹穿障碍物 → 降低SUPPRESS_FRAC (如0.3)
# - 轨迹太保守 → 提高SUPPRESS_FRAC (如0.7)
```

### CBF强度
```python
cfg.ODE_CBF_PASSES = 5         # 投影迭代次数 [3-10]
cfg.ODE_CBF_MAX_CORR_NORM = 1.9  # 最大修正幅度 [1-3]
```

### ODE求解器
```python
cfg.T_SPAN = torch.linspace(0., 1., 100)  # 时间点数
cfg.SOLVER_METHOD = 'dopri5'  # euler/rk4/dopri5
```

---

## 🧪 推荐实验

### 实验1: 时间门控影响
修改 `SUPPRESS_FRAC` 和 `RAMP_FRAC`，观察避障效果

### 实验2: CBF强度
调整 `MASK_GATE` 从 0.3 到 1.0，对比轨迹

### 实验3: 添加障碍物
在代码中添加新的椭圆，测试鲁棒性

### 实验4: 求解器对比
测试 'euler', 'rk4', 'dopri5' 的精度和速度

---

## 📊 预期输出

运行成功后会生成：

1. **控制台输出**
   - 模型加载信息
   - 轨迹生成进度
   - 统计信息

2. **可视化图像**
   - `results/educational_comparison_cbf.png`
   - 左图: 无CBF（红色，可能碰撞）
   - 右图: 有CBF（绿色，安全避障）

3. **学到的知识**
   - Flow Matching工作原理
   - CBF如何保证安全
   - 如何调参优化性能

---

## 🔧 故障排除

### 问题1: 找不到checkpoint
```bash
# 检查文件
ls -lh checkpoints/maze_fixed_point_best.pt

# 如果不存在，需要从主项目复制
```

### 问题2: CUDA内存不足
```python
# 在notebook中设置使用CPU
cfg.DEVICE = torch.device('cpu')
```

### 问题3: 导入fmtorch失败
```bash
# 确保在正确的Python环境中
# 并且fmtorch已安装
pip install -e /home/nvidiapc/Flow_ICLR/fmtorch
```

---

## 📦 移植到其他机器

### 方法1: 压缩整个文件夹
```bash
# 在edu_tutorial的父目录
tar -czf flow_cbf_tutorial.tar.gz edu_tutorial/
# 传输文件，然后在目标机器解压
tar -xzf flow_cbf_tutorial.tar.gz
cd edu_tutorial
jupyter notebook inference_maze_flow_educational.ipynb
```

### 方法2: Git仓库
```bash
cd edu_tutorial
git init
git add .
git commit -m "Flow Matching CBF Educational Tutorial"
# Push到你的仓库，在其他机器clone
```

---

## 🌟 进阶学习

完成基础教程后，可以尝试：

1. **扩展障碍物**: 实现多边形、不规则形状
2. **动态环境**: 移动的障碍物
3. **多智能体**: 多条轨迹相互避让
4. **3D扩展**: 将2D迷宫扩展到3D空间
5. **实时应用**: 集成到ROS机器人系统

---

## 📚 推荐阅读

**论文**:
- Flow Matching for Generative Modeling (ICLR 2023)
- Control Barrier Functions: Theory and Applications
- Neural Ordinary Differential Equations (NeurIPS 2018)

**教程**:
- docs/USAGE_GUIDE.md - 详细使用指南
- docs/PARAMETER_TUNING.md - 调参大全
- docs/FAQ.md - 常见问题

---

## 🤝 获取帮助

1. **查看文档**: docs/ 文件夹中的详细文档
2. **检查代码**: notebook中有详细注释
3. **运行示例**: 先运行默认参数，再修改
4. **记录实验**: 建议做实验笔记

---

## 🎉 学习成就

完成这个教程后，你将能够：

- ✅ 解释Flow Matching的数学原理
- ✅ 实现CBF安全约束
- ✅ 调优参数以达到最佳效果
- ✅ 设计自己的安全控制实验
- ✅ 向他人讲解这些概念

---

## 📝 版本信息

- **版本**: 1.0
- **创建日期**: 2025-01-09
- **适用于**: Flow Matching + CBF 学习
- **维护者**: Flow Matching教学团队

---

## 📄 许可

本教程遵循MIT许可证。自由使用、修改和分享！

---

## 🚀 现在就开始吧！

```bash
cd edu_tutorial
jupyter notebook inference_maze_flow_educational.ipynb
```

**记住**: 学习是一个过程，不要急于求成。理解每个概念比快速完成更重要！

祝学习愉快！ 🎓📚✨

---

**快速链接**:
- 📖 [详细使用指南](docs/USAGE_GUIDE.md)
- 🎛️ [调参手册](docs/PARAMETER_TUNING.md)
- ❓ [常见问题](docs/FAQ.md)
