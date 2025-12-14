# 📚 Educational Notebook 使用指南

## 文件概览

### ✨ 新创建的教学版Notebook
**`inference_maze_flow_educational.ipynb`** - 完整教学版本（强烈推荐！）⭐⭐⭐
- 37个精心设计的cells
- 完整的理论讲解（LaTeX公式）
- 详细的中文注释
- 渐进式学习路径
- 调参指南和常见问题解答
- 图表使用英文标注（避免中文报错）

### 📁 其他Notebook文件
1. `inference_maze_flow.ipynb` - 基础版本
2. `inference_maze_flow_enhanced.ipynb` - 增强版本（如果存在）

## 🎯 Educational Notebook 特点

### 1. 结构完整（37个cells）

#### 📖 理论部分 (Cells 1-6)
- Cell 1: **标题和理论介绍**
  - Flow Matching数学原理
  - CBF理论基础
  - 时间门控策略
  
- Cells 2-6: **环境配置**
  - 导入依赖库
  - 定义迷宫和障碍物
  - 参数配置（详细注释）

#### 🏗️ 模型定义 (Cells 7-15)
- Cell 7: **模型架构总览**
- Cells 8-12: **核心组件**
  - Scaler（归一化）
  - 激活函数（Mish, ResidualFC）
  - UpSampler/DownSampler
  - UNet with Attention
  - 完整Flow模型

- Cells 13-15: **模型加载**
  - 加载训练数据
  - 初始化模型
  - 加载权重

#### 🛡️ CBF实现 (Cells 16-24)
- Cells 16-18: **时间门控函数**
  - smoothstep函数
  - time_gate_ft（门控）
  - time_blow_up（惩罚）

- Cells 19-20: **CBF计算**
  - 椭圆h值和梯度计算
  
- Cells 21-22: **CBF投影核心算法**
  - 速度修正
  - 闭式投影解

#### 🚀 推理和可视化 (Cells 23-30)
- Cell 23: **可视化函数**
- Cells 24-25: **轨迹生成**
  - 无CBF版本
  - 有CBF版本
- Cells 26-28: **生成轨迹**
- Cells 29-30: **对比可视化**

#### 📊 总结 (Cells 31-37)
- Cell 31: **完整总结和调参指南**
  - 关键要点回顾
  - 4种常见场景的解决方案
  - 建议的实验
  - 进一步学习资源

### 2. 教学特色 ✨

#### LaTeX公式支持
```markdown
$$\frac{dx}{dt} = v_\theta(x, t)$$
$$h(x) = \left(\frac{x_{rot}}{a}\right)^2 + \left(\frac{y_{rot}}{b}\right)^2 - 1$$
```

#### 详细的中文注释
```python
# 1. 转换到椭圆局部坐标
y = ellipse.R_inv @ (points_2N - ellipse.center.view(2,1))

# 2. 椭圆半轴
a, b = ellipse.width/2.0, ellipse.height/2.0

# 3. 计算h值
h = (x_rot/a)**2 + (y_rot/b)**2 - 1.0
```

#### 英文图表标注
```python
axes[0].set_title("WITHOUT CBF\n(may collide with ellipses)")
axes[0].set_xlabel('X coordinate')
axes[0].set_ylabel('Y coordinate')
```

#### 渐进式学习
- 从简单到复杂
- 每个概念都有解释
- 代码和理论并行

## 🚀 快速开始

### 步骤1: 启动Jupyter
```bash
cd /home/nvidiapc/Flow_ICLR/fmtorch
jupyter notebook research/Maze_Experiment/
```

### 步骤2: 打开Notebook
在浏览器中点击 `inference_maze_flow_educational.ipynb`

### 步骤3: 运行
- **推荐**: 逐个运行cells，仔细阅读每个说明
- **快速**: 点击 "Cell" → "Run All"

### 步骤4: 观察输出
- 配置信息
- 模型加载状态
- 生成轨迹的进度
- 对比可视化图

## 📖 学习路径

### 初学者路径 (第一次使用)
1. **通读Cell 1** - 理解理论基础（30分钟）
2. **运行Cells 2-6** - 环境配置（5分钟）
3. **阅读Cells 7-15** - 理解模型架构（45分钟）
4. **运行Cells 16-30** - 看CBF的魔力（15分钟）
5. **研读Cell 31** - 学习调参（30分钟）

**总时长**: ~2小时

### 进阶路径 (已了解基础)
1. **快速浏览理论** - Cell 1（10分钟）
2. **直接运行生成** - Cells 2-30（10分钟）
3. **修改参数实验** - 调整cfg参数（30分钟）
4. **添加新障碍物** - 扩展ELLIPSES（20分钟）

**总时长**: ~1小时

### 专家路径 (深入研究)
1. **修改模型架构** - 调整网络层数/宽度
2. **实现新CBF函数** - 支持其他障碍物形状
3. **优化求解器** - 尝试自定义ODE solver
4. **批量实验** - 系统性测试参数空间

## 🎛️ 关键参数速查

### 时间门控（最重要！）
```python
cfg.MASK_GATE = 1.0        # 全局强度 [0,1]
cfg.SUPPRESS_FRAC = 0.50   # CBF开始时刻
cfg.RAMP_FRAC = 0.80       # CBF完全激活时刻
```

**调参口诀**：
- 轨迹穿障碍 → 降SUPPRESS_FRAC（更早激活）
- 轨迹太保守 → 升SUPPRESS_FRAC（更晚激活）
- 想要平滑过渡 → 拉大(RAMP - SUPPRESS)

### CBF强度
```python
cfg.ODE_CBF_PASSES = 5         # 迭代次数 [3-10]
cfg.ODE_CBF_MAX_CORR_NORM = 1.9  # 修正幅度 [1.0-3.0]
cfg.ODE_CBF_MARGIN = 0.0       # 安全距离 [0-0.05]
```

### ODE求解器
```python
cfg.T_SPAN = torch.linspace(0., 1., 100)  # 时间点数
cfg.SOLVER_METHOD = 'dopri5'  # 算法选择
cfg.SOLVER_TOLERANCE = 1e-5   # 精度
```

## 🧪 推荐的实验

### 实验1: 时间门控的影响
```python
# 在Cell中修改并重新运行
experiments = [
    (0.3, 0.5),  # 早期激活
    (0.5, 0.8),  # 默认
    (0.7, 0.9),  # 晚期激活
]
for suppress, ramp in experiments:
    cfg.SUPPRESS_FRAC = suppress
    cfg.RAMP_FRAC = ramp
    # 生成并保存轨迹
```

### 实验2: CBF强度测试
```python
for strength in [0.3, 0.5, 0.7, 1.0]:
    cfg.MASK_GATE = strength
    # 观察避障效果的变化
```

### 实验3: 障碍物密度
```python
# 添加更多椭圆
ELLIPSES.append(EllipseObstacle(5.0, 5.5, 2.0, 1.5, 0))
ELLIPSES.append(EllipseObstacle(9.5, 5.0, 1.8, 2.2, 45))
# 重新运行生成，测试CBF鲁棒性
```

### 实验4: 求解器对比
```python
results = {}
for method in ['euler', 'rk4', 'dopri5', 'dopri8']:
    cfg.SOLVER_METHOD = method
    start = time.time()
    traj = generate_trajectory(flow_model, scaler, cfg, use_cbf=True)
    duration = time.time() - start
    results[method] = (traj, duration)
# 比较精度和速度
```

## 🐛 常见问题排查

### Q1: Checkpoint not found
```
⚠️  WARNING: Checkpoint file not found!
```
**解决**: 确保已训练模型，或检查路径是否正确
```python
print(cfg.CHECKPOINT_PATH)  # 检查路径
```

### Q2: CUDA out of memory
```
RuntimeError: CUDA out of memory
```
**解决**: 
1. 减少T_SPAN点数
2. 使用CPU: `cfg.DEVICE = torch.device('cpu')`

### Q3: 轨迹仍然穿过障碍物
**解决**: 按Cell 31的调参指南调整参数

### Q4: 运行很慢
**解决**:
- 减少`ODE_CBF_PASSES`
- 降低`T_SPAN`点数
- 增大`SOLVER_TOLERANCE`

## 📊 输出文件

运行后会生成：
- `educational_comparison_cbf.png` - 对比图（无CBF vs 有CBF）
- 控制台输出包含详细的统计信息

## 💡 进阶技巧

### 技巧1: 保存实验结果
```python
import pickle
data = {
    'params': {
        'MASK_GATE': cfg.MASK_GATE,
        'SUPPRESS_FRAC': cfg.SUPPRESS_FRAC,
        # ... 其他参数
    },
    'trajs_no_cbf': trajs_no_cbf,
    'trajs_with_cbf': trajs_with_cbf,
}
with open('experiment_results.pkl', 'wb') as f:
    pickle.dump(data, f)
```

### 技巧2: 批量生成
```python
all_trajs = []
for i in range(100):
    traj = generate_trajectory(flow_model, scaler, cfg, use_cbf=True)
    all_trajs.append(traj)
np.save('batch_100_trajs.npy', all_trajs)
```

### 技巧3: 交互式调参
```python
from ipywidgets import interact, FloatSlider

@interact(mask_gate=FloatSlider(min=0, max=1, step=0.1, value=1.0))
def interactive_cbf(mask_gate):
    cfg.MASK_GATE = mask_gate
    traj = generate_trajectory(flow_model, scaler, cfg, use_cbf=True)
    # 可视化
```

### 技巧4: h值追踪
```python
# 在生成过程中记录h值
h_values = []
def fm_ode_func_with_logging(t, x_flat):
    # ... 原有代码 ...
    # 计算h值
    for e in ELLIPSES:
        h, _ = ellipse_h_grad_batch(x_world_2N, e)
        h_values.append((t, h.min().item()))
    # ...
```

## 📚 推荐阅读顺序

1. **第一遍**: 通读所有markdown cells，理解理论
2. **第二遍**: 运行所有code cells，观察输出
3. **第三遍**: 修改参数，做实验
4. **第四遍**: 扩展代码，添加新功能

## 🎓 学习检查清单

完成以下任务后，你就真正掌握了：

- [ ] 能解释Flow Matching的ODE公式
- [ ] 能画出模型架构图
- [ ] 理解时间门控的作用
- [ ] 会计算椭圆的h值和梯度
- [ ] 能推导CBF投影的闭式解
- [ ] 知道4种场景如何调参
- [ ] 成功生成了安全轨迹
- [ ] 完成了至少2个实验
- [ ] 能够添加新的障碍物
- [ ] 可以向他人解释CBF原理

## 🌟 进阶项目建议

1. **3D扩展**: 将2D迷宫扩展到3D空间
2. **动态障碍物**: 实现移动的椭圆
3. **多智能体**: 多条轨迹相互避让
4. **实时规划**: 在线更新CBF约束
5. **其他形状**: 实现多边形、不规则障碍物
6. **优化算法**: 尝试更高效的投影算法
7. **可视化增强**: 添加动画、3D可视化
8. **性能分析**: benchmark不同参数配置

## 📞 获取帮助

如果遇到问题：
1. 查看Cell 31的调参指南
2. 阅读代码中的详细注释
3. 检查README_NOTEBOOK.md
4. 对比原始脚本 `inference_maze_flow_cbf.py`

## 🎉 成就解锁

- 🌟 **Bronze**: 成功运行全部cells
- 🏆 **Silver**: 完成3个以上实验
- 💎 **Gold**: 成功添加新功能
- 👑 **Platinum**: 向他人教学CBF

---

**祝学习愉快！记住：理解比记忆更重要！** 🚀

最后更新: 2025-01-09
作者: Flow Matching + CBF教学团队

