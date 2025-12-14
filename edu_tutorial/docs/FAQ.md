# ❓ 常见问题解答 (FAQ)

## 安装和环境

### Q1: 如何安装所需的依赖？
```bash
pip install torch torchdiffeq matplotlib numpy jupyter
```

如果需要GPU支持：
```bash
# CUDA 11.x
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.x
pip install torch torchvision torchaudio
```

### Q2: 导入fmtorch失败怎么办？
```python
# 错误: ModuleNotFoundError: No module named 'fmtorch'
```

**解决方案**:
1. 检查fmtorch是否安装：
```bash
pip list | grep fmtorch
```

2. 如果未安装，从源码安装：
```bash
cd /path/to/fmtorch  # fmtorch根目录
pip install -e .
```

3. 或者添加到Python路径：
```python
import sys
sys.path.append('/path/to/fmtorch')
```

### Q3: Jupyter Notebook打不开怎么办？
```bash
# 安装Jupyter
pip install jupyter

# 启动
jupyter notebook

# 如果端口被占用
jupyter notebook --port 8889
```

---

## 文件和路径

### Q4: 找不到checkpoint文件
```
FileNotFoundError: checkpoints/maze_fixed_point_best.pt
```

**原因**: checkpoint文件缺失或路径错误

**解决**:
1. 检查文件是否存在：
```bash
ls -lh checkpoints/maze_fixed_point_best.pt
```

2. 如果不存在，从主项目复制：
```bash
cp /path/to/fmtorch/checkpoints/maze_fixed_point_best.pt checkpoints/
```

3. 或者重新训练模型（需要较长时间）

### Q5: 找不到训练数据
```
FileNotFoundError: data/trajectories_fixed_point.npy
```

**解决**: 从主项目复制数据文件
```bash
cp /path/to/Maze_Experiment/trajectories_fixed_point.npy data/
```

### Q6: 移动文件夹后路径不对
**症状**: 路径指向旧位置

**解决**: Notebook会自动使用当前目录作为PROJECT_ROOT
```python
# 在notebook中检查
print(f"Current directory: {os.getcwd()}")
print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print(f"Checkpoint exists: {os.path.exists(cfg.CHECKPOINT_PATH)}")
```

---

## 运行错误

### Q7: CUDA out of memory
```
RuntimeError: CUDA out of memory
```

**解决方案**:
1. 使用CPU（慢但稳定）：
```python
cfg.DEVICE = torch.device('cpu')
```

2. 减少batch大小（如果有）

3. 减少T_SPAN点数：
```python
cfg.T_SPAN = torch.linspace(0., 1., 50)  # 从100降到50
```

4. 清空GPU缓存：
```python
import torch
torch.cuda.empty_cache()
```

### Q8: 轨迹生成很慢
**症状**: 生成一条轨迹超过1分钟

**原因和解决**:

1. **使用CPU而非GPU**:
```python
print(cfg.DEVICE)  # 应该是 cuda
```

2. **参数设置过高**:
```python
# 优化这些参数
cfg.T_SPAN = torch.linspace(0., 1., 50)  # 减少点数
cfg.ODE_CBF_PASSES = 3  # 减少迭代
cfg.SOLVER_METHOD = 'rk4'  # 更快的求解器
```

3. **求解器精度过高**:
```python
cfg.SOLVER_TOLERANCE = 1e-4  # 从1e-5提高
```

### Q9: 出现NaN或Inf
```
RuntimeError: Function 'XxxBackward' returned nan values
```

**可能原因**:
1. 学习率过大（训练时）
2. 数值不稳定
3. Blow-up cap设置不当

**解决**:
```python
# 检查中间值
print(f"x range: {x.min():.2f} to {x.max():.2f}")
print(f"v range: {v.min():.2f} to {v.max():.2f}")

# 降低blow-up cap
cfg.ODE_CBF_BLOWUP_CAP = 100  # 从1000降低

# 增加数值稳定性
cfg.SOLVER_TOLERANCE = 1e-4  # 降低精度要求
```

---

## CBF相关

### Q10: 轨迹仍然穿过障碍物
**症状**: 即使启用CBF，轨迹still与椭圆相交

**诊断**:
```python
# 检查CBF是否真的启用
print(f"MASK_GATE: {cfg.MASK_GATE}")  # 应该 > 0
print(f"SUPPRESS_FRAC: {cfg.SUPPRESS_FRAC}")

# 检查h值
traj_tensor = torch.from_numpy(traj.T).to(cfg.DEVICE)
h_min = min_h_over_all(traj_tensor, ELLIPSES)
print(f"Min h: {h_min:.4f}")  # 应该 > 0
```

**解决**（按优先级）:
1. 降低SUPPRESS_FRAC到0.3
2. 增加ODE_CBF_PASSES到10
3. 增加ODE_CBF_MARGIN到0.02
4. 增加ODE_CBF_MAX_CORR_NORM到2.5

### Q11: 轨迹过于保守/不自然
**症状**: 轨迹过度弯曲，远离障碍物

**解决**:
1. 提高SUPPRESS_FRAC到0.7
2. 降低MASK_GATE到0.7
3. 降低ODE_CBF_PASSES到3
4. 降低MAX_CORR_NORM到1.0

### Q12: CBF修正太突然，轨迹有断点
**解决**: 平滑过渡
```python
# 增大过渡区间
cfg.SUPPRESS_FRAC = 0.4
cfg.RAMP_FRAC = 0.9  # 差值0.5

# 降低修正幅度
cfg.ODE_CBF_MAX_CORR_NORM = 1.0

# 增加时间点
cfg.T_SPAN = torch.linspace(0., 1., 200)
```

---

## 可视化

### Q13: 图表显示中文乱码
**原因**: 字体不支持中文

**解决**: Notebook中已使用英文标注，不应出现此问题

如果自己添加中文：
```python
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
```

### Q14: 图片保存后看不清
**解决**: 提高DPI
```python
plt.savefig('output.png', dpi=300, bbox_inches='tight')  # 默认是300
```

### Q15: 想要动画而不是静态图
```python
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

fig, ax = plt.subplots(figsize=(10, 10))
create_maze_background(ax, show_ellipses=True)
line, = ax.plot([], [], 'r-', linewidth=2)
point, = ax.plot([], [], 'ro', markersize=10)

def animate(i):
    line.set_data(traj[:i, 0], traj[:i, 1])
    point.set_data([traj[i, 0]], [traj[i, 1]])
    return line, point

anim = FuncAnimation(fig, animate, frames=len(traj), interval=50, blit=True)
HTML(anim.to_jshtml())
```

---

## 参数调优

### Q16: 不知道该调哪个参数
**参考顺序**:
1. 先调 SUPPRESS_FRAC（最重要）
2. 再调 MASK_GATE
3. 然后调 ODE_CBF_PASSES
4. 最后调其他参数

### Q17: 调参后效果反而更差
**可能原因**: 同时改了多个参数

**建议**:
1. 每次只改一个参数
2. 记录基准性能
3. 对比改动前后
4. 如果变差，回滚

### Q18: 如何评估参数的好坏？
**定义指标**:
```python
def evaluate_config(trajs, ellipses):
    metrics = {
        'safety_rate': 0,  # 完全安全的轨迹比例
        'avg_min_h': 0,    # 平均最小h值
        'smoothness': 0,   # 轨迹平滑度
        'computation_time': 0  # 计算时间
    }
    
    for traj in trajs:
        traj_tensor = torch.from_numpy(traj.T).to(device)
        h_min = min_h_over_all(traj_tensor, ellipses)
        
        if h_min > 0:
            metrics['safety_rate'] += 1
        metrics['avg_min_h'] += h_min
        
        # 计算平滑度（相邻点的加速度）
        diff1 = np.diff(traj, axis=0)
        diff2 = np.diff(diff1, axis=0)
        metrics['smoothness'] += -np.mean(np.linalg.norm(diff2, axis=1))
    
    metrics['safety_rate'] /= len(trajs)
    metrics['avg_min_h'] /= len(trajs)
    metrics['smoothness'] /= len(trajs)
    
    return metrics
```

---

## 扩展和修改

### Q19: 如何添加新的障碍物？
```python
# 在定义ELLIPSES的cell中添加
ELLIPSES.append(EllipseObstacle(
    center_x=5.0,  # x坐标
    center_y=6.0,  # y坐标
    width=2.5,     # 宽度
    height=2.0,    # 高度
    angle=30,      # 旋转角度（度）
    buffer_ratio=1.0  # 缓冲比例
))

# 记得移动到GPU
ELLIPSES[-1].to(cfg.DEVICE)

# 可能需要增加passes
cfg.ODE_CBF_PASSES = len(ELLIPSES) * 2
```

### Q20: 如何修改迷宫？
```python
# 修改LARGE_MAZE_DIVERSE_GR定义
LARGE_MAZE_DIVERSE_GR = [
    [1, 1, 1, 1, 1],  # 可以改成任意大小
    [1, 0, 0, 0, 1],
    [1, 0, 1, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1],
]

# 注意: 迷宫只是背景，主要约束来自椭圆
```

### Q21: 如何保存生成的轨迹？
```python
import pickle

# 保存单条轨迹
np.save('my_trajectory.npy', traj)

# 保存多条轨迹和配置
data = {
    'trajectories': generated_trajectories,
    'config': vars(cfg),
    'timestamp': time.time()
}

with open('experiment_results.pkl', 'wb') as f:
    pickle.dump(data, f)
```

---

## 理论理解

### Q22: Flow Matching和Diffusion有什么区别？
**Flow Matching**:
- 确定性ODE: $dx/dt = v_\theta(x, t)$
- 学习速度场
- 推理时一次前向pass

**Diffusion**:
- 随机SDE: $dx = v_\theta(x, t)dt + \sigma dW$
- 学习噪声或score
- 推理时多步去噪

**优势**: Flow Matching通常更快，且更容易与CBF结合

### Q23: 为什么需要时间门控？
**原因**:
1. 早期（t≈0）：轨迹在噪声空间，障碍物无意义
2. 中期（t≈0.5）：轨迹逐渐成形，开始需要避障
3. 后期（t≈1）：接近真实空间，必须严格避障

**如果不用门控**: CBF会在早期过度修正噪声，破坏Flow的自然演化

### Q24: CBF如何保证安全？
**数学保证**:
$$\frac{dh}{dt} = \nabla h \cdot v \geq -\alpha(h)$$

如果初始状态安全（$h(x_0) > 0$），且速度满足CBF条件，则$h(x_t) > 0, \forall t$

**实践中**: 通过投影修正速度，确保每一步都满足CBF条件

### Q25: Blow-up函数φ(t)的作用？
$$\phi(t) = \begin{cases}
1 & t < \theta T \\
\frac{1}{T-t} & t \geq \theta T
\end{cases}$$

**作用**: 在接近终点时，$\phi(t) \to \infty$，使得违反约束的惩罚 $\to \infty$，强制满足安全约束

---

## 性能优化

### Q26: 如何加速推理？
**策略**（按效果排序）:
1. 使用GPU（最重要，10x加速）
2. 减少T_SPAN点数（50 vs 100）
3. 降低ODE_CBF_PASSES（3 vs 5）
4. 切换到rk4求解器
5. 提高SOLVER_TOLERANCE（1e-4 vs 1e-5）

### Q27: 如何批量生成轨迹？
```python
# 串行生成
trajs = []
for i in range(100):
    traj = generate_trajectory(flow_model, scaler, cfg, use_cbf=True)
    trajs.append(traj)

# 可以考虑多进程（需要小心GPU内存）
from multiprocessing import Pool

def generate_one(seed):
    torch.manual_seed(seed)
    return generate_trajectory(flow_model, scaler, cfg, use_cbf=True)

with Pool(4) as p:
    trajs = p.map(generate_one, range(100))
```

### Q28: 模型太大，如何压缩？
```python
# 量化模型（训练后）
import torch.quantization as quantization

model_quantized = quantization.quantize_dynamic(
    flow_model, {nn.Linear}, dtype=torch.qint8
)

# 可以减小模型大小，略微降低精度
```

---

## 移植和分享

### Q29: 如何分享这个教程给别人？
**方法1: 压缩文件夹**
```bash
cd edu_tutorial/..
tar -czf flow_cbf_tutorial.tar.gz edu_tutorial/
# 发送这个压缩包
```

**方法2: Git仓库**
```bash
cd edu_tutorial
git init
git add .
git commit -m "Flow Matching CBF Tutorial"
# Push到GitHub等平台
```

### Q30: 在另一台机器运行需要什么？
**必需**:
1. Python 3.8+
2. PyTorch
3. torchdiffeq, matplotlib, numpy
4. Jupyter Notebook

**可选**:
- CUDA GPU（推理快）
- 足够的RAM（8GB+）

**步骤**:
```bash
# 解压
tar -xzf flow_cbf_tutorial.tar.gz
cd edu_tutorial

# 安装依赖
pip install torch torchdiffeq matplotlib numpy jupyter

# 启动
jupyter notebook inference_maze_flow_educational.ipynb
```

---

## 其他

### Q31: 这个教程适合什么水平？
- ✅ Python基础
- ✅ 了解神经网络概念
- ✅ 基本的数学（微积分、线性代数）
- ❌ 不需要Flow Matching或CBF先验知识

### Q32: 学完这个教程能做什么？
- ✅ 理解Flow Matching原理和实现
- ✅ 掌握CBF安全控制方法
- ✅ 能够调参优化性能
- ✅ 可以扩展到其他应用（机器人导航等）
- ✅ 有基础进行相关研究

### Q33: 有推荐的进阶资料吗？
**论文**:
- Flow Matching for Generative Modeling (ICLR 2023)
- Control Barrier Functions: Theory and Applications
- Neural ODEs (NeurIPS 2018)

**代码**:
- 查看主项目的其他实验
- 研究inference_maze_flow_cbf.py完整实现

**应用**:
- ROS集成教程
- 多智能体协同
- 实时路径规划

---

## 联系和反馈

### Q34: 发现bug或有建议怎么办？
1. 检查docs/中的文档
2. 查看本FAQ
3. 在主项目提issue
4. 或联系维护者

### Q35: 可以用于商业项目吗？
请查看LICENSE文件。通常遵循MIT许可，允许商业使用。

---

**找不到答案？**

1. 查看docs/USAGE_GUIDE.md
2. 查看docs/PARAMETER_TUNING.md
3. 阅读notebook中的详细注释
4. 对比inference_maze_flow_cbf.py源码

**持续更新中** - 如果你有新问题，欢迎贡献到这个FAQ！

最后更新: 2025-01-09
版本: 1.0

