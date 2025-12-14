# 快速参考卡片 - OCP 1.2 圈配置

## 🎯 一行命令

```bash
python pin_fr3_draw_eight_ocp.py
```

## 📊 核心配置

```
轨迹: 1.2 圈 figure-8
时长: 13.612 秒
结点: 128 个 (K=128)
样本: 6806 个 (500 Hz)
变量: 2688 个 (7×128×3)
```

## ⚙️ 关键参数位置

| 参数 | 位置 | 默认值 |
|------|------|--------|
| 圈数 | 第 217 行 | `laps = 1.2` |
| 时长 | 第 218 行 | `T_total = 13.612` |
| 结点数 | 第 263 行 | `K = 128` |
| 路径权重 | 第 302 行 | `w_e = 1e5` |
| 平滑权重 | 第 303 行 | `w_dv = 1.0, w_da = 5.0` |
| 相位权重 | 第 304 行 | `w_dth = 1.0` |
| 配点 | 第 308 行 | `2 点 Gauss` |

## 🚀 常用修改

### 快速模式（K=64）
```python
# 第 263 行
K = 64  # ~2-4 分钟
```

### 高精度（K=256）
```python
# 第 263 行
K = 256  # ~8-15 分钟
```

### 更平滑扭矩
```python
# 第 303 行
w_dv, w_da = 2.0, 10.0
```

### 更严格相位均匀
```python
# 第 304 行
w_dth = 2.0
```

### 3 点配点（更高精度）
```python
# 第 308 行
colloc_s = [0.112701665379258, 0.5, 0.887298334620742]
```

## 📈 预期输出

```
[ocp] laps=1.2, T_total=13.612s
[ocp] Spline mode: K=128 knots, h_seg=0.1072s
[ocp] Building NLP with 2688 variables...
[solve] IPOPT start...
[solve] done in XXX.XXs
[dense] Generating 6806 samples...
[save] npz saved -> eight_ocp_dataset.npz
       Dense trajectory: 6806 samples @ 500 Hz (13.612s)
       Knot data: 128 knots @ 0.1072s intervals
```

## 🔍 验证脚本

```python
import numpy as np
d = np.load('eight_ocp_dataset.npz')
print(f"Samples: {len(d['t'])}")      # 6806
print(f"Duration: {d['t'][-1]:.3f}s") # 13.612
print(f"Rate: {d['sample_rate']} Hz") # 500.0
print(f"Laps: {(d['knots_theta'][-1] - d['knots_theta'][0])/(2*np.pi):.2f}") # 1.20
```

## ⚡ 性能

| K | 变量 | 约束 | 时间 | 内存 |
|---|------|------|------|------|
| 64 | 1344 | ~1400 | 2-4分钟 | 150 MB |
| **128** | **2688** | **~2800** | **4-7分钟** | **300 MB** |
| 256 | 5376 | ~5600 | 8-15分钟 | 600 MB |

## 🎨 可视化

MeshCat URL: `http://127.0.0.1:7000/static/`

控制速度: `--speed 2.5` (默认)

## 📁 输出文件

```
eight_ocp_dataset.npz  (~11 MB)
├── t, q, dq, ddq, tau (6806 × 维度)
├── ee, ee_des, theta
├── knots_q, knots_dq, knots_ddq, knots_theta (128 × 维度)
└── sample_rate, x_plane, fig8_params
```

## 🐛 故障排除

**IPOPT 不收敛**
```python
# 第 350 行
"ipopt.max_iter": 3000,  # 从 2000 增加
"ipopt.tol": 1e-3,       # 从 1e-4 放松
```

**内存不足**
```python
# 第 263 行
K = 64  # 降低结点数
```

**扭矩尖峰**
```python
# 第 303 行
w_dv, w_da = 2.0, 10.0  # 增加平滑权重
```

## 📞 环境要求

```bash
# 必需
conda activate casadi_env
pip install pinocchio[meshcat]>=2.7.0
pip install casadi numpy matplotlib

# 检查 URDF
ls panda_description/urdf/panda_stick.urdf
```

## 🔗 相关文件

- **完整文档**: `USAGE_1.2_LAPS.md`
- **改进说明**: `RECENT_IMPROVEMENTS.md`
- **对比分析**: `COMPARISON.md`
- **升级笔记**: `UPGRADE_NOTES.md`

---

**快速开始**: `python pin_fr3_draw_eight_ocp.py` 🚀

