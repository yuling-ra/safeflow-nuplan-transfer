# ⚡ Quick Guide - 30 Second Start

## 🎯 What This Does

手动调参 → 立即看结果 → 快速迭代优化

## 🚀 How to Use

### Step 1: 打开文件
```bash
cd Quick_Tuning
nano quick_tune.py  # 或用任何编辑器
```

### Step 2: 修改参数（第 24-45 行）
```python
# 控制点数量（越大越精确，但维度越高）
M_Q = 128    # 位置：96-160
M_DQ = 64    # 速度：16-64  
M_TAU = 64   # 力矩：16-64

# 样条阶数（越高越平滑）
DEGREE_Q = 5     # 位置：3 或 5
DEGREE_DQ = 3    # 速度：3
DEGREE_TAU = 5   # 力矩：3 或 5

# 正则化（越大越稳定）
LAMBDA_Q = 1e-6    # 位置：1e-6 到 1e-5
LAMBDA_DQ = 1e-6   # 速度：1e-6
LAMBDA_TAU = 1e-5  # 力矩：1e-6 到 1e-4

# 测试哪条轨迹
SAMPLE_INDEX = 0   # 0 到 2039
```

### Step 3: 运行
```bash
python3 quick_tune.py
```

### Step 4: 查看结果
生成 3 张图在当前目录：
- `position_comparison.png`
- `velocity_comparison.png`
- `torque_comparison.png`

### Step 5: 调整 → 重跑
修改参数 → 再次运行 → 图片自动覆盖

---

## 📊 输出示例

```
[Results]
  Position (q):   RMSE = 0.000714 rad  (0.0409°)  ← 极好！
  Velocity (dq):  RMSE = 0.083130 rad/s           ← 很好
  Torque (tau):   RMSE = 0.779984 Nm              ← 可接受
  Weighted RMSE:  0.250131
  
  Total Dimensions: 1792
```

---

## 💡 常见调整

### 想要更高精度？
```python
M_Q = 160    # 增加控制点
M_TAU = 96   # 力矩也增加
```

### 想要更少维度？
```python
M_Q = 96     # 减少控制点
M_DQ = 16
M_TAU = 16
# 总维度 = 896 dims
```

### 力矩误差太大？
```python
M_TAU = 96        # 增加控制点
DEGREE_TAU = 5    # 用五次样条
LAMBDA_TAU = 1e-5 # 增加正则化
```

### 看到端点振荡？
```python
LAMBDA_Q = 1e-5    # 增加正则化
LAMBDA_TAU = 1e-4  # 力矩更多正则化
```

---

## 📈 图片说明

每张图包含：
- **7 个子图**（每个关节一个）
- **蓝色实线** = 原始轨迹
- **红色虚线** = 重建轨迹
- **橙色阴影** = 差异区域
- **RMSE** = 每个关节的误差

---

## 🎯 目标 RMSE

- **Position**: < 0.001 rad (优秀)
- **Velocity**: < 0.1 rad/s (优秀)
- **Torque**: < 0.8 Nm (良好)

---

## 🔄 典型流程

1. 用默认参数跑一次（Strategy 1 配置）
2. 看哪个物理量误差大
3. 调整对应的 M/Degree/Lambda
4. 重跑，对比结果
5. 重复直到满意
6. 记录最佳配置

---

## ⚡ 更多技巧

详见同目录下 `README.md`

---

**开始调参吧！** 🎛️
