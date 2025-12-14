#!/usr/bin/env python
# coding: utf-8

# In[30]:


# Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import os
from scipy.spatial import cKDTree



# In[31]:


def load_track_data(track_name, base_path="../racetrack-database"):
    """
    Load track data
    
    Args:
        track_name: Track name (e.g., 'Silverstone')
        base_path: Database path
    
    Returns:
        track_data: Track centerline and width data
        raceline_data: Racing line data (if exists)
    """
    track_file = os.path.join(base_path, "tracks", f"{track_name}.csv")
    raceline_file = os.path.join(base_path, "racelines", f"{track_name}.csv")
    
    # Load track data
    if not os.path.exists(track_file):
        raise FileNotFoundError(f"Track file not found: {track_file}")
    
    track_data = pd.read_csv(track_file, comment='#')
    track_data.columns = ['x_m', 'y_m', 'w_tr_right_m', 'w_tr_left_m']
    
    # Load racing line data (if exists)
    raceline_data = None
    if os.path.exists(raceline_file):
        raceline_data = pd.read_csv(raceline_file, comment='#')
        raceline_data.columns = ['x_m', 'y_m']
        # Ensure raceline has the same number of points as centerline for simplicity
        if len(raceline_data) != len(track_data):
            # Interpolate raceline data
            path_x = raceline_data['x_m'].values
            path_y = raceline_data['y_m'].values
            distance = np.cumsum(np.sqrt(np.diff(path_x, prepend=path_x[0])**2 + np.diff(path_y, prepend=path_y[0])**2))
            f_x = interp1d(distance, path_x, kind='cubic', fill_value="extrapolate")
            f_y = interp1d(distance, path_y, kind='cubic', fill_value="extrapolate")

            center_x = track_data['x_m'].values
            center_y = track_data['y_m'].values
            center_dist = np.cumsum(np.sqrt(np.diff(center_x, prepend=center_x[0])**2 + np.diff(center_y, prepend=center_y[0])**2))
            
            new_raceline_x = f_x(center_dist)
            new_raceline_y = f_y(center_dist)
            raceline_data = pd.DataFrame({'x_m': new_raceline_x, 'y_m': new_raceline_y})

    return track_data, raceline_data

def calculate_track_boundaries(track_data):
    """
    Calculate track boundary points
    
    Args:
        track_data: Data containing centerline and width information
    
    Returns:
        left_boundary: Left boundary points
        right_boundary: Right boundary points
        normals: Normal vectors for each centerline point
    """
    x = track_data['x_m'].values
    y = track_data['y_m'].values
    w_right = track_data['w_tr_right_m'].values
    w_left = track_data['w_tr_left_m'].values
    
    # Calculate tangent direction (along track direction)
    dx = np.gradient(x)
    dy = np.gradient(y)
    
    # Normalize tangent vectors
    tangent_length = np.sqrt(dx**2 + dy**2)
    tangent_length = np.where(tangent_length == 0, 1e-10, tangent_length)
    dx_norm = dx / tangent_length
    dy_norm = dy / tangent_length
    
    # Calculate normal vectors (perpendicular to track)
    nx = -dy_norm  # Normal vector x component
    ny = dx_norm   # Normal vector y component
    
    # Calculate boundary points
    left_boundary_x = x + w_left * nx
    left_boundary_y = y + w_left * ny
    right_boundary_x = x - w_right * nx
    right_boundary_y = y - w_right * ny
    
    left_boundary = np.column_stack([left_boundary_x, left_boundary_y])
    right_boundary = np.column_stack([right_boundary_x, right_boundary_y])
    normals = np.column_stack([nx, ny])
    
    return left_boundary, right_boundary, normals

# Load data for Nuerburgring
track_name = 'Nuerburgring'
track_data, raceline_data = load_track_data(track_name, base_path="../racetrack-database")
left_boundary, right_boundary, normals = calculate_track_boundaries(track_data)

# ----------------------------
# Calculate Track Segment Lengths (2% - 23%)
# ----------------------------
def calculate_path_length(x_coords, y_coords):
    """Calculate the total length of a path given x,y coordinates"""
    distances = np.sqrt(np.diff(x_coords)**2 + np.diff(y_coords)**2)
    return np.sum(distances)

# Define the segment we'll be using (2% to 23%)
start_percent = 0.02
end_percent = 0.23
total_points = len(track_data)
start_idx = int(start_percent * total_points)
end_idx = int(end_percent * total_points)

print(f"Track segment: {start_percent*100:.1f}% to {end_percent*100:.1f}% of full track")
print(f"Point indices: {start_idx} to {end_idx} (out of {total_points} total points)")

# Extract the segment coordinates
centerline_segment_x = track_data['x_m'].iloc[start_idx:end_idx].values
centerline_segment_y = track_data['y_m'].iloc[start_idx:end_idx].values

# Calculate centerline segment length
centerline_length = calculate_path_length(centerline_segment_x, centerline_segment_y)
print(f"\nCenterline segment length: {centerline_length:.2f} meters")

# Calculate raceline segment length if available
if raceline_data is not None:
    raceline_segment_x = raceline_data['x_m'].iloc[start_idx:end_idx].values
    raceline_segment_y = raceline_data['y_m'].iloc[start_idx:end_idx].values
    raceline_length = calculate_path_length(raceline_segment_x, raceline_segment_y)
    print(f"Raceline segment length: {raceline_length:.2f} meters")
    print(f"Length difference (raceline vs centerline): {raceline_length - centerline_length:.2f} meters")
    print(f"Raceline is {'shorter' if raceline_length < centerline_length else 'longer'} by {abs(raceline_length - centerline_length):.2f}m")
else:
    print("Raceline data not available")

# Visualize the segment we'll be using
plt.figure(figsize=(12, 8))
plt.plot(track_data['x_m'], track_data['y_m'], 'lightblue', alpha=0.5, label='Full Track Centerline')
plt.plot(centerline_segment_x, centerline_segment_y, 'b-', linewidth=2, label=f'Centerline Segment ({centerline_length:.0f}m)')

if raceline_data is not None:
    plt.plot(raceline_data['x_m'], raceline_data['y_m'], 'lightcoral', alpha=0.5, label='Full Track Raceline')
    plt.plot(raceline_segment_x, raceline_segment_y, 'r-', linewidth=2, label=f'Raceline Segment ({raceline_length:.0f}m)')

# Highlight start and end points
plt.scatter(centerline_segment_x[0], centerline_segment_y[0], c='green', s=100, label='Start Point', zorder=5)
plt.scatter(centerline_segment_x[-1], centerline_segment_y[-1], c='red', s=100, label='End Point', zorder=5)

plt.axis('equal')
plt.legend()
plt.title(f'{track_name} Track Segment ({start_percent*100:.1f}% - {end_percent*100:.1f}%)')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.grid(True, alpha=0.3)
plt.show()


# In[32]:


# ----------------------------
# Parameters and Vehicle Dynamics
# ----------------------------
state_steps = 100
control_steps = state_steps - 1

# Control limits（保持你的参数名与数值）
a_max = 30.0    # 最大纵向加速度 [m/s^2]
a_min = -30.0   # 最大纵向减速度(制动) [m/s^2]
delta_max = 0.4 # 最大转角 [rad] (~23°)

# Fixed dt value (no longer optimized)
dt_fixed = 0.2  # 固定步长 [s]

# —— 新增的合理车辆参数与限幅（不改你现有参数名，仅新增）——
L_wb = 3.6            # 轴距 [m]（F1 量级）
deltadot_max = 6.0    # 转角角速度上限 [rad/s]
v_top = 95.0          # 最高车速上限 [m/s]（~342 km/h，数值稳定用）

def a_long_limits(v):
    """速度相关的纵向加速/制动能力上限（经验简化版，F1 量级）
    返回: (a_eng_max, a_brk_max)，分别为加速上限与制动上限的正值
    """
    a_eng = max(0.0, 18.0 - 0.004 * (v**2))   # 动力随速衰减，低速~1.8 g
    a_brk = min(45.0, 20.0 + 0.01 * (v**2))   # 刹车随速略升，封顶~4.6 g
    return a_eng, a_brk

def simulate_vehicle(x0, u_seq, dt, state_steps=100):
    """
    基于“简化自行车模型”的离散推进（保持原状态维度与返回形状不变）:
      状态: [x, y, theta, v, r]，其中 r = (v/L_wb)*tan(delta) 为“实时计算”的偏航角速度
      控制: [a, delta]，对 delta 施加角速度限幅，对 a 施加速度相关的纵向限幅

    离散更新:
      r_k      = (v_k / L_wb) * tan(delta_k_eff)
      x_{k+1}  = x_k + dt * v_k * cos(theta_k)
      y_{k+1}  = y_k + dt * v_k * sin(theta_k)
      theta_{k+1} = theta_k + dt * r_k
      v_{k+1}  = clip(v_k + dt * a_k_eff, 0, v_top)
      r_{k+1}  = (v_{k+1} / L_wb) * tan(delta_k_eff)   # 仅作为记录输出

    返回:
      states: 形状 (state_steps, 5) 的数组
    """
    control_steps = state_steps - 1
    states = np.zeros((state_steps, 5), dtype=float)
    states[0] = x0

    # 复制以避免原始序列被外部修改
    u_seq = np.array(u_seq, dtype=float).copy()

    # 基础幅度限幅（保留你原有的 a/delta 外层限制）
    u_seq[:, 0] = np.clip(u_seq[:, 0], a_min, a_max)
    u_seq[:, 1] = np.clip(u_seq[:, 1], -delta_max, delta_max)

    # 转角速率限幅需要“上一时刻实际应用的转角”
    delta_prev = float(np.clip(u_seq[0, 1], -delta_max, delta_max))

    for k in range(control_steps):
        x, y, theta, v, _ = states[k]
        a_cmd, delta_cmd = u_seq[k]

        # 1) 转角角速度限幅（先按速率限幅，再确保幅度限幅）
        max_step = deltadot_max * dt
        delta_eff = np.clip(delta_cmd, delta_prev - max_step, delta_prev + max_step)
        delta_eff = float(np.clip(delta_eff, -delta_max, delta_max))

        # 2) 纵向加减速“速度相关”的二次限幅
        a_eng_max, a_brk_max = a_long_limits(v)
        a_upper = min(a_max, a_eng_max)     # 上界取更严格者
        a_lower = max(a_min, -a_brk_max)    # 下界取更严格者（注意制动为负）
        a_eff = float(np.clip(a_cmd, a_lower, a_upper))

        # 3) 基于简化自行车模型计算 r（非积分状态，而是由 delta 派生）
        r = (v / L_wb) * np.tan(delta_eff)

        # 4) 状态推进
        x_next = x + dt * v * np.cos(theta)
        y_next = y + dt * v * np.sin(theta)
        theta_next = theta + dt * r
        v_next = np.clip(v + dt * a_eff, 0.0, v_top)
        r_next = (v_next / L_wb) * np.tan(delta_eff)  # 仅作记录

        states[k + 1] = [x_next, y_next, theta_next, v_next, r_next]

        # 更新上一时刻实际应用的转角
        delta_prev = delta_eff

    return states


# In[33]:


# ----------------------------
# Cost Function for Trajectory Optimization
# ----------------------------

def find_closest_point_index(point, path):
    """Find the index of the closest point on a path to a given point."""
    path_points = path[['x_m', 'y_m']].values
    distances = np.linalg.norm(path_points - point, axis=1)
    return np.argmin(distances)

def make_index_schedules(centerline_df, raceline_df, N_states):
    """为中心线(走廊约束)与参考线(跟随)分别生成均匀索引表"""
    M_c = len(centerline_df)
    idx_c = np.linspace(0, M_c - 1, N_states).round().astype(int)
    if raceline_df is None:
        idx_r = idx_c
    else:
        M_r = len(raceline_df)
        idx_r = np.linspace(0, M_r - 1, N_states).round().astype(int)
    return idx_c, idx_r

def lane_barrier_penalty_nn(positions, SEG, margin=0.6, weight=2e6, p=2):
    """
    最近邻中心线走廊惩罚（越界 -> 铰链^p，默认 p=2；可设 p=4 更硬）。
    允许区间：[-w_right+margin,  w_left-margin]
    """
    # 最近邻中心线索引
    _, idx = SEG['kdtree'].query(positions)  # shape: (N,)
    n = SEG['normals'][idx]
    c = SEG['center_xy'][idx]
    wL = SEG['w_left'][idx]
    wR = SEG['w_right'][idx]
    # 侧向偏移（沿法向）
    offset = np.sum((positions - c) * n, axis=1)
    # 越界量（>0 表示违反）
    v_left  = np.maximum(0.0, offset - (wL - margin))
    v_right = np.maximum(0.0, (-wR + margin) - offset)
    # 惩罚
    if p == 2:
        pen = np.sum(v_left*v_left + v_right*v_right)
    else:
        pen = np.sum(v_left**p + v_right**p)
    return weight * pen, idx

def progress_monotonic_penalty(idx, SEG, weight=2e4):
    """
    弧长单调性惩罚：鼓励 s_{k+1} >= s_k。
    用最近邻索引把每帧映射到弧长 s(idx)，对负的增量加铰链^2。
    """
    s = SEG['s'][idx]
    ds = np.diff(s)
    back = np.maximum(0.0, -ds)  # 只罚“倒退”
    return weight * np.sum(back*back)

def heading_align_penalty(theta, idx, SEG, weight=0.2):
    """
    车头朝向对齐中心线切向的小惩罚（避免贴墙横滑）。
    """
    psi = SEG['psi'][idx]
    # wrap 到 [-pi, pi]
    d = (theta - psi + np.pi) % (2*np.pi) - np.pi
    return weight * np.sum(d*d)

def raceline_progress_reward(positions, raceline_df, weight=50.0):
    """
    沿赛道线的前进奖励项：计算在赛道线上的前进距离
    """
    if raceline_df is None:
        return 0.0
    
    raceline_xy = raceline_df[['x_m', 'y_m']].values
    
    # 为每个位置找到最近的赛道线点
    distances = np.linalg.norm(raceline_xy[np.newaxis, :, :] - positions[:, np.newaxis, :], axis=2)
    closest_indices = np.argmin(distances, axis=1)
    
    # 计算沿赛道线的前进进度（索引增加 = 前进）
    progress = np.diff(closest_indices.astype(float))
    # 只奖励正向前进，忽略负向（可能由于噪声）
    forward_progress = np.sum(np.maximum(0, progress))
    
    return -weight * forward_progress  # 负号因为我们要最小化成本

def movement_penalty(positions, weight=100.0):
    """
    不移动惩罚：惩罚总移动距离过小
    """
    total_distance = 0.0
    for i in range(1, len(positions)):
        total_distance += np.linalg.norm(positions[i] - positions[i-1])
    
    # 期望的最小移动距离（基于轨迹长度和时间）
    expected_min_distance = 500.0  # 期望至少移动500米
    
    if total_distance < expected_min_distance:
        penalty = (expected_min_distance - total_distance) ** 2
        return weight * penalty
    return 0.0

def raceline_guidance_cost(positions, raceline_df, weight=10.0):
    """
    强化赛道线引导：鼓励跟随赛道线而不是中心线
    """
    if raceline_df is None:
        return 0.0
    
    raceline_xy = raceline_df[['x_m', 'y_m']].values
    
    # 计算到赛道线的距离
    distances = []
    for pos in positions:
        dist_to_raceline = np.min(np.linalg.norm(raceline_xy - pos, axis=1))
        distances.append(dist_to_raceline)
    
    return weight * np.sum(np.array(distances) ** 2)

def forward_distance_reward(positions, weight=200.0):
    """
    前进距离奖励：车辆前进得越多，成本越低
    基于起点到终点的直线距离以及实际行驶距离来计算奖励
    """
    if len(positions) < 2:
        return 0.0
    
    start_pos = positions[0]
    end_pos = positions[-1]
    
    # 计算起点到终点的直线距离（主要前进方向）
    straight_line_distance = np.linalg.norm(end_pos - start_pos)
    
    # 计算总行驶距离
    total_travel_distance = 0.0
    for i in range(1, len(positions)):
        total_travel_distance += np.linalg.norm(positions[i] - positions[i-1])
    
    # 前进效率：直线距离与总行驶距离的比率
    if total_travel_distance > 0:
        efficiency = straight_line_distance / total_travel_distance
        # 奖励既要距离长又要效率高
        forward_reward = straight_line_distance * (0.5 + 0.5 * efficiency)
    else:
        forward_reward = 0.0
    
    # 返回负值（因为我们最小化成本，所以奖励应该是负的）
    return -weight * forward_reward

def cumulative_progress_reward(positions, centerline_df, weight=150.0):
    """
    累积前进奖励：基于在赛道上的累积前进距离
    """
    if len(positions) < 2:
        return 0.0
    
    center_xy = centerline_df[['x_m', 'y_m']].values
    
    # 为每个位置找到最近的中心线点索引
    cumulative_progress = 0.0
    
    for i in range(len(positions)):
        distances = np.linalg.norm(center_xy - positions[i], axis=1)
        closest_idx = np.argmin(distances)
        
        # 使用索引作为进度指标（更大的索引表示更远的前进）
        cumulative_progress += closest_idx
    
    # 平均进度
    avg_progress = cumulative_progress / len(positions)
    
    # 返回负值作为奖励
    return -weight * avg_progress / len(center_xy)  # 标准化到0-1范围

def cost_function(vars_flat, x0, centerline, raceline, normals):
    dt = dt_fixed
    u_seq = vars_flat.reshape((control_steps, 2))
    states = simulate_vehicle(x0, u_seq, dt, state_steps)

    # 轨迹位置/速度/朝向
    pos   = states[:, :2]
    v     = states[:, 3]
    theta = states[:, 2]

    # 参考线
    ref_df = raceline if raceline is not None else centerline
    ref_xy = ref_df[['x_m','y_m']].values
    # 依据状态步长在参考线上均匀采样（仅用于“贴线”项）
    M_r = len(ref_df)
    idx_r = np.linspace(0, M_r - 1, state_steps).round().astype(int)

    # ---- 权重（重新调整以促进前进）----
    w_track    = 3.0    # 参考线跟随权重
    w_barrier  = 1.0    # 走廊惩罚外层倍率
    w_energy   = 0.005  # 进一步降低能量惩罚
    w_smooth   = 0.5    # 进一步降低平滑性要求
    w_speed    = 0.3    # 速度要求
    w_terminal = 50.0   # 终端到达权重
    w_prog     = 1.5    # 弧长单调项权重
    w_head     = 0.3    # 朝向对齐项
    w_movement = 0.5    # 不移动惩罚权重
    w_raceline_progress = 1.0  # 赛道线前进奖励权重
    w_raceline_guide = 0.5     # 赛道线引导权重
    w_forward_distance = 2.0   # 新增：前进距离奖励权重（重要）
    w_cumulative_progress = 1.5  # 新增：累积前进奖励权重（重要）

    # 1) 参考线贴合
    ref_err = pos - ref_xy[idx_r]
    cost_track = np.sum(ref_err[:,0]**2 + ref_err[:,1]**2)

    # 2) 新的走廊惩罚（最近邻）
    cost_corridor, idx_nn = lane_barrier_penalty_nn(
        positions=pos, SEG=SEG, margin=0.6, weight=2e6, p=2
    )

    # 3) 控制能量 & 平滑
    cost_energy = np.sum(u_seq**2)
    cost_smooth = np.sum(np.diff(u_seq, axis=0)**2)

    # 4) 速度惩罚（鼓励更高速度）
    target_speed = 25.0  # 提高目标速度
    cost_speed = np.sum((np.maximum(0.0, target_speed - v))**2)

    # 5) 终端到达（对参考线末端）
    target = ref_xy[-1]
    cost_terminal = np.sum((states[-1, :2] - target)**2)

    # 6) 弧长单调性（防倒退/抄近道）
    cost_prog = progress_monotonic_penalty(idx_nn, SEG, weight=2e4)

    # 7) 朝向对齐（小权重，抑制“横着挤墙”）
    cost_head = heading_align_penalty(theta, idx_nn, SEG, weight=0.2)

    # 8) 新增：不移动惩罚
    cost_movement = movement_penalty(pos, weight=100.0)

    # 9) 新增：赛道线前进奖励
    cost_raceline_progress = raceline_progress_reward(pos, raceline, weight=50.0)

    # 10) 新增：赛道线引导
    cost_raceline_guide = raceline_guidance_cost(pos, raceline, weight=10.0)

    # 11) 新增：前进距离奖励（车辆前进越多，成本越低）
    cost_forward_distance = forward_distance_reward(pos, weight=200.0)

    # 12) 新增：累积前进奖励（基于在赛道上的累积前进）
    cost_cumulative_progress = cumulative_progress_reward(pos, centerline, weight=150.0)

    total_cost = (w_track * cost_track +
                  w_barrier * cost_corridor +
                  w_energy * cost_energy +
                  w_smooth * cost_smooth +
                  w_speed * cost_speed +
                  w_terminal * cost_terminal +
                  w_prog * cost_prog +
                  w_head * cost_head +
                  w_movement * cost_movement +
                  w_raceline_progress * cost_raceline_progress +
                  w_raceline_guide * cost_raceline_guide +
                  w_forward_distance * cost_forward_distance +
                  w_cumulative_progress * cost_cumulative_progress)
    
    return total_cost




# In[34]:


# ----------------------------
# Data Generation Loop
# ----------------------------
num_trajectories = 1 # Generate 5 trajectories for demonstration
trajectories = []
save_filename = 'Track_MPC_trajectories.npy'

# Since we only optimize for control_steps steps, total decision variables: control_steps * 2
bounds = []
for _ in range(control_steps):
    bounds.append((a_min, a_max))         # for a_k
    bounds.append((-delta_max, delta_max)) # for delta_k

# Use only a segment of the track (2% to 25%)
start_percent = 0.02
end_percent = 0.25
total_points = len(track_data)
start_idx = int(start_percent * total_points)
end_idx = int(end_percent * total_points)

# Extract track segment
track_segment = track_data.iloc[start_idx:end_idx].reset_index(drop=True)
raceline_segment = raceline_data.iloc[start_idx:end_idx].reset_index(drop=True) if raceline_data is not None else None
left_boundary_segment = left_boundary[start_idx:end_idx]
right_boundary_segment = right_boundary[start_idx:end_idx]
normals_segment = normals[start_idx:end_idx]

centerline_points = track_segment[['x_m', 'y_m']].values

def prepare_segment_geometry(track_segment, normals_segment):
    """预计算赛段几何（KDTree、弧长、切向角）。"""
    center_xy = track_segment[['x_m','y_m']].values
    # 弧长
    ds = np.sqrt(np.sum(np.diff(center_xy, axis=0)**2, axis=1))
    s = np.concatenate([[0.0], np.cumsum(ds)])
    # 切向（供朝向对齐）
    dx = np.gradient(center_xy[:,0])
    dy = np.gradient(center_xy[:,1])
    psi = np.arctan2(dy, dx)  # 中心线切向角
    # KDTree
    kdtree = cKDTree(center_xy)
    # 宽度
    w_left  = track_segment['w_tr_left_m' ].values
    w_right = track_segment['w_tr_right_m'].values
    # 法向（单位化以防数值误差）
    nrm = normals_segment / (np.linalg.norm(normals_segment, axis=1, keepdims=True) + 1e-12)
    return dict(center_xy=center_xy, s=s, psi=psi, kdtree=kdtree,
                w_left=w_left, w_right=w_right, normals=nrm)

SEG = prepare_segment_geometry(track_segment, normals_segment)



for i in range(num_trajectories):
    print(f"Generating trajectory {i+1}/{num_trajectories}...")
    
    # Randomize start position laterally
    start_point = centerline_points[0]
    start_normal = normals_segment[0]
    random_offset = np.random.uniform(-track_segment['w_tr_right_m'].iloc[0] * 0.8, 
                                     track_segment['w_tr_left_m'].iloc[0] * 0.8)
    start_pos = start_point + random_offset * start_normal

    # Initial state
    dx = centerline_points[1, 0] - centerline_points[0, 0]
    dy = centerline_points[1, 1] - centerline_points[0, 1]
    initial_theta = np.arctan2(dy, dx)
    initial_speed = 5.0  # 给初始速度，避免从静止开始
    x0 = np.array([start_pos[0], start_pos[1], initial_theta, initial_speed, 0.0]) # x, y, theta, v, r

    # 改进的初始控制序列：更强的前进驱动
    vars0 = np.zeros(control_steps * 2)
    u0 = vars0.reshape(control_steps, 2)
    
    # 前半段给予持续加速，后半段准备减速
    mid_point = control_steps // 2
    u0[:mid_point, 0] = 15.0    # 前半段强力加速
    u0[mid_point:, 0] = 5.0     # 后半段轻微加速
    
    # 基于赛道线计算合理的转向角度
    if raceline_segment is not None:
        raceline_xy = raceline_segment[['x_m', 'y_m']].values
        for i in range(min(control_steps, len(raceline_xy)-1)):
            if i < len(raceline_xy)-1:
                dx_r = raceline_xy[i+1, 0] - raceline_xy[i, 0]
                dy_r = raceline_xy[i+1, 1] - raceline_xy[i, 1]
                target_heading = np.arctan2(dy_r, dx_r)
                # 简单的转向控制：朝向目标方向
                heading_diff = target_heading - initial_theta
                heading_diff = (heading_diff + np.pi) % (2*np.pi) - np.pi
                u0[i, 1] = np.clip(heading_diff * 0.1, -delta_max*0.5, delta_max*0.5)
    
    vars0 = u0.ravel()

    # 仅 bounds -> L-BFGS-B 更稳
    res = minimize(
        cost_function, vars0,
        args=(x0, track_segment, raceline_segment, normals_segment),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 1500, 'ftol': 1e-6}
    )



    # Extract control sequence (dt is fixed)
    u_opt = res.x.reshape((control_steps, 2))
    controls_padded = np.vstack([u_opt, [0, 0]]) # Pad for same length as states

    # Simulate final trajectory with fixed dt
    states = simulate_vehicle(x0, u_opt, dt_fixed, state_steps)
    
    print(f"Fixed dt for trajectory {i+1}: {dt_fixed:.4f} seconds")

    trajectories.append({
        'states': states, 
        'controls': controls_padded,
        'dt_fixed': dt_fixed,
        'start': start_pos,
        'target': centerline_points[-1]
    })

# np.save(save_filename, trajectories, allow_pickle=True)
# print(f"Trajectories generation completed and saved to {save_filename}")


# In[35]:


# ----------------------------
# Visualization and Consistency Check
# ----------------------------
if not trajectories:
    try:
        trajectories = np.load(save_filename, allow_pickle=True)
        print(f"Loaded {len(trajectories)} trajectories from {save_filename}")
    except FileNotFoundError:
        print("No trajectories to visualize. Please run the generation cell first.")

if trajectories:
    # 1. Visualize the generated trajectory on the track
    idx = np.random.choice(len(trajectories))
    traj = trajectories[idx]
    states = traj['states']
    controls = traj['controls']

    plt.figure(figsize=(12, 9))
    plt.plot(track_segment['x_m'], track_segment['y_m'], 'b--', label='Centerline')
    if raceline_segment is not None:
        plt.plot(raceline_segment['x_m'], raceline_segment['y_m'], 'g:', label='Raceline')
    plt.plot(left_boundary_segment[:, 0], left_boundary_segment[:, 1], 'k-', linewidth=1)
    plt.plot(right_boundary_segment[:, 0], right_boundary_segment[:, 1], 'k-', linewidth=1)
    
    plt.plot(states[:, 0], states[:, 1], 'r.-', label='Generated Trajectory')
    plt.scatter(states[0, 0], states[0, 1], c='g', s=100, label='Start', zorder=5)
    plt.scatter(states[-1, 0], states[-1, 1], c='m', s=100, label='End', zorder=5)

    plt.title(f'Generated Trajectory vs. Track ({track_name})')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 2. Visualize the generated actions
    dt_used = traj['dt_fixed']
    time_steps = np.arange(state_steps) * dt_used
    print(f"Fixed dt for this trajectory: {dt_used:.4f} seconds")
    print(f"Total trajectory time: {time_steps[-1]:.2f} seconds")
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axs[0].plot(time_steps, controls[:, 0], 'o-')
    axs[0].set_ylabel('Acceleration [m/s^2]')
    axs[0].set_title('Longitudinal Acceleration')
    axs[0].grid(True)

    axs[1].plot(time_steps, controls[:, 1], 'o-')
    axs[1].set_xlabel('Time [s]')
    axs[1].set_ylabel('Steering Angle [rad]')
    axs[1].set_title('Steering Angle')
    axs[1].grid(True)
    plt.tight_layout()
    plt.show()

    # 3. Consistency check: Re-simulate with actions and compare states
    x0_check = states[0]
    u_seq_check = controls[:-1, :] # Use the first N-1 controls
    states_resimulated = simulate_vehicle(x0_check, u_seq_check, dt_used, state_steps)

    plt.figure(figsize=(12, 9))
    plt.plot(states[:, 0], states[:, 1], 'r.-', label='Original Generated States')
    plt.plot(states_resimulated[:, 0], states_resimulated[:, 1], 'g--', label='Re-simulated States')
    plt.title('Consistency Check: Original vs. Re-simulated Trajectory')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Calculate the maximum position error
    pos_error = np.linalg.norm(states[:, :2] - states_resimulated[:, :2], axis=1)
    print(f"Maximum position error between original and re-simulated trajectory: {np.max(pos_error):.4f} m")

