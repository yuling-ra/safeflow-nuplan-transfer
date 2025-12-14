import open3d as o3d
import numpy as np
import os
import sys
import argparse
from copy import deepcopy

def main(file_path):
    """
    加载保存的轨迹数据并使用Open3D进行可视化。
    """
    # 动态添加路径以找到vis_utils模块
    # 该脚本预计位于 Dian_Experiment 目录中
    module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'MMLfD-Tutorial_Modified'))
    if module_path not in sys.path:
        sys.path.append(module_path)

    try:
        from vis_utils.open3d_utils import get_mesh_bottle, get_mesh_mug
    except ImportError:
        print(f"错误: 无法从 '{module_path}' 导入 'vis_utils'。")
        print("请确保路径正确，并且 __init__.py 文件存在于相应目录中。")
        return

    # --- 1. 加载数据 ---
    if not os.path.exists(file_path):
        print(f"错误: 找不到文件 '{file_path}'")
        return
        
    print(f"正在从 '{file_path}' 加载轨迹...")
    data = np.load(file_path, allow_pickle=True)
    T_seq_raw = data['T_seq_raw']
    T_fm_abs = data['T_fm_abs']
    bottle_idx = data['bottle_idx'].item()
    mug_idx = data['mug_idx'].item()

    print("初始化3D可视化...")

    # --- 2. 加载3D模型 ---
    try:
        model_root_dir = os.path.join(module_path, '3dmodels')
        mesh_bottle = get_mesh_bottle(root=os.path.join(model_root_dir, 'bottles'), bottle_idx=bottle_idx)
        mesh_mug = get_mesh_mug(root=os.path.join(model_root_dir, 'mugs'), mug_idx=mug_idx)
        if mesh_bottle is None or mesh_mug is None:
            raise FileNotFoundError("get_mesh_bottle/mug returned None.")
        mesh_bottle.compute_vertex_normals()
        mesh_mug.compute_vertex_normals()
    except Exception as e:
        print(f"错误: 加载3D模型失败: {e}")
        return

    # --- 3. 创建场景 ---
    geometries = []
    mesh_table = o3d.geometry.TriangleMesh.create_box(width=1.5, height=1.5, depth=0.02)
    mesh_table.translate([-0.75, -0.75, -0.02])
    mesh_table.paint_uniform_color([0.8, 0.6, 0.4])
    mesh_table.compute_vertex_normals()
    geometries.append(mesh_table)

    mesh_mug.paint_uniform_color([0.5, 0.5, 0.5])
    geometries.append(mesh_mug)
    
    # --- 4. 渲染轨迹 ---
    trajectories = [T_seq_raw, T_fm_abs]
    labels = ["Ground Truth", "FM Generated"]
    colors = [[0.1, 0.8, 0.1], [0.9, 0.2, 0.2]] # Green, Red
    skip_size = 40

    print("渲染轨迹...")
    for traj, label, color in zip(trajectories, labels, colors):
        print(f"  - {label} (颜色: {color})")
        for t_idx in range(0, len(traj), skip_size):
            T = traj[t_idx]
            bottle_instance = deepcopy(mesh_bottle)
            bottle_instance.transform(T)
            bottle_instance.paint_uniform_color(color)
            geometries.append(bottle_instance)

    # --- 5. 可视化 ---
    print("启动Open3D窗口...")
    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"轨迹对比: {', '.join(labels)}",
        width=1280,
        height=720,
    )
    print("可视化窗口已关闭。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize saved pouring trajectories.")
    parser.add_argument(
        '--file',
        type=str,
        default='artifacts/pouring_overfit_single/trajectories_for_visualizer.npz',
        help='Path to the .npz file containing the trajectories.'
    )
    args = parser.parse_args()
    
    main(args.file)

