import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np
import os
from copy import deepcopy
import time
import threading
from datetime import datetime

# ==============================================================================
# Helper functions from vis_utils/open3d_utils.py
# ==============================================================================

def get_mesh_bottle(root, bottle_idx=3):
    """Loads a bottle mesh and applies necessary transformations."""
    path_bottle = os.path.join(root, f'{bottle_idx}', 'models', 'model_normalized.obj')
    if not os.path.exists(path_bottle):
        print(f"Warning: Bottle model not found at {path_bottle}. Using a placeholder.")
        mesh_bottle = o3d.geometry.TriangleMesh.create_cylinder(radius=0.035, height=0.2)
        mesh_bottle.translate([0,0,0.1])
        mesh_bottle.compute_vertex_normals()
        return mesh_bottle

    mesh_bottle = o3d.io.read_triangle_mesh(path_bottle)
    R = mesh_bottle.get_rotation_matrix_from_xyz((np.pi / 2, 0, 0))
    mesh_bottle.rotate(R, center=(0, 0, 0))
    mesh_bottle.translate([0, 0, 0.])

    spec_path = os.path.join(root, f'{bottle_idx}', 'spec.txt')
    if os.path.exists(spec_path):
        with open(spec_path, 'rb') as f:
            spec = f.readlines()
        bottle_height = float(spec[0].decode('utf-8').split(':')[1])
        bottle_width = float(spec[1].decode('utf-8').split(':')[1])
        
        # bottle_rescalling_translation
        vertices_bottle_numpy = np.asarray(mesh_bottle.vertices) # n * 3 
        bottle_width_current = vertices_bottle_numpy[:, 0].max() - vertices_bottle_numpy[:, 0].min()
        bottle_height_current = vertices_bottle_numpy[:, 2].max() - vertices_bottle_numpy[:, 2].min()
        if bottle_width_current > 1e-6 and bottle_height_current > 1e-6:
            vertices_bottle_numpy[:, :2] *= bottle_width/bottle_width_current
            vertices_bottle_numpy[:, 2] *= bottle_height/bottle_height_current
        min_x_bottle = vertices_bottle_numpy[:, 0].min()
        min_z_bottle = vertices_bottle_numpy[:, 2].min()
        mesh_bottle.translate([(-bottle_width/2 - min_x_bottle), 0, -(min_z_bottle)])
    return mesh_bottle

def get_mesh_mug(root, mug_idx=4):
    """Loads a mug mesh and applies necessary transformations."""
    path_mug = os.path.join(root, f'{mug_idx}', 'models', 'model_normalized.obj')
    if not os.path.exists(path_mug):
        print(f"Warning: Mug model not found at {path_mug}. Using a placeholder.")
        mesh_mug = o3d.geometry.TriangleMesh.create_cylinder(radius=0.04, height=0.08, open_cylinder=True)
        mesh_mug.translate([0,0,0.04])
        mesh_mug.compute_vertex_normals()
        return mesh_mug
        
    mesh_mug = o3d.io.read_triangle_mesh(path_mug)
    R = mesh_mug.get_rotation_matrix_from_xyz((np.pi / 2, 0, 0))
    mesh_mug.rotate(R, center=(0, 0, 0))
    
    spec_path = os.path.join(root, f'{mug_idx}', 'spec.txt')
    if os.path.exists(spec_path):
        with open(spec_path, 'rb') as f:
            spec = f.readlines()
        mug_height = float(spec[0].decode('utf-8').split(':')[1])
        mug_width_outer = float(spec[2].decode('utf-8').split(':')[1])
        
        # mug_rescalling_translation
        vertices_mug_numpy = np.asarray(mesh_mug.vertices) # n * 3
        mug_width_current = vertices_mug_numpy[:, 0].max() - vertices_mug_numpy[:, 0].min()
        mug_height_current = vertices_mug_numpy[:, 2].max() - vertices_mug_numpy[:, 2].min()
        if mug_width_current > 1e-6 and mug_height_current > 1e-6:
            vertices_mug_numpy[:, :2] *= mug_width_outer/mug_width_current
            vertices_mug_numpy[:, 2] *= mug_height/mug_height_current
        min_y_mug = vertices_mug_numpy[:, 1].min()
        min_z_mug = vertices_mug_numpy[:, 2].min()
        mesh_mug.translate([0, (-mug_width_outer/2 - min_y_mug), -min_z_mug])
    return mesh_mug


# ==============================================================================
# Main Visualizer Class
# ==============================================================================

class TrajectoryVisualizer:
    def __init__(self, width=1024, height=768, bottle_model_path='./3dmodels/bottles', mug_model_path='./3dmodels/mugs'):
        # Color template
        rgb = np.zeros((10, 3))
        rgb[0, :] = [208, 28, 31]      # fiery red
        rgb[1, :] = [207, 45, 113]     # beetroot purple
        rgb[2, :] = [249, 77, 0]       # tangelo
        rgb[3, :] = [250, 154, 133]    # peach pink
        rgb[4, :] = [247, 208, 0]      # empire yellow
        rgb[5, :] = [253, 195, 198]    # crystal rose
        rgb[6, :] = [57, 168, 69]      # classic green
        rgb[7, :] = [193, 219, 60]     # love bird 
        rgb[8, :] = [75, 129, 191]     # blue perennial 
        rgb[9, :] = [161, 195, 218]    # summer song
        self.rgb = rgb / 255  
        self.water_color_indices = [8, 6, 9, 4, 5]
        self.wine_color_indices = [0, 1, 2, 3, 7]

        self.bottle_model_path = bottle_model_path
        self.mug_model_path = mug_model_path
        
        self.thread_finished = True
        
        # Initialize GUI application
        gui.Application.instance.initialize()

        # Setup window
        self.window = gui.Application.instance.create_window(
            str(datetime.now().strftime('%H%M%S')), width=width, height=height
        )
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(self.window.renderer)
        self.window.add_child(self._scene)

        # Setup camera and lighting
        self._setup_scene()

        # Material
        self.mat = rendering.MaterialRecord()
        self.mat.shader = 'defaultLit'
        self.mat.base_color = [1.0, 1.0, 1.0, 0.9]

        # Initial scene objects
        self.frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        self.mesh_box = o3d.geometry.TriangleMesh.create_box(width=2, height=2, depth=0.03)
        self.mesh_box.translate([-1, -1, -0.03])
        self.mesh_box.paint_uniform_color([222/255, 184/255, 135/255])
        self.mesh_box.compute_vertex_normals()

        self._scene.scene.add_geometry('frame_init', self.frame, self.mat)
        self._scene.scene.add_geometry('box_init', self.mesh_box, self.mat)

    def _setup_scene(self):
        self._scene.scene.camera.look_at(
            [0, 0, 0],       # camera lookat
            [0.7, 0, 0.9],   # camera position
            [0, 0, 1]        # up vector
        )
        self._scene.scene.set_lighting(self._scene.scene.LightingProfile.DARK_SHADOWS, (-0.3, 0.3, -0.9))
        self._scene.scene.set_background([1.0, 1.0, 1.0, 1.0], image=None)

    def _create_assets_for_trajectories(self, trajectories):
        """Creates and colors meshes for each trajectory."""
        self.mesh_bottles = []
        
        color_indices = self.water_color_indices + self.wine_color_indices
        
        for i, traj in enumerate(trajectories):
            mesh_bottle = get_mesh_bottle(root=self.bottle_model_path, bottle_idx=3) # Using a default idx
            mesh_bottle.compute_vertex_normals()

            color_idx = color_indices[i % len(color_indices)]
            
            bottle_normals = np.asarray(mesh_bottle.vertex_normals)
            bottle_colors = np.ones_like(bottle_normals)
            bottle_colors[:, :3] = self.rgb[color_idx]
            mesh_bottle.vertex_colors = o3d.utility.Vector3dVector(bottle_colors)
            
            self.mesh_bottles.append(mesh_bottle)

        # Load a representative mug
        self.mesh_mug = get_mesh_mug(root=self.mug_model_path, mug_idx=4) # Using a default idx
        self.mesh_mug.paint_uniform_color(self.rgb[2] * 0.6)
        self.mesh_mug.compute_vertex_normals()

    def visualize(self, trajectories, mode='video', skip_size=5):
        if not isinstance(trajectories, list):
            trajectories = [trajectories]
        
        self.trajectories = trajectories
        self.skip_size = skip_size
        self._create_assets_for_trajectories(trajectories)
        
        if mode == 'static':
            self._visualize_static()
        elif mode == 'video':
            if self.thread_finished:
                threading.Thread(target=self._visualize_video).start()
        else:
            raise ValueError(f"Unknown visualization mode: {mode}. Use 'static' or 'video'.")

    def _visualize_static(self):
        self.clear_scene()
        self._scene.scene.add_geometry('mug_init', self.mesh_mug, self.mat)

        for traj_idx, traj in enumerate(self.trajectories):
            for step_idx in range(0, len(traj), self.skip_size):
                mesh_bottle_ = deepcopy(self.mesh_bottles[traj_idx])
                T = traj[step_idx]
                mesh_bottle_.transform(T)
                self._scene.scene.add_geometry(f'bottle_{traj_idx}_{step_idx}', mesh_bottle_, self.mat)
    
    def _visualize_video(self):
        self.thread_finished = False
        
        max_length = max(len(traj) for traj in self.trajectories) if self.trajectories else 0

        for step_idx in range(0, max_length, 5): # 5 is the step for video playback
            self.current_bottles = []
            
            for traj_idx, traj in enumerate(self.trajectories):
                if step_idx < len(traj):
                    mesh_bottle_ = deepcopy(self.mesh_bottles[traj_idx])
                    T = traj[step_idx]
                    mesh_bottle_.transform(T)
                    self.current_bottles.append((traj_idx, mesh_bottle_))
            
            self.current_step_idx = step_idx
            gui.Application.instance.post_to_main_thread(self.window, self._update_video_frame)
            time.sleep(0.05)
            
        self.thread_finished = True

    def _update_video_frame(self):
        self.clear_scene()
        self._scene.scene.add_geometry('mug_init', self.mesh_mug, self.mat)
        
        for traj_idx, bottle in self.current_bottles:
            self._scene.scene.add_geometry(f'bottle_{traj_idx}_{self.current_step_idx}', bottle, self.mat)
            
    def clear_scene(self):
        self._scene.scene.clear_geometry()
        self._scene.scene.add_geometry('frame_init', self.frame, self.mat)
        self._scene.scene.add_geometry('box_init', self.mesh_box, self.mat)

    def run(self):
        gui.Application.instance.run()

# ==============================================================================
# Example Usage
# ==============================================================================

if __name__ == '__main__':
    
    def generate_dummy_trajectory(num_steps=100, radius=0.2, height_start=0.3, height_end=0.15):
        """Generates a simple pouring-like trajectory."""
        trajectory = []
        for i in range(num_steps):
            angle = (i / num_steps) * np.pi
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            
            T = np.eye(4)
            
            # Translation
            T[0, 3] = x
            T[1, 3] = y
            T[2, 3] = height_start + (height_end - height_start) * (i / num_steps)
            
            # Rotation (to tilt the bottle)
            tilt_angle = (i / num_steps) * (np.pi / 2)
            rot_y = np.array([
                [np.cos(tilt_angle), 0, np.sin(tilt_angle)],
                [0, 1, 0],
                [-np.sin(tilt_angle), 0, np.cos(tilt_angle)]
            ])
            T[:3, :3] = rot_y
            
            trajectory.append(T)
            
        return np.array(trajectory)

    # Note: This script assumes that you have 3D models in a folder named '3dmodels'
    # in the same directory where you run the script.
    # If not, it will use placeholder geometry.
    if not os.path.exists('3dmodels'):
        print("Warning: '3dmodels' directory not found. The visualizer will use placeholder shapes.")
        print("For full functionality, please provide the 3d model assets.")


    # --- Create Visualizer ---
    # You can specify paths to your 3d models here
    visualizer = TrajectoryVisualizer(
        bottle_model_path='./3dmodels/bottles',
        mug_model_path='./3dmodels/mugs'
    )
    
    # --- Generate Trajectories ---
    traj1 = generate_dummy_trajectory(num_steps=100, radius=0.1)
    traj2 = generate_dummy_trajectory(num_steps=120, radius=0.15, height_start=0.35)

    # --- Visualize ---
    # Call the visualize method with your trajectory data.
    # mode can be 'video' (animated) or 'static' (afterimage).
    visualizer.visualize([traj1, traj2], mode='video')
    
    # --- Run Application ---
    # This will open the window and start the visualization.
    # The call is blocking and will return when the window is closed.
    visualizer.run()
