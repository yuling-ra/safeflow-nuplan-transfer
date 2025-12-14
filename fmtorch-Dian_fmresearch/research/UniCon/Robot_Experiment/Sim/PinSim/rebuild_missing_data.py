#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Rebuild missing data (ee_pos_log, ee_pos_des_log, phase0, random_init_yz) 
from existing trajectory data.
"""

import os
import numpy as np
import argparse
import pinocchio as pin

def rebuild_missing_data(input_npz, output_npz=None, urdf_path=None):
    """Rebuild missing end-effector position logs and metadata."""
    
    if output_npz is None:
        base, ext = os.path.splitext(input_npz)
        output_npz = f"{base}_complete{ext}"
    
    print(f"Loading data from: {input_npz}")
    data = np.load(input_npz)
    
    print(f"Arrays in file: {list(data.keys())}")
    
    # Load robot model
    if urdf_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        urdf_path = os.path.join(script_dir, "panda_description", "urdf", "panda_stick.urdf")
        package_dirs = [os.path.join(script_dir, "panda_description")]
    else:
        package_dirs = [os.path.dirname(urdf_path)]
    
    print(f"Loading robot model from: {urdf_path}")
    model, _, _ = pin.buildModelsFromUrdf(urdf_path, package_dirs)
    pin_data = model.createData()
    
    EE_NAME = "panda_tool_tip"
    ee_fid = model.getFrameId(EE_NAME)
    
    # Get existing data
    seeds = data['seeds']
    q_init = data['q_init']
    q_log = data['q_log']
    t_log = data['t_log']
    
    n_trajs = len(seeds)
    N_max = q_log.shape[1]
    
    print(f"\nRebuilding data for {n_trajs} trajectories...")
    
    # Rebuild ee_pos_log by forward kinematics
    print("1. Rebuilding ee_pos_log using forward kinematics...")
    ee_pos_log = np.zeros((n_trajs, N_max, 3))
    
    for i in range(n_trajs):
        for j in range(N_max):
            q = q_log[i, j]
            pin.forwardKinematics(model, pin_data, q)
            pin.updateFramePlacements(model, pin_data)
            ee_pos_log[i, j] = pin_data.oMf[ee_fid].translation.copy()
        
        if (i + 1) % 100 == 0:
            print(f"   Processed {i+1}/{n_trajs} trajectories...")
    
    print("✓ ee_pos_log rebuilt")
    
    # Try to infer trajectory parameters from initial positions
    print("\n2. Inferring trajectory parameters...")
    
    # Get trajectory parameters from first trajectory's initial EE position
    q0 = q_init[0]
    pin.forwardKinematics(model, pin_data, q0)
    pin.updateFramePlacements(model, pin_data)
    p0_ref = pin_data.oMf[ee_fid].translation.copy()
    
    x_plane = p0_ref[0] + 0.25
    cy = p0_ref[1]
    cz = p0_ref[2] + 0.12
    Ay, Az = 0.10, 0.05
    omega, phi = 0.6, 0.0
    
    print(f"   Inferred parameters: x_plane={x_plane:.4f}, cy={cy:.4f}, cz={cz:.4f}")
    
    # Try to infer dt and cyc
    if 'dt' in data:
        dt = float(data['dt'])
    else:
        # Infer from t_log
        dt = np.mean(np.diff(t_log[0, :100]))
    
    if 'cyc' in data:
        cyc = float(data['cyc'])
    else:
        # Infer from total time
        T_total = t_log[0, -1]
        T_cycle = 2.0 * np.pi / omega
        cyc = T_total / T_cycle
    
    print(f"   Inferred dt={dt:.6f}, cyc={cyc:.2f}")
    
    # Rebuild phase0 and random_init_yz
    print("\n3. Inferring phase0 and random_init_yz from initial positions...")
    
    phase0_all = np.zeros(n_trajs)
    random_init_yz_all = np.zeros((n_trajs, 2))
    
    def nearest_phase(y0, z0, cy, cz, Ay, Az, omega, phi, K=1200):
        thetas = np.linspace(0.0, 2.0*np.pi/omega, K, endpoint=False)
        y = cy + Ay*np.sin(omega*thetas)
        z = cz + Az*np.sin(2.0*omega*thetas + phi)
        i = np.argmin((y - y0)**2 + (z - z0)**2)
        return float(thetas[i])
    
    for i in range(n_trajs):
        # Get initial EE position
        p_init = ee_pos_log[i, 0]
        y_init, z_init = p_init[1], p_init[2]
        
        # Find nearest phase
        phase0_all[i] = nearest_phase(y_init, z_init, cy, cz, Ay, Az, omega, phi)
        
        # Store as "random" initial position (it's the landed position)
        random_init_yz_all[i] = [y_init, z_init]
        
        if (i + 1) % 500 == 0:
            print(f"   Processed {i+1}/{n_trajs} trajectories...")
    
    print("✓ phase0 and random_init_yz inferred")
    
    # Rebuild ee_pos_des_log using trajectory function
    print("\n4. Rebuilding ee_pos_des_log from trajectory equations...")
    
    ee_pos_des_log = np.zeros((n_trajs, N_max, 3))
    
    for i in range(n_trajs):
        phase0 = phase0_all[i]
        for j in range(N_max):
            t = t_log[i, j]
            tau = t + phase0
            
            y = cy + Ay * np.sin(omega * tau)
            z = cz + Az * np.sin(2.0 * omega * tau + phi)
            
            ee_pos_des_log[i, j] = [x_plane, y, z]
        
        if (i + 1) % 100 == 0:
            print(f"   Processed {i+1}/{n_trajs} trajectories...")
    
    print("✓ ee_pos_des_log rebuilt")
    
    # Get N_per_traj
    if 'N_per_traj' in data:
        N_per_traj = data['N_per_traj']
    else:
        # Infer from non-zero elements in t_log
        N_per_traj = np.zeros(n_trajs, dtype=int)
        for i in range(n_trajs):
            N_per_traj[i] = np.sum(t_log[i] > 0)
    
    # Compile all data
    print(f"\n💾 Saving complete data to {output_npz}...")
    
    save_dict = {
        'seeds': seeds,
        'q_init': q_init,
        'dq_init': data['dq_init'],
        'q_log': q_log,
        'dq_log': data['dq_log'],
        'tau_log': data['tau_log'],
        't_log': t_log,
        'ee_pos_log': ee_pos_log,
        'ee_pos_des_log': ee_pos_des_log,
        'phase0': phase0_all,
        'random_init_yz': random_init_yz_all,
        'N_per_traj': N_per_traj,
        'dt': dt,
        'cyc': cyc,
    }
    
    np.savez_compressed(output_npz, **save_dict)
    
    print("✓ Complete data saved")
    
    # Verify
    print("\n🔍 Verifying complete file...")
    verified = np.load(output_npz)
    print(f"✓ All arrays present: {list(verified.keys())}")
    print(f"\n📊 Summary:")
    print(f"   Trajectories: {len(verified['seeds'])}")
    print(f"   Seed range: {verified['seeds'][0]} to {verified['seeds'][-1]}")
    print(f"   dt: {verified['dt']}")
    print(f"   cyc: {verified['cyc']}")
    
    verified.close()
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Rebuild missing data in NPZ file')
    parser.add_argument('input_file', type=str,
                        help='Path to repaired NPZ file')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path (default: input_file_complete.npz)')
    parser.add_argument('--urdf', type=str, default=None,
                        help='Path to URDF file (default: auto-detect)')
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"❌ File not found: {args.input_file}")
        return
    
    success = rebuild_missing_data(args.input_file, args.output, args.urdf)
    
    if success:
        print("\n✅ Rebuild completed successfully!")
    else:
        print("\n❌ Rebuild failed!")

if __name__ == "__main__":
    main() 