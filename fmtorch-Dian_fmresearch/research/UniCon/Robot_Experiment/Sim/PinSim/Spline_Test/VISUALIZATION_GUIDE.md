# Trajectory Reconstruction Visualization Guide

## 📊 Overview

This guide explains how to visualize and compare original vs reconstructed trajectories using B-Spline compression.

## 🎯 What You Get

**6 comparison plots** showing original vs reconstructed trajectories:

### Strategy 1: Fixed q High-Fidelity (1792 dims)
1. **Position (q)** - 7 joints, each in a subplot
2. **Velocity (dq)** - 7 joints, each in a subplot
3. **Torque (tau)** - 7 joints, each in a subplot

### Strategy 2: 1k Budget (994 dims)
4. **Position (q)** - 7 joints, each in a subplot
5. **Velocity (dq)** - 7 joints, each in a subplot
6. **Torque (tau)** - 7 joints, each in a subplot

Each plot contains:
- **7 subplots** (one per joint)
- **Blue line**: Original trajectory
- **Red dashed line**: Reconstructed trajectory
- **RMSE** displayed for each joint
- **Overall RMSE** in the title

---

## 🚀 Quick Start

### Default: Visualize Sample 0

```bash
cd /home/nvidiapc/Flow_ICLR/fmtorch/research/UniCon/Robot_Experiment/Sim/PinSim/Spline_Test
bash run_reconstruction_visualization.sh
```

### Visualize Different Samples

```bash
# Sample 10
bash run_reconstruction_visualization.sh --sample 10

# Sample 100
bash run_reconstruction_visualization.sh --sample 100
```

### Custom Data File

```bash
bash run_reconstruction_visualization.sh \
  --data /path/to/your/data.npz \
  --sample 5 \
  --out ./my_comparison_results
```

---

## 📈 Results Interpretation

### Example Output (Sample 0)

```
Comparison Summary:
Strategy     Position RMSE   Velocity RMSE   Torque RMSE     Latent Dims 
--------------------------------------------------------------------------------
Strategy 1   0.000714        0.083130        0.779984        1792        
Strategy 2   0.001139        0.159976        1.124002        994         
```

**Key Observations**:
- **Strategy 1** (1792 dims):
  - Excellent position reconstruction (0.0007 rad ≈ 0.04°)
  - Good velocity reconstruction (0.083 rad/s)
  - Decent torque reconstruction (0.78 Nm)

- **Strategy 2** (994 dims, ~44% fewer dims):
  - Still good position (0.0011 rad ≈ 0.06°)
  - Acceptable velocity (0.16 rad/s, ~2x worse)
  - Higher torque error (1.12 Nm, ~1.4x worse)

**Trade-off**: Strategy 2 uses half the dimensions but has slightly higher errors.

---

## 📁 Output Files

After running, you'll get:

```
reconstruction_comparison_TIMESTAMP/
├── strategy1_position_comparison.png   # Strategy 1: Position (q)
├── strategy1_velocity_comparison.png   # Strategy 1: Velocity (dq)
├── strategy1_torque_comparison.png     # Strategy 1: Torque (tau)
├── strategy2_position_comparison.png   # Strategy 2: Position (q)
├── strategy2_velocity_comparison.png   # Strategy 2: Velocity (dq)
└── strategy2_torque_comparison.png     # Strategy 2: Torque (tau)
```

---

## 🎨 Plot Layout

Each plot has a **3×3 grid** layout:

```
┌─────────────┬─────────────┬─────────────┐
│  Joint 1    │  Joint 2    │  Joint 3    │
│  (subplot)  │  (subplot)  │  (subplot)  │
├─────────────┼─────────────┼─────────────┤
│  Joint 4    │  Joint 5    │  Joint 6    │
│  (subplot)  │  (subplot)  │  (subplot)  │
├─────────────┼─────────────┼─────────────┤
│  Joint 7    │             │             │
│  (subplot)  │             │             │
└─────────────┴─────────────┴─────────────┘
```

Each subplot shows:
- **X-axis**: Time steps (0 to T-1)
- **Y-axis**: Physical quantity value
- **Blue solid line**: Original trajectory
- **Red dashed line**: Reconstructed trajectory
- **Title**: Joint name + per-joint RMSE
- **Legend**: "Original" and "Reconstructed"

---

## 🔧 Advanced Usage

### Custom Configurations

If you want to test different configurations:

```bash
python3 visualize_reconstruction.py \
  --data ../Compress_Test/test_complete.npz \
  --sample-idx 0 \
  --out ./custom_results \
  --strategy1-Mq 160 \
  --strategy1-Mdq 48 \
  --strategy1-Mtau 48 \
  --strategy1-deg-q 5 \
  --strategy1-deg-dq 3 \
  --strategy1-deg-tau 5 \
  --strategy2-Mq 96 \
  --strategy2-Mdq 16 \
  --strategy2-Mtau 16 \
  --strategy2-deg-q 3 \
  --strategy2-deg-dq 3 \
  --strategy2-deg-tau 5
```

### Visualize Multiple Samples

```bash
# Create a batch script
for sample in 0 10 50 100 200; do
    bash run_reconstruction_visualization.sh \
      --sample $sample \
      --out ./comparison_sample_$sample
done
```

---

## 📊 What to Look For

### Good Reconstruction
- Original and reconstructed lines **overlap closely**
- **Small RMSE** values (< 0.01 for position)
- **Smooth reconstructed curves** (no artifacts)

### Poor Reconstruction
- Visible **gap** between lines
- **Large RMSE** values
- **Oscillations** or artifacts in reconstructed curves
- **Endpoint effects** (振铃效应)

### Common Patterns
1. **Position (q)** typically reconstructs best (smooth, low frequency)
2. **Velocity (dq)** has moderate errors (higher frequency)
3. **Torque (tau)** hardest to reconstruct (high frequency, transients)

---

## 💡 Tips

### 1. Sample Selection
- **Sample 0**: Quick test
- **Random samples**: Better representation
- **Extreme samples**: Test edge cases

```bash
# Get dataset size
python3 -c "import numpy as np; print(np.load('../Compress_Test/test_complete.npz')['q_log'].shape[0])"

# Visualize random sample
bash run_reconstruction_visualization.sh --sample $RANDOM
```

### 2. Identifying Issues
- **High position error**: Increase `Mq`
- **High velocity error**: Increase `Mdq` or use higher degree
- **High torque error**: Increase `Mtau`, use degree=5, or increase lambda

### 3. Trade-offs
- **More control points (M)** → Better reconstruction, more dimensions
- **Higher degree** → Smoother curves, may overfit
- **Higher lambda** → More stable, may underfit

---

## 🔍 Comparison with Other Methods

### vs DCT-PCA (Compress_Test)
```bash
# Run both and compare
cd ../Compress_Test
python3 PCAK96R64_TestVis.py  # If exists

cd ../Spline_Test
bash run_reconstruction_visualization.sh --sample 0
```

### vs Multihead PCA (Multihead_Compress_Test)
Similar visualization can be done for multihead PCA results.

---

## 📝 Citing in Reports

When using these visualizations in papers/reports:

```
B-Spline compression achieves excellent position reconstruction 
(RMSE = 0.0007 rad) using 1792 latent dimensions (Strategy 1) 
or acceptable reconstruction (RMSE = 0.0011 rad) using only 
994 dimensions (Strategy 2), demonstrating effective 
compression-reconstruction trade-offs.
```

---

## 🆘 Troubleshooting

### Issue: "Data file not found"
**Solution**: Check the data path
```bash
ls -lh ../Compress_Test/test_complete.npz
```

### Issue: "Sample index exceeds dataset size"
**Solution**: Check dataset size
```bash
python3 -c "import numpy as np; d = np.load('../Compress_Test/test_complete.npz'); print(f'Dataset has {d[\"q_log\"].shape[0]} samples')"
```

### Issue: Plots look bad
**Solution**: 
- Increase figure DPI in `visualize_reconstruction.py` (line with `dpi=150`)
- Try different samples
- Check if data is corrupted

---

## 📚 Related Files

- `visualize_reconstruction.py` - Main visualization script
- `run_reconstruction_visualization.sh` - Convenience wrapper
- `regenerate_plots.sh` - Regenerate all strategy comparison plots
- `visualize_results.py` - Visualize sweep results

---

## 🎯 Next Steps

1. **Review the 6 generated plots**
2. **Compare Strategy 1 vs Strategy 2** trade-offs
3. **Try different samples** to verify consistency
4. **Use best configuration** for downstream tasks (e.g., Flow Matching)

---

**Happy Visualizing!** 📊✨
