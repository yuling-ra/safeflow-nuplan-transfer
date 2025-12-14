# Quick Tuning Tool

## 🎯 Purpose

Fast, manual, interactive tuning of B-Spline compression parameters with instant visual feedback.

## 🚀 Quick Start

### 1. Edit Parameters

Open `quick_tune.py` and modify the configuration section (lines 24-45):

```python
# Control points (M)
M_Q = 128    # Position control points
M_DQ = 64    # Velocity control points
M_TAU = 64   # Torque control points

# Spline degree
DEGREE_Q = 5     # Position degree
DEGREE_DQ = 3    # Velocity degree
DEGREE_TAU = 5   # Torque degree

# Regularization (lambda)
LAMBDA_Q = 1e-6    # Position regularization
LAMBDA_DQ = 1e-6   # Velocity regularization
LAMBDA_TAU = 1e-5  # Torque regularization

# Sample to visualize
SAMPLE_INDEX = 0   # Change to test different trajectories
```

### 2. Run

```bash
cd Quick_Tuning
python3 quick_tune.py
```

### 3. View Results

Three plots are generated in the same directory:
- `position_comparison.png` - Position (q)
- `velocity_comparison.png` - Velocity (dq)
- `torque_comparison.png` - Torque (tau)

Each plot shows:
- **7 subplots** (one per joint)
- **Blue line**: Original trajectory
- **Red dashed line**: Reconstructed trajectory
- **Orange shading**: Difference between original and reconstructed
- **RMSE** for each joint and overall

### 4. Iterate

Edit parameters → Run again → Previous plots are automatically overwritten

---

## 📊 Output Example

```
================================================================================
B-Spline Quick Tuning Tool
================================================================================

[Configuration]
  Data: ../../Compress_Test/test_complete.npz
  Sample: 0

  Control Points (M):
    q:   128  ( 896 dims)
    dq:   64  ( 448 dims)
    tau:  64  ( 448 dims)
    ----------------------
    Total:    1792 dims

  Spline Degree:
    q:   5
    dq:  3
    tau: 5

  Regularization (λ):
    q:   1e-06
    dq:  1e-06
    tau: 1e-05

[Results]
  Position (q):   RMSE = 0.000714 rad  (0.0409°)
  Velocity (dq):  RMSE = 0.083130 rad/s
  Torque (tau):   RMSE = 0.779984 Nm
  Weighted RMSE:  0.150387
```

---

## 🎛️ Parameter Guide

### Control Points (M)

Controls spline flexibility and dimension count.

| Parameter | Typical Range | Effect |
|-----------|--------------|--------|
| `M_Q` | 96-160 | Position accuracy (smooth, needs fewer) |
| `M_DQ` | 16-64 | Velocity accuracy (moderate complexity) |
| `M_TAU` | 16-96 | Torque accuracy (high freq, needs more) |

**Total dimensions** = (M_Q + M_DQ + M_TAU) × 7

### Spline Degree

Controls curve smoothness.

| Value | Smoothness | Use Case |
|-------|-----------|----------|
| 3 | C² continuous | Standard, good balance |
| 4 | C³ continuous | Very smooth |
| 5 | C⁴ continuous | Extremely smooth, best for q and tau |

**Recommendation**:
- `DEGREE_Q = 5` (position is very smooth)
- `DEGREE_DQ = 3` (velocity has some jumps)
- `DEGREE_TAU = 5` (torque has large transients, needs smoothing)

### Regularization (λ)

Controls stability vs accuracy trade-off.

| Value | Effect | Use Case |
|-------|--------|----------|
| 1e-7 | Almost no regularization | Maximum accuracy |
| **1e-6** | **Standard** | **Balanced (recommended)** |
| 1e-5 | Light regularization | Suppress oscillations |
| 1e-4 | Moderate regularization | Very stable, some accuracy loss |

**When to increase λ**:
- See oscillations/ringing at endpoints
- Reconstructed curve is too "wiggly"
- Want smoother but less accurate curves

---

## 💡 Tuning Strategies

### Strategy 1: Maximum Accuracy (Current Default)
```python
M_Q = 128, M_DQ = 64, M_TAU = 64
DEGREE_Q = 5, DEGREE_DQ = 3, DEGREE_TAU = 5
LAMBDA_Q = 1e-6, LAMBDA_DQ = 1e-6, LAMBDA_TAU = 1e-5
# Result: 1792 dims, excellent accuracy
```

### Strategy 2: 1k Dimension Budget
```python
M_Q = 110, M_DQ = 16, M_TAU = 16
DEGREE_Q = 5, DEGREE_DQ = 3, DEGREE_TAU = 5
LAMBDA_Q = 1e-6, LAMBDA_DQ = 1e-6, LAMBDA_TAU = 1e-5
# Result: 994 dims, good accuracy
```

### Strategy 3: Extreme Low Dimensions
```python
M_Q = 96, M_DQ = 8, M_TAU = 8
DEGREE_Q = 3, DEGREE_DQ = 3, DEGREE_TAU = 5
LAMBDA_Q = 1e-6, LAMBDA_DQ = 1e-6, LAMBDA_TAU = 1e-4
# Result: 784 dims, acceptable for some tasks
```

### Strategy 4: Focus on Position Only
```python
M_Q = 160, M_DQ = 16, M_TAU = 16
DEGREE_Q = 5, DEGREE_DQ = 3, DEGREE_TAU = 3
LAMBDA_Q = 1e-7, LAMBDA_DQ = 1e-5, LAMBDA_TAU = 1e-4
# Result: Best position accuracy, moderate dq/tau
```

### Strategy 5: Torque-Heavy
```python
M_Q = 96, M_DQ = 16, M_TAU = 96
DEGREE_Q = 3, DEGREE_DQ = 3, DEGREE_TAU = 5
LAMBDA_Q = 1e-6, LAMBDA_DQ = 1e-6, LAMBDA_TAU = 1e-5
# Result: Best torque reconstruction
```

---

## 🔍 What to Look For

### Good Reconstruction
✅ Original and reconstructed curves overlap closely  
✅ Smooth reconstructed curve (no artifacts)  
✅ Low RMSE (< 0.01 for position)  
✅ No oscillations at endpoints  

### Problems and Solutions

#### Problem: High position error
**Solution**: Increase `M_Q` to 128-160

#### Problem: High velocity error
**Solution**: 
- Increase `M_DQ` to 32-64
- Try `DEGREE_DQ = 5` for smoother curves

#### Problem: High torque error
**Solution**:
- Increase `M_TAU` to 64-96
- Use `DEGREE_TAU = 5`
- Increase `LAMBDA_TAU` to 1e-5 or 1e-4

#### Problem: Oscillations at trajectory ends
**Solution**: Increase lambda (1e-5 to 1e-4)

#### Problem: Reconstructed curve is too smooth (missing details)
**Solution**: 
- Increase M (more control points)
- Decrease lambda (less regularization)

#### Problem: Too many dimensions
**Solution**: 
- Reduce M values
- Focus dimensions on most important quantity (usually q)

---

## 📈 Metrics Interpretation

### Position (q) RMSE
- **< 0.001 rad (0.06°)**: Excellent ⭐⭐⭐
- **< 0.01 rad (0.6°)**: Good ⭐⭐
- **< 0.1 rad (6°)**: Acceptable ⭐
- **> 0.1 rad**: Poor ❌

### Velocity (dq) RMSE
- **< 0.1 rad/s**: Excellent ⭐⭐⭐
- **< 0.2 rad/s**: Good ⭐⭐
- **< 0.5 rad/s**: Acceptable ⭐
- **> 0.5 rad/s**: Poor ❌

### Torque (tau) RMSE
- **< 0.5 Nm**: Excellent ⭐⭐⭐
- **< 1.0 Nm**: Good ⭐⭐
- **< 2.0 Nm**: Acceptable ⭐
- **> 2.0 Nm**: Poor ❌

### Weighted RMSE
Combined metric with weights: wq=1.0, wdq=0.25, wtau=0.1
- **< 0.05**: Excellent ⭐⭐⭐
- **< 0.15**: Good ⭐⭐
- **< 0.3**: Acceptable ⭐
- **> 0.3**: Poor ❌

---

## 🔄 Typical Workflow

1. **Start with defaults** (Strategy 1 config)
   ```bash
   python3 quick_tune.py
   ```

2. **Check results**
   - Open the 3 generated plots
   - Look at RMSE values
   - Identify problem areas (q/dq/tau)

3. **Adjust parameters**
   - Edit `quick_tune.py` configuration
   - Focus on the problematic quantity

4. **Re-run and compare**
   ```bash
   python3 quick_tune.py
   ```
   - Plots are auto-overwritten
   - Compare metrics

5. **Iterate** until satisfied

6. **Try different samples**
   - Change `SAMPLE_INDEX`
   - Verify configuration works across samples

7. **Record best configuration**
   - Note down the parameters
   - Use for downstream tasks

---

## 🎯 Tips

### Testing Multiple Configurations Quickly

Create a simple batch script:

```bash
# test_configs.sh
configs=(
    "128 64 64"    # M_Q M_DQ M_TAU
    "110 32 32"
    "96 16 16"
)

for config in "${configs[@]}"; do
    read MQ MDQ MTAU <<< "$config"
    echo "Testing M=($MQ,$MDQ,$MTAU)"
    
    # Modify quick_tune.py with sed or manually
    python3 quick_tune.py
    
    # Save results
    mkdir -p results_M${MQ}_${MDQ}_${MTAU}
    mv *.png results_M${MQ}_${MDQ}_${MTAU}/
done
```

### Auto-open Plots

Set in `quick_tune.py`:
```python
AUTO_OPEN = True
```

### Compare with Original Method

```bash
# Run DCT-PCA for comparison
cd ../../Compress_Test
python3 PCAK96R64_TestVis.py

# Compare plots side-by-side
```

---

## 📝 Notes

- **Auto-overwrite**: Previous plots are replaced each run
- **Fast**: ~5-10 seconds per run (depends on M values)
- **Standalone**: No external config files needed
- **Self-contained**: All parameters in one place

---

## 🆘 Troubleshooting

### Error: "Data file not found"
**Fix**: Update `DATA_PATH` in `quick_tune.py` to correct location

### Error: "Sample index exceeds dataset size"
**Fix**: Reduce `SAMPLE_INDEX` (max = N-1, typically N=2040)

### Plots look blurry
**Fix**: Increase DPI in code (line ~150): `dpi=150` → `dpi=300`

### Memory error
**Fix**: Reduce M values or use smaller sample

---

## 🔗 Related Tools

- `../visualize_reconstruction.py` - Compare multiple strategies
- `../run_sweep_optimized.sh` - Automated grid search
- `../visualize_results.py` - Visualize sweep results

---

**Happy Tuning!** 🎛️✨
